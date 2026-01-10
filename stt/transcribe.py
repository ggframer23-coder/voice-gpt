from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from .settings import Settings
from .offline import apply_offline_env


class TranscriptionError(RuntimeError):
    pass


def _resolve_offline(offline: Optional[bool]) -> bool:
    if offline is not None:
        return offline
    offline_raw = os.environ.get("STT_OFFLINE")
    if offline_raw is None:
        return True
    return offline_raw.strip().lower() in {"1", "true", "yes", "on"}


def convert_audio_ffmpeg(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranscriptionError(result.stderr.strip() or "ffmpeg conversion failed")


def transcribe_audio(
    settings: Settings,
    audio_path: Path,
    model_path: Path,
    output_dir: Optional[Path] = None,
    language: str = "en",
    convert: bool = False,
) -> str:
    if not settings.whisper_bin:
        raise TranscriptionError("STT_WHISPER_BIN is not set")
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    if not model_path.exists():
        raise TranscriptionError(f"Model file not found: {model_path}")

    def run_transcribe(out_dir: Path, input_path: Path) -> str:
        out_prefix = out_dir / "transcript"
        cmd = [
            str(settings.whisper_bin),
            "-m",
            str(model_path),
            "-f",
            str(input_path),
            "-l",
            language,
            "-otxt",
            "-of",
            str(out_prefix),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise TranscriptionError(result.stderr.strip() or "whisper.cpp failed")
        txt_path = out_prefix.with_suffix(".txt")
        if not txt_path.exists():
            raise TranscriptionError("Transcription output not found")
        return txt_path.read_text(encoding="utf-8").strip()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if convert:
            converted = output_dir / "converted.wav"
            convert_audio_ffmpeg(audio_path, converted)
            return run_transcribe(output_dir, converted)
        return run_transcribe(output_dir, audio_path)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        if convert:
            converted = tmp_path / "converted.wav"
            convert_audio_ffmpeg(audio_path, converted)
            return run_transcribe(tmp_path, converted)
        return run_transcribe(tmp_path, audio_path)


def _collect_word_timestamps(segments: list[Any], offset_seconds: float = 0.0) -> list[dict[str, Any]]:
    timestamps: list[dict[str, Any]] = []
    for segment in segments:
        words = getattr(segment, "words", None) or []
        for word in words:
            start = (word.start or 0.0) + offset_seconds
            end = (word.end or 0.0) + offset_seconds
            timestamps.append({"word": word.word, "start": start, "end": end})
    return timestamps


def transcribe_audio_faster_whisper(
    audio_path: Path,
    model_name_or_path: str,
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    convert: bool = False,
    offline: Optional[bool] = None,
    word_timestamps: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    def run_transcribe(input_path: Path) -> tuple[str, list[dict[str, Any]]]:
        model = load_faster_whisper_model(
            model_name_or_path,
            device=device,
            compute_type=compute_type,
            offline=offline,
        )
        segments, _info = model.transcribe(
            str(input_path),
            language=language,
            word_timestamps=word_timestamps,
        )
        segment_list = list(segments)
        lines = [seg.text.strip() for seg in segment_list if seg.text]
        words = _collect_word_timestamps(segment_list) if word_timestamps else []
        return "\n".join(lines).strip(), words

    if not convert:
        return run_transcribe(audio_path)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        converted = tmp_path / "converted.wav"
        convert_audio_ffmpeg(audio_path, converted)
        return run_transcribe(converted)


def load_faster_whisper_model(
    model_name_or_path: str,
    device: str = "cpu",
    compute_type: str = "int8",
    offline: Optional[bool] = None,
):
    offline = _resolve_offline(offline)
    if offline:
        apply_offline_env()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError("faster-whisper is not installed; uv pip install -e '.[faster]'") from exc
    try:
        return WhisperModel(model_name_or_path, device=device, compute_type=compute_type)
    except Exception as exc:
        if offline:
            raise TranscriptionError(
                "Faster-whisper model unavailable offline. Provide a local model path or pre-download the cache."
            ) from exc
        raise


def load_parakeet_model(
    model_name: str,
    model_dir: Optional[Path] = None,
    quantization: Optional[str] = None,
    offline: Optional[bool] = None,
) -> Any:
    offline = _resolve_offline(offline)
    if offline:
        apply_offline_env()
    try:
        import onnx_asr
    except ImportError as exc:
        raise TranscriptionError("onnx-asr is not installed; uv pip install -e '.[parakeet]'") from exc

    try:
        model_path = str(model_dir) if model_dir else None
        return onnx_asr.load_model(model_name, model_path, quantization=quantization)
    except Exception as exc:
        if offline:
            raise TranscriptionError(
                "Parakeet model unavailable offline. Provide STT_PARAKEET_DIR or a local model directory."
            ) from exc
        raise TranscriptionError(f"Parakeet model load failed: {exc}") from exc


def transcribe_audio_parakeet(
    audio_path: Path,
    model_name: str,
    model_dir: Optional[Path] = None,
    quantization: Optional[str] = None,
    convert: bool = False,
    offline: Optional[bool] = None,
) -> str:
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    model = load_parakeet_model(model_name, model_dir=model_dir, quantization=quantization, offline=offline)

    def run_transcribe(input_path: Path) -> str:
        result = model.recognize(str(input_path))
        return result.strip()

    if not convert:
        return run_transcribe(audio_path)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        converted = tmp_path / "converted.wav"
        convert_audio_ffmpeg(audio_path, converted)
        return run_transcribe(converted)


def transcribe_audio_whisperx(
    audio_path: Path,
    model_name: str,
    device: str = "cpu",
    language: str = "en",
    convert: bool = False,
    diarize: bool = False,
    diarize_model: Optional[str] = None,
    offline: Optional[bool] = None,
) -> tuple[str, list[dict[str, Any]], dict]:
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    def run_transcribe(input_path: Path) -> tuple[str, list[dict[str, Any]], dict]:
        try:
            import whisperx
        except ImportError as exc:
            raise TranscriptionError("whisperx is not installed; uv pip install -e '.[whisperx]'") from exc

        offline_mode = _resolve_offline(offline)
        if offline_mode:
            apply_offline_env()

        try:
            model = whisperx.load_model(model_name, device=device)
        except Exception as exc:
            raise TranscriptionError(f"Failed to load WhisperX model '{model_name}': {exc}") from exc

        result = model.transcribe(str(input_path), language=language)
        align_model = whisperx.load_align_model(language_code=result["language"], device=device)
        audio_meta = result.get("audio", {})
        aligned = whisperx.align(
            result["segments"],
            align_model,
            str(input_path),
            audio_meta,
            device=device,
        )
        segments = aligned.get("segments", [])
        text = "\n".join(seg.get("text", "").strip() for seg in segments if seg.get("text"))
        words = _collect_word_timestamps(segments)
        metadata: dict[str, Any] = {
            "whisperx": {
                "model": model_name,
                "language": result.get("language"),
                "segments": segments,
            }
        }

        if diarize:
            diarize_name = diarize_model or "pyannote/speaker-diarization"
            try:
                diarizer = whisperx.load_diarize_model(diarize_name, device=device)
                diarization = diarizer({"uri": input_path.stem, "audio": str(input_path)})
                speaker_segments = []
                for segment, _, label in diarization.itertracks(yield_label=True):
                    speaker_segments.append(
                        {
                            "start": float(segment.start),
                            "end": float(segment.end),
                            "speaker": label,
                        }
                    )
                metadata["whisperx"]["diarization"] = speaker_segments
            except Exception as exc:
                raise TranscriptionError(f"WhisperX diarization failed: {exc}") from exc

        return text, words, metadata

    if not convert:
        return run_transcribe(audio_path)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        converted = tmp_path / "converted.wav"
        convert_audio_ffmpeg(audio_path, converted)
        return run_transcribe(converted)
