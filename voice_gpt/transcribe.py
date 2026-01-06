from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from .settings import Settings


class TranscriptionError(RuntimeError):
    pass


def convert_audio_ffmpeg(input_path: Path, output_path: Path, vad: bool = False) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if vad:
        cmd += [
            "-af",
            "silenceremove=start_periods=1:start_silence=0.2:start_threshold=-35dB:"
            "stop_periods=1:stop_silence=0.2:stop_threshold=-35dB",
        ]
    cmd += [
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
    vad: bool = False,
) -> str:
    if not settings.whisper_bin:
        raise TranscriptionError("VOICE_GPT_WHISPER_BIN is not set")
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
        if convert or vad:
            converted = output_dir / "converted.wav"
            convert_audio_ffmpeg(audio_path, converted, vad=vad)
            return run_transcribe(output_dir, converted)
        return run_transcribe(output_dir, audio_path)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        if convert or vad:
            converted = tmp_path / "converted.wav"
            convert_audio_ffmpeg(audio_path, converted, vad=vad)
            return run_transcribe(tmp_path, converted)
        return run_transcribe(tmp_path, audio_path)


def transcribe_audio_faster_whisper(
    audio_path: Path,
    model_name_or_path: str,
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    convert: bool = False,
    vad: bool = False,
) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError("faster-whisper is not installed; uv pip install -e '.[faster]'") from exc

    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    def run_transcribe(input_path: Path) -> str:
        model = WhisperModel(model_name_or_path, device=device, compute_type=compute_type)
        segments, _info = model.transcribe(str(input_path), language=language)
        lines = [seg.text.strip() for seg in segments if seg.text]
        return "\n".join(lines).strip()

    if not convert and not vad:
        return run_transcribe(audio_path)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        converted = tmp_path / "converted.wav"
        convert_audio_ffmpeg(audio_path, converted, vad=vad)
        return run_transcribe(converted)
