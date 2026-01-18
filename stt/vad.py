from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from .transcribe import TranscriptionError


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_vad_bin() -> Path:
    return _project_root() / "third_party/whisper.cpp/build/bin/vad-speech-segments"


def _default_vad_model() -> Path:
    return _project_root() / "third_party/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin"


def _ensure_file(path: Path, desc: str) -> None:
    if not path.exists():
        raise TranscriptionError(f"{desc} not found: {path}")


def _slice_audio_ffmpeg(input_path: Path, output_path: Path, start_ms: int, end_ms: int) -> None:
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        return
    start_sec = start_ms / 1000.0
    duration_sec = duration_ms / 1000.0
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
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
        raise TranscriptionError(result.stderr.strip() or "ffmpeg clip failed")


def _run_vad_segments(
    audio_path: Path,
    vad_bin: Path,
    vad_model: Path,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    max_speech_s: Optional[float],
    speech_pad_ms: int,
    samples_overlap: float,
) -> list[tuple[float, float]]:
    cmd = [
        str(vad_bin),
        "--file",
        str(audio_path),
        "--vad-model",
        str(vad_model),
        "--vad-threshold",
        str(threshold),
        "--vad-min-speech-duration-ms",
        str(min_speech_ms),
        "--vad-min-silence-duration-ms",
        str(min_silence_ms),
        "--vad-speech-pad-ms",
        str(speech_pad_ms),
        "--vad-samples-overlap",
        str(samples_overlap),
    ]
    if max_speech_s is not None:
        cmd += ["--vad-max-speech-duration-s", str(max_speech_s)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranscriptionError(result.stderr.strip() or "VAD segmentation failed")

    segments = []
    pattern = re.compile(r"Speech segment \d+: start = ([0-9.]+), end = ([0-9.]+)")
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        start_s = float(match.group(1))
        end_s = float(match.group(2))
        if end_s <= start_s:
            continue
        segments.append((start_s, end_s))
    return segments


def create_vad_clips(
    audio_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 30,
    max_speech_s: Optional[float] = None,
    samples_overlap: float = 0.1,
    vad_bin: Optional[Path] = None,
    vad_model: Optional[Path] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    vad_bin = vad_bin or _default_vad_bin()
    vad_model = vad_model or _default_vad_model()
    _ensure_file(vad_bin, "VAD binary")
    _ensure_file(vad_model, "VAD model")

    segments = _run_vad_segments(
        audio_path=audio_path,
        vad_bin=vad_bin,
        vad_model=vad_model,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        max_speech_s=max_speech_s,
        speech_pad_ms=speech_pad_ms,
        samples_overlap=samples_overlap,
    )

    clip_entries = []
    for idx, (start_s, end_s) in enumerate(segments):
        start_ms = int(round(start_s * 1000))
        end_ms = int(round(end_s * 1000))
        if end_ms <= start_ms:
            continue
        clip_path = output_dir / f"clip_{idx:04d}.wav"
        _slice_audio_ffmpeg(audio_path, clip_path, start_ms, end_ms)
        clip_entries.append(
            {
                "index": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "clip_path": str(clip_path),
            }
        )

    return {
        "source_audio": str(audio_path),
        "sample_rate": 16000,
        "segments": clip_entries,
        "vad": {
            "threshold": threshold,
            "min_speech_ms": min_speech_ms,
            "min_silence_ms": min_silence_ms,
            "speech_pad_ms": speech_pad_ms,
            "max_speech_s": max_speech_s,
            "samples_overlap": samples_overlap,
            "vad_bin": str(vad_bin),
            "vad_model": str(vad_model),
        },
    }
