from pathlib import Path
from types import SimpleNamespace

import pytest

import stt.vad as vad
from stt.transcribe import TranscriptionError


def test_default_paths() -> None:
    project_root = vad._project_root()
    assert (project_root / "third_party").exists() or project_root.exists()
    assert "vad-speech-segments" in str(vad._default_vad_bin())
    assert "for-tests-silero" in str(vad._default_vad_model())


def test_slice_audio_noop(monkeypatch, tmp_path) -> None:
    called = {"ran": False}

    def fake_run(*_args, **_kwargs):
        called["ran"] = True
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(vad.subprocess, "run", fake_run)
    vad._slice_audio_ffmpeg(tmp_path / "in.wav", tmp_path / "out.wav", start_ms=100, end_ms=100)
    assert called["ran"] is False


def test_ensure_file_raises(tmp_path) -> None:
    with pytest.raises(TranscriptionError):
        vad._ensure_file(tmp_path / "missing.bin", "VAD binary")


def test_run_vad_segments_parses_output(monkeypatch, tmp_path) -> None:
    output = "\n".join(
        [
            "Speech segment 0: start = 0.10, end = 1.20",
            "Speech segment 1: start = 2.00, end = 1.00",
        ]
    )

    monkeypatch.setattr(
        vad.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    segments = vad._run_vad_segments(
        audio_path=tmp_path / "sample.wav",
        vad_bin=tmp_path / "vad",
        vad_model=tmp_path / "model.bin",
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        max_speech_s=None,
        speech_pad_ms=30,
        samples_overlap=0.1,
    )

    assert segments == [(0.1, 1.2)]


def test_run_vad_segments_raises_on_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        vad.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad"),
    )

    with pytest.raises(TranscriptionError):
        vad._run_vad_segments(
            audio_path=tmp_path / "sample.wav",
            vad_bin=tmp_path / "vad",
            vad_model=tmp_path / "model.bin",
            threshold=0.5,
            min_speech_ms=250,
            min_silence_ms=100,
            max_speech_s=None,
            speech_pad_ms=30,
            samples_overlap=0.1,
        )


def test_create_vad_clips(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_text("x", encoding="utf-8")
    output_dir = tmp_path / "out"

    monkeypatch.setattr(vad, "_ensure_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vad, "_run_vad_segments", lambda *_args, **_kwargs: [(0.0, 0.5)])
    slices = []

    def fake_slice(_in_path: Path, out_path: Path, start_ms: int, end_ms: int) -> None:
        slices.append((out_path, start_ms, end_ms))
        out_path.write_text("clip", encoding="utf-8")

    monkeypatch.setattr(vad, "_slice_audio_ffmpeg", fake_slice)

    metadata = vad.create_vad_clips(
        audio_path=audio_path,
        output_dir=output_dir,
        threshold=0.7,
        min_speech_ms=200,
        min_silence_ms=50,
        speech_pad_ms=20,
        max_speech_s=None,
        samples_overlap=0.2,
        vad_bin=tmp_path / "vad",
        vad_model=tmp_path / "model.bin",
    )

    assert metadata["source_audio"] == str(audio_path)
    assert len(metadata["segments"]) == 1
    assert slices[0][1:] == (0, 500)
