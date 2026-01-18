from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import stt.transcribe as transcribe


def _make_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(whisper_bin=tmp_path / "whisper-bin")


def _fake_run_factory(content: str):
    def _fake_run(cmd, capture_output, text):  # noqa: ANN001
        out_prefix = None
        for idx, token in enumerate(cmd):
            if token == "-of" and idx + 1 < len(cmd):
                out_prefix = Path(cmd[idx + 1])
                break
        if out_prefix is None:
            raise AssertionError("missing -of output prefix")
        out_prefix.with_suffix(".txt").write_text(content, encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    return _fake_run


def test_transcribe_audio_whispercpp(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    model = tmp_path / "ggml-base.bin"
    audio.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    settings = _make_settings(tmp_path)

    monkeypatch.setattr(transcribe.subprocess, "run", _fake_run_factory("hello"))
    text = transcribe.transcribe_audio(settings, audio, model, convert=False)
    assert text == "hello"


def test_transcribe_audio_whispercpp_convert(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    model = tmp_path / "ggml-base.bin"
    audio.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    settings = _make_settings(tmp_path)

    def _fake_convert(_src: Path, dst: Path) -> None:
        dst.write_text("converted", encoding="utf-8")

    monkeypatch.setattr(transcribe, "convert_audio_ffmpeg", _fake_convert)
    monkeypatch.setattr(transcribe.subprocess, "run", _fake_run_factory("hello"))
    text = transcribe.transcribe_audio(settings, audio, model, convert=True)
    assert text == "hello"


def test_transcribe_audio_requires_whisper_bin(tmp_path):
    audio = tmp_path / "sample.wav"
    model = tmp_path / "ggml-base.bin"
    audio.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    settings = SimpleNamespace(whisper_bin=None)

    with pytest.raises(transcribe.TranscriptionError):
        transcribe.transcribe_audio(settings, audio, model, convert=False)


def test_transcribe_audio_missing_output(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    model = tmp_path / "ggml-base.bin"
    audio.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    settings = _make_settings(tmp_path)

    def _fake_run(cmd, capture_output, text):  # noqa: ANN001
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(transcribe.subprocess, "run", _fake_run)

    with pytest.raises(transcribe.TranscriptionError):
        transcribe.transcribe_audio(settings, audio, model, convert=False)


def test_transcribe_audio_faster_whisper(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_text("x", encoding="utf-8")

    word = SimpleNamespace(word="hi", start=0.0, end=1.0)
    segment = SimpleNamespace(text="hi", words=[word])
    model = SimpleNamespace(transcribe=lambda *_args, **_kwargs: ([segment], None))
    monkeypatch.setattr(transcribe, "load_faster_whisper_model", lambda *_args, **_kwargs: model)

    text, words = transcribe.transcribe_audio_faster_whisper(
        audio_path=audio,
        model_name_or_path="base.en",
        convert=False,
    )
    assert text == "hi"
    assert words[0]["word"] == "hi"


def test_transcribe_audio_faster_whisper_convert(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_text("x", encoding="utf-8")

    def _fake_convert(_src: Path, dst: Path) -> None:
        dst.write_text("converted", encoding="utf-8")

    model = SimpleNamespace(transcribe=lambda *_args, **_kwargs: ([], None))
    monkeypatch.setattr(transcribe, "convert_audio_ffmpeg", _fake_convert)
    monkeypatch.setattr(transcribe, "load_faster_whisper_model", lambda *_args, **_kwargs: model)

    text, words = transcribe.transcribe_audio_faster_whisper(
        audio_path=audio,
        model_name_or_path="base.en",
        convert=True,
    )
    assert text == ""
    assert words == []


def test_transcribe_audio_parakeet(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_text("x", encoding="utf-8")

    model = SimpleNamespace(recognize=lambda *_args, **_kwargs: "parakeet")
    monkeypatch.setattr(transcribe, "load_parakeet_model", lambda *_args, **_kwargs: model)

    text = transcribe.transcribe_audio_parakeet(
        audio_path=audio,
        model_name="nemo-parakeet-tdt-0.6b-v3",
        convert=False,
    )
    assert text == "parakeet"


def test_transcribe_audio_parakeet_convert(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_text("x", encoding="utf-8")

    def _fake_convert(_src: Path, dst: Path) -> None:
        dst.write_text("converted", encoding="utf-8")

    model = SimpleNamespace(recognize=lambda *_args, **_kwargs: "ok")
    monkeypatch.setattr(transcribe, "convert_audio_ffmpeg", _fake_convert)
    monkeypatch.setattr(transcribe, "load_parakeet_model", lambda *_args, **_kwargs: model)

    text = transcribe.transcribe_audio_parakeet(
        audio_path=audio,
        model_name="nemo-parakeet-tdt-0.6b-v3",
        convert=True,
    )
    assert text == "ok"


def test_transcribe_audio_whisperx_with_diarize(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_text("x", encoding="utf-8")

    word = {"word": "test", "start": 0.0, "end": 0.5}
    segment = {"text": "test", "words": [word]}
    wx_model = SimpleNamespace(transcribe=lambda *_args, **_kwargs: {"segments": [segment], "language": "en"})

    diarization_segment = SimpleNamespace(start=0.0, end=1.0)
    diarization = SimpleNamespace(itertracks=lambda yield_label: [(diarization_segment, None, "SPEAKER_0")])
    diarizer = lambda _data: diarization

    dummy_whisperx = SimpleNamespace(
        load_model=lambda *_args, **_kwargs: wx_model,
        load_align_model=lambda *_args, **_kwargs: ("align_model", "align_metadata"),
        align=lambda *_args, **_kwargs: {"segments": [segment]},
        load_diarize_model=lambda *_args, **_kwargs: diarizer,
    )

    sys.modules["whisperx"] = dummy_whisperx
    try:
        text, words, metadata = transcribe.transcribe_audio_whisperx(
            audio_path=audio,
            model_name="medium",
            device="cpu",
            convert=False,
            diarize=True,
        )
    finally:
        sys.modules.pop("whisperx", None)

    assert text == "test"
    assert words[0]["word"] == "test"
    assert metadata["whisperx"]["model"] == "medium"
    assert metadata["whisperx"]["diarization"][0]["speaker"] == "SPEAKER_0"


def test_transcribe_audio_whisperx_no_diarize(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_text("x", encoding="utf-8")

    segment = {"text": "ok", "words": []}
    wx_model = SimpleNamespace(transcribe=lambda *_args, **_kwargs: {"segments": [segment], "language": "en"})

    dummy_whisperx = SimpleNamespace(
        load_model=lambda *_args, **_kwargs: wx_model,
        load_align_model=lambda *_args, **_kwargs: ("align_model", "align_metadata"),
        align=lambda *_args, **_kwargs: {"segments": [segment]},
    )

    sys.modules["whisperx"] = dummy_whisperx
    try:
        text, words, metadata = transcribe.transcribe_audio_whisperx(
            audio_path=audio,
            model_name="medium",
            device="cpu",
            convert=False,
            diarize=False,
        )
    finally:
        sys.modules.pop("whisperx", None)

    assert text == "ok"
    assert words == []
    assert "diarization" not in metadata["whisperx"]


def test_load_faster_whisper_model_offline_error(monkeypatch):
    called = {"offline": False}

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("no model")

    def fake_offline():
        called["offline"] = True

    sys.modules["faster_whisper"] = SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setattr(transcribe, "apply_offline_env", fake_offline)

    try:
        with pytest.raises(transcribe.TranscriptionError):
            transcribe.load_faster_whisper_model("base.en", offline=True)
    finally:
        sys.modules.pop("faster_whisper", None)

    assert called["offline"] is True


def test_load_parakeet_model_import_error(monkeypatch):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "onnx_asr":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(transcribe.TranscriptionError):
        transcribe.load_parakeet_model("parakeet", offline=True)


def test_transcribe_audio_whisperx_import_error(monkeypatch, tmp_path):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "whisperx":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    audio = tmp_path / "sample.wav"
    audio.write_text("x", encoding="utf-8")

    with pytest.raises(transcribe.TranscriptionError):
        transcribe.transcribe_audio_whisperx(
            audio_path=audio,
            model_name="medium",
            device="cpu",
            convert=False,
            diarize=False,
        )
