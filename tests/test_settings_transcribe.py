import sys
from types import SimpleNamespace

import pytest

import stt.settings as settings
import stt.torch_compat as torch_compat
import stt.transcribe as transcribe


def test_load_settings_from_env(monkeypatch, tmp_path) -> None:
    base_dir = tmp_path / "base"
    monkeypatch.setenv("STT_HOME", str(base_dir))
    monkeypatch.setenv("STT_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("STT_INDEX", str(tmp_path / "index.faiss"))
    monkeypatch.setenv("STT_INDEX_BACKEND", "chroma")
    monkeypatch.setenv("STT_EMBED_MODEL", "fake-model")
    monkeypatch.setenv("STT_WHISPER_BIN", "/bin/whisper")
    monkeypatch.setenv("STT_VAD_BIN", "/bin/vad")
    monkeypatch.setenv("STT_VAD_MODEL", "/bin/model")
    monkeypatch.setenv("STT_PARAKEET_MODEL", "parakeet")
    monkeypatch.setenv("STT_PARAKEET_DIR", str(tmp_path / "parakeet"))
    monkeypatch.setenv("STT_PARAKEET_QUANT", "int8")
    monkeypatch.setenv("STT_OFFLINE", "false")

    loaded = settings.load_settings()

    assert loaded.base_dir == base_dir
    assert loaded.index_backend == "chroma"
    assert loaded.model_name == "fake-model"
    assert loaded.offline is False
    assert str(loaded.whisper_bin) == "/bin/whisper"


def test_patch_torch_load_forces_weights_only(monkeypatch) -> None:
    calls = {}

    def fake_load(*_args, **kwargs):
        calls.update(kwargs)
        return "ok"

    fake_torch = SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    torch_compat.patch_torch_load()
    result = fake_torch.load("x", weights_only=True)

    assert result == "ok"
    assert calls["weights_only"] is False


def test_patch_torch_load_import_error(monkeypatch) -> None:
    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    torch_compat.patch_torch_load()


def test_resolve_offline_env(monkeypatch) -> None:
    monkeypatch.delenv("STT_OFFLINE", raising=False)
    assert transcribe._resolve_offline(None) is True
    monkeypatch.setenv("STT_OFFLINE", "no")
    assert transcribe._resolve_offline(None) is False
    assert transcribe._resolve_offline(True) is True


def test_collect_word_timestamps_handles_dict_and_object() -> None:
    obj_word = SimpleNamespace(start=1.0, end=2.0, word="hi")
    obj_segment = SimpleNamespace(words=[obj_word])
    dict_segment = {"words": [{"start": 0.5, "end": 1.0, "word": "hey"}]}

    result = transcribe._collect_word_timestamps([dict_segment, obj_segment], offset_seconds=1.0)

    assert result[0]["word"] == "hey"
    assert result[0]["start"] == 1.5
    assert result[1]["word"] == "hi"
    assert result[1]["end"] == 3.0


def test_convert_audio_ffmpeg_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        transcribe.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="nope"),
    )
    with pytest.raises(transcribe.TranscriptionError):
        transcribe.convert_audio_ffmpeg(tmp_path / "in.wav", tmp_path / "out.wav")
