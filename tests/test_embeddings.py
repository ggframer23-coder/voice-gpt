import sys
import types

import numpy as np

import stt.embeddings as embeddings


def _install_fake_sentence_transformer(class_obj) -> None:
    module = types.SimpleNamespace(SentenceTransformer=class_obj)
    sys.modules["sentence_transformers"] = module


def test_load_model_offline_local_files(monkeypatch) -> None:
    embeddings.load_model.cache_clear()
    called = {"offline": False}

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    def fake_apply_offline() -> None:
        called["offline"] = True

    _install_fake_sentence_transformer(FakeSentenceTransformer)
    monkeypatch.setattr(embeddings, "apply_offline_env", fake_apply_offline)

    model = embeddings.load_model("fake-model", offline=True)

    assert called["offline"] is True
    assert model.kwargs["local_files_only"] is True


def test_load_model_fallback_on_typeerror(monkeypatch) -> None:
    embeddings.load_model.cache_clear()
    call_kwargs = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            call_kwargs.append(kwargs)
            if "local_files_only" in kwargs:
                raise TypeError("unsupported")

    _install_fake_sentence_transformer(FakeSentenceTransformer)

    model = embeddings.load_model("fake-model", offline=True)

    assert model is not None
    assert call_kwargs == [{"local_files_only": True}, {}]


def test_load_model_offline_error(monkeypatch) -> None:
    embeddings.load_model.cache_clear()

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("no model")

    _install_fake_sentence_transformer(FakeSentenceTransformer)

    try:
        embeddings.load_model("fake-model", offline=True)
    except RuntimeError as exc:
        assert "Embedding model unavailable offline" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_embed_texts_uses_encoder(monkeypatch) -> None:
    class FakeModel:
        def encode(self, *args, **kwargs):
            return np.array([[1.0, 2.0], [3.0, 4.0]])

    monkeypatch.setattr(embeddings, "load_model", lambda *_args, **_kwargs: FakeModel())

    result = embeddings.embed_texts("fake-model", ["a", "b"], offline=True)

    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_load_model_online(monkeypatch) -> None:
    embeddings.load_model.cache_clear()

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

    _install_fake_sentence_transformer(FakeSentenceTransformer)

    model = embeddings.load_model("fake-model", offline=False)

    assert model.kwargs == {}
