from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, TYPE_CHECKING

from .offline import apply_offline_env

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_model(model_name: str, offline: bool = True) -> "SentenceTransformer":
    if offline:
        apply_offline_env()
    from sentence_transformers import SentenceTransformer

    try:
        if offline:
            return SentenceTransformer(model_name, local_files_only=True)
        return SentenceTransformer(model_name)
    except TypeError:
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:
            if offline:
                raise RuntimeError(
                    f"Embedding model unavailable offline: {model_name}. Set STT_EMBED_MODEL to a local path or "
                    "pre-download the model cache."
                ) from exc
            raise
    except Exception as exc:
        if offline:
            raise RuntimeError(
                f"Embedding model unavailable offline: {model_name}. Set STT_EMBED_MODEL to a local path or "
                "pre-download the model cache."
            ) from exc
        raise


def embed_texts(model_name: str, texts: Iterable[str], offline: bool = True) -> List[list[float]]:
    model = load_model(model_name, offline=offline)
    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()
