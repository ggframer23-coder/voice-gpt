from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(model_name: str, texts: Iterable[str]) -> List[list[float]]:
    model = load_model(model_name)
    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()
