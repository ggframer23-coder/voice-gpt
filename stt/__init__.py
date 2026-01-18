"""stt package."""

from .settings import Settings

__all__ = ["Settings"]


def __getattr__(name: str):
    """Lazy import heavy modules only when accessed."""
    if name == "add_entry":
        from .journal import add_entry
        return add_entry
    if name == "search":
        from .journal import search
        return search
    if name == "transcribe_audio":
        from .transcribe import transcribe_audio
        return transcribe_audio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
