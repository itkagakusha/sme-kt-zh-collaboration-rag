from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Any, Generator

metadata_context: ContextVar[Optional[list[dict[str, Any]]]] = ContextVar("metadata", default=None)


class MetadataProvider:
    @staticmethod
    @contextmanager
    def get_manager() -> Generator[None, Any, None]:
        metadata_context.set([])
        try:
            yield
        finally:
            metadata_context.set([])

    @staticmethod
    def add_metadata(metadata: dict[str, Any]) -> None:
        prev = metadata_context.get()
        if prev is None:
            prev = []
        prev.append(metadata)
        metadata_context.set(prev)

    @staticmethod
    def get_metadata() -> list[dict[str, Any]]:
        return metadata_context.get() or []
