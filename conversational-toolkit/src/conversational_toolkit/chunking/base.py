from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    title: str
    content: str
    mime_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunker(ABC):
    @abstractmethod
    def make_chunks(self, *args: Any, **kwargs: Any) -> list[Chunk]:
        pass
