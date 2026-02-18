from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from conversational_toolkit.chunking.base import Chunk

T = TypeVar("T", bound=Chunk)  # Ensures T is a subclass of Chunk


class Retriever(ABC, Generic[T]):
    def __init__(self, top_k: int):
        self.top_k = top_k

    @abstractmethod
    async def retrieve(self, query: str) -> list[T]:
        pass
