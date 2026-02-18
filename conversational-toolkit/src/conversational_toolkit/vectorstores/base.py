from abc import ABC, abstractmethod
from typing import Union, Any

import numpy as np
from numpy.typing import NDArray

from conversational_toolkit.chunking.base import Chunk


class ChunkRecord(Chunk):
    id: str
    embedding: list[float]


class ChunkMatch(ChunkRecord):
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def insert_chunks(self, chunks: list[Chunk], embedding: NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    async def get_chunks_by_embedding(
        self, embedding: NDArray[np.float64], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[ChunkMatch]:
        pass

    @abstractmethod
    async def get_chunks_by_ids(self, chunk_ids: Union[int, list[int]]) -> list[Chunk]:
        pass
