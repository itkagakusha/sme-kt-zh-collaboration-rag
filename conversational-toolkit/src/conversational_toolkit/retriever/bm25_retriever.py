from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.retriever.base import Retriever


class BM25Retriever(Retriever[Chunk]):
    def __init__(self, top_k: int) -> None:
        super().__init__(top_k)
        raise NotImplementedError

    async def retrieve(self, query: str) -> list[Chunk]:
        raise NotImplementedError
