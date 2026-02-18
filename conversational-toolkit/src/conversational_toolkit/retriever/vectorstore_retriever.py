from conversational_toolkit.embeddings.base import EmbeddingsModel
from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.vectorstores.base import VectorStore, ChunkMatch


class VectorStoreRetriever(Retriever[ChunkMatch]):
    def __init__(self, embedding_model: EmbeddingsModel, vector_store: VectorStore, top_k: int):
        super().__init__(top_k)
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    async def retrieve(self, query: str) -> list[ChunkMatch]:
        embeddings = await self.embedding_model.get_embeddings(query)
        results = await self.vector_store.get_chunks_by_embedding(embeddings[0], self.top_k)
        return results
