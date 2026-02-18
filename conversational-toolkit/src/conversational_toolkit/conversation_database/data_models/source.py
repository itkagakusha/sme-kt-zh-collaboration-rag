from abc import ABC, abstractmethod

from pydantic import BaseModel


class Source(BaseModel):
    id: str
    message_id: str
    content: str
    metadata: dict[str, float | int | str | None]


class SourceDatabase(ABC):
    @abstractmethod
    async def create_source(self, source: Source) -> Source:
        pass

    @abstractmethod
    async def get_sources_by_message_id(self, message_id: str) -> list[Source]:
        pass

    @abstractmethod
    async def delete_sources(self, source_ids: list[str]) -> bool:
        pass
