from abc import ABC, abstractmethod

from pydantic import BaseModel


class Conversation(BaseModel):
    id: str
    user_id: str
    create_timestamp: int
    update_timestamp: int
    title: str


class ConversationDatabase(ABC):
    @abstractmethod
    async def create_conversation(self, conversation: Conversation) -> Conversation:
        pass

    @abstractmethod
    async def get_conversations_by_user_id(self, user_id: str) -> list[Conversation]:
        pass

    @abstractmethod
    async def get_conversation_by_id(self, conversation_id: str) -> Conversation:
        pass

    @abstractmethod
    async def update_conversation(self, conversation: Conversation) -> Conversation:
        pass

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        pass
