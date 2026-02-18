from abc import ABC, abstractmethod
from typing import Optional, Any

from pydantic import BaseModel

from conversational_toolkit.llms.base import Roles


class Message(BaseModel):
    id: str
    user_id: Optional[str]
    conversation_id: str
    content: str
    role: Roles
    create_timestamp: int
    parent_id: Optional[str] = None
    metadata: Optional[list[dict[str, Any]]] = None


class MessageDatabase(ABC):
    @abstractmethod
    async def create_message(
        self,
        message: Message,
    ) -> Message:
        pass

    @abstractmethod
    async def get_messages_by_conversation_id(self, conversation_id: str) -> list[Message]:
        pass

    @abstractmethod
    async def get_message_by_id(self, conversation_id: str) -> Message:
        pass

    @abstractmethod
    async def delete_message(self, message_id: str) -> bool:
        pass
