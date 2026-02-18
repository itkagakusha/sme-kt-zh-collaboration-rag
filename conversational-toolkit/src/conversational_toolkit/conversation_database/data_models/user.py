from abc import ABC, abstractmethod

from pydantic import BaseModel


class User(BaseModel):
    id: str


class UserDatabase(ABC):
    @abstractmethod
    async def create_user(self, user: User) -> User:
        pass

    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> User | None:
        pass
