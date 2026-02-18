from abc import ABC, abstractmethod
from fastapi import Request, FastAPI


class AuthProvider(ABC):
    @abstractmethod
    def get_current_user_id(self, request: Request) -> str:
        """Dependency to retrieve the current user"""
        pass

    @abstractmethod
    def bind_to_app(self, app: FastAPI) -> None:
        """Bind anything to the app, like routes or middleware. The frontend expect /auth/refresh and /auth/login to be available"""
        pass
