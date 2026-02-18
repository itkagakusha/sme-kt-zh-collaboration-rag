from abc import ABC, abstractmethod
from typing import Any, TypedDict, Literal


class FunctionDescription(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolDescription(TypedDict):
    type: Literal["function"]
    function: FunctionDescription


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        pass

    def json_schema(self) -> ToolDescription:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
