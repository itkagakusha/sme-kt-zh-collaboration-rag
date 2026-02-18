import json
from typing import Optional, Literal, AsyncGenerator, Any

from loguru import logger
from openai.types.chat import completion_create_params

from conversational_toolkit.llms.base import LLM, LLMMessage, ToolCall, Roles, Function
from openai import AsyncOpenAI

from conversational_toolkit.tools.base import Tool
from conversational_toolkit.utils.metadata_provider import MetadataProvider


def message_to_openai(msg: LLMMessage) -> dict[str, Any]:
    message = {
        "role": msg.role.value,
        "content": msg.content,
    }

    if msg.name:
        message["name"] = msg.name

    if msg.role == Roles.TOOL and msg.tool_call_id:
        message["tool_call_id"] = msg.tool_call_id

    if msg.role == Roles.ASSISTANT and msg.tool_calls:
        message["tool_calls"] = [  # type: ignore
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    return message


class OpenAILLM(LLM):
    def __init__(  # noqa: PLR0913
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.5,
        seed: int = 42,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[Literal["none", "auto", "required"]] = None,
        response_format: Optional[completion_create_params.ResponseFormat] = None,
        openai_api_key: Optional[str] = None,
    ):
        super().__init__()
        if response_format is None:
            response_format = {"type": "text"}

        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model_name
        self.temperature = temperature
        self.seed = seed
        self.tools = tools
        self.tool_choice = tool_choice

        self.response_format = response_format
        logger.debug(
            f"OpenAI LLM loaded: {model_name}; temperature: {temperature}; seed: {seed}; tools: {tools}; tool_choice: {tool_choice}; response_format: {response_format}"
        )

    async def generate(self, conversation: list[LLMMessage]) -> LLMMessage:
        """
        Generate a completion for the given conversation.
        Args:
            conversation:

        Returns:

        """
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[message_to_openai(msg) for msg in conversation],  # type: ignore
            temperature=self.temperature,
            seed=self.seed,
            tools=[tool.json_schema() for tool in self.tools] if self.tools else None,  # type: ignore
            tool_choice=self.tool_choice,  # type: ignore
            response_format=self.response_format,
        )
        logger.debug(f"Completion: {completion}")
        logger.info(f"LLM Usage: {completion.usage}")

        MetadataProvider.add_metadata(
            {
                "model": completion.model,
                "usage": completion.usage.to_dict() if completion.usage else {},
            }
        )

        return LLMMessage(
            content=completion.choices[0].message.content or "",
            role=Roles(completion.choices[0].message.role),
            tool_calls=[
                ToolCall(
                    id=tc.id,
                    function=Function(name=tc.function.name, arguments=json.loads(tc.function.arguments)),
                    type="function",
                )
                for tc in completion.choices[0].message.tool_calls
            ]
            if completion.choices[0].message.tool_calls
            else [],
        )

    async def generate_stream(self, conversation: list[LLMMessage]) -> AsyncGenerator[LLMMessage, None]:
        response = await self.client.chat.completions.create(  # type: ignore
            model=self.model,
            messages=[message_to_openai(msg) for msg in conversation],
            temperature=self.temperature,
            seed=self.seed,
            tools=[tool.json_schema() for tool in self.tools] if self.tools else None,
            tool_choice=self.tool_choice,
            stream=True,
            response_format=self.response_format,
            stream_options={"include_usage": True},
        )

        parsed_tool_calls: list[ToolCall] = []

        async for chunk in response:
            if not chunk.choices:  # Last chunk has empty choices list
                continue
            if chunk.choices[0].delta.content:
                yield LLMMessage(
                    content=chunk.choices[0].delta.content,
                )
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunk_list = chunk.choices[0].delta.tool_calls
                for tool_call_chunk in tool_call_chunk_list:
                    if len(parsed_tool_calls) <= tool_call_chunk.index:
                        if len(parsed_tool_calls) > 0:
                            yield LLMMessage(
                                tool_calls=[parsed_tool_calls[-1]],
                            )
                        parsed_tool_calls.append(
                            ToolCall(id="", type="function", function=Function(name="", arguments=""))
                        )
                    tool_call = parsed_tool_calls[tool_call_chunk.index]
                    if tool_call_chunk.id:
                        tool_call.id += tool_call_chunk.id
                    if tool_call_chunk.function.name:
                        tool_call.function.name += tool_call_chunk.function.name
                    if tool_call_chunk.function.arguments:
                        tool_call.function.arguments += tool_call_chunk.function.arguments

        MetadataProvider.add_metadata(
            {
                "model": chunk.model,
                "usage": chunk.usage.to_dict() if chunk.usage else {},
            }
        )

        if parsed_tool_calls:
            yield LLMMessage(
                tool_calls=[parsed_tool_calls[-1]],
            )
