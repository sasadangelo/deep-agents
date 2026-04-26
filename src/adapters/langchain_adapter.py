"""
LangChain adapter for our ChatModel.

Questo adapter fa l'opposto di quello che fa beeai-framework:
- BeeAI: wrappa BaseChatModel di LangChain per usarlo nel loro sistema
- Noi: wrappiamo il nostro ChatModel per usarlo con LangChain/LangGraph
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Iterator

from langchain_core.callbacks import (AsyncCallbackManagerForLLMRun,
                                      CallbackManagerForLLMRun)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.outputs import (ChatGeneration, ChatGenerationChunk,
                                    ChatResult)
from langchain_core.tools import BaseTool

from backend.chat import ChatModel, Message


class LangChainChatModelAdapter(BaseChatModel):
    """
    Adapter che rende il nostro ChatModel compatibile con LangChain.

    Permette di usare il nostro sistema con create_deep_agent e LangGraph.
    """

    chat_model: ChatModel
    """Il nostro ChatModel da wrappare."""

    bound_tools: list[Any] = []
    """Tools bound via LangGraph/LangChain."""

    class Config:
        """Configurazione Pydantic."""
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Tipo del modello."""
        return f"{self.chat_model.get_provider()}_adapter"

    def _tool_name(self, tool: Any) -> str:
        """Extract tool name from LangChain tool-like objects."""
        if isinstance(tool, BaseTool):
            return tool.name
        if hasattr(tool, "name"):
            return str(tool.name)
        if callable(tool) and hasattr(tool, "__name__"):
            return tool.__name__
        raise ValueError(f"Cannot determine tool name for {tool!r}")

    def _tool_description(self, tool: Any) -> str:
        """Extract tool description for prompt injection."""
        if isinstance(tool, BaseTool):
            return tool.description or ""
        if hasattr(tool, "description"):
            return str(tool.description or "")
        if callable(tool) and tool.__doc__:
            return tool.__doc__.strip()
        return ""

    def _build_tool_system_prompt(self) -> str:
        """Build tool instructions for non-native tool-calling models."""
        if not self.bound_tools:
            return ""

        lines = [
            "You can call external tools.",
            "When you need a tool, respond ONLY with valid JSON.",
            'Schema: {"tool_name":"<tool name>","arguments":{...}}',
            "Do not wrap the JSON in markdown fences.",
            "Available tools:",
        ]
        for tool in self.bound_tools:
            lines.append(
                f"- {self._tool_name(tool)}: {self._tool_description(tool)}"
            )
        return "\n".join(lines)

    def _extract_tool_call(self, content: str) -> dict[str, Any] | None:
        """Parse a tool call emitted as JSON text."""
        text = content.strip()
        if not text:
            return None

        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})

        if not isinstance(tool_name, str):
            function_data = data.get("function")
            if isinstance(function_data, dict):
                tool_name = function_data.get("name")
                arguments = function_data.get("parameters", {})
            else:
                tool_name = data.get("name")
                arguments = data.get("parameters", arguments)

        if not isinstance(tool_name, str):
            return None

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None

        if not isinstance(arguments, dict):
            return None

        return {
            "id": f"call_{tool_name}",
            "name": tool_name,
            "args": arguments,
            "type": "tool_call",
        }

    def _convert_messages(self, messages: list[BaseMessage]) -> list[Message]:
        """Converte messaggi LangChain nel nostro formato."""
        converted = []
        tool_system_prompt = self._build_tool_system_prompt()
        tool_prompt_injected = False

        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
                content = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )
                if tool_system_prompt:
                    content = f"{content}\n\n{tool_system_prompt}"
                    tool_prompt_injected = True
            elif isinstance(msg, HumanMessage):
                role = "user"
                content = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )
            elif isinstance(msg, ToolMessage):
                role = "user"
                tool_content = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )
                content = (
                    f"Tool result from {msg.name or 'tool'}:\n"
                    f"{tool_content}"
                )
            elif isinstance(msg, AIMessage):
                role = "assistant"
                content = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )
            else:
                role = "user"
                content = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )

            converted.append(Message(role=role, content=content))

        if tool_system_prompt and not tool_prompt_injected:
            converted.insert(
                0,
                Message(role="system", content=tool_system_prompt),
            )

        return converted

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generazione sincrona (esegue il metodo async)."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Se siamo già in un contesto async, crea un nuovo loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._agenerate(messages, stop, None, **kwargs)
                )
                return future.result()
        return loop.run_until_complete(
            self._agenerate(messages, stop, None, **kwargs)
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generazione asincrona."""
        from backend.chat import ChatModelOutput

        # Converti i messaggi nel nostro formato
        converted = self._convert_messages(messages)

        # Chiama il nostro ChatModel (non streaming)
        result = await self.chat_model.generate(
            converted, stream=False, **kwargs
        )

        # Type guard: assicurati che sia ChatModelOutput
        if not isinstance(result, ChatModelOutput):
            raise TypeError("Expected ChatModelOutput, got AsyncIterator")

        tool_call = self._extract_tool_call(result.content)
        if tool_call:
            message = AIMessage(content="", tool_calls=[tool_call])
        else:
            message = AIMessage(content=result.content)
        generation = ChatGeneration(message=message)

        llm_output = {}
        if result.usage:
            llm_output["token_usage"] = result.usage
        if result.model:
            llm_output["model_name"] = result.model

        return ChatResult(
            generations=[generation],
            llm_output=llm_output if llm_output else None,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Streaming sincrono."""
        result = self._generate(messages, stop, run_manager, **kwargs)
        from langchain_core.messages import AIMessageChunk
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=result.generations[0].message.content
            )
        )
        yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Streaming asincrono."""
        from langchain_core.messages import AIMessageChunk

        from backend.chat import ChatModelOutput

        converted = self._convert_messages(messages)
        result = await self.chat_model.generate(
            converted, stream=True, **kwargs
        )

        # Type guard: assicurati che sia AsyncIterator
        if isinstance(result, ChatModelOutput):
            raise TypeError("Expected AsyncIterator, got ChatModelOutput")

        async for chunk_text in result:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=chunk_text)
            )
            if run_manager:
                await run_manager.on_llm_new_token(chunk_text)
            yield chunk

    def bind_tools(
        self,
        tools,
        **kwargs: Any,
    ) -> LangChainChatModelAdapter:
        """
        Bind tools to the model (required by LangGraph).

        For providers without native tool calling, inject tool
        instructions into the prompt and parse JSON tool calls.
        """
        return self.model_copy(update={"bound_tools": list(tools)})


def create_langchain_adapter(
    chat_model: ChatModel,
) -> LangChainChatModelAdapter:
    """
    Crea un adapter LangChain per il nostro ChatModel.

    Args:
        chat_model: Il nostro ChatModel da wrappare

    Returns:
        Adapter compatibile con LangChain

    Example:
        >>> chat_model = ChatModel.from_name("watsonx:llama-3-3-70b")
        >>> langchain_model = create_langchain_adapter(chat_model)
        >>> agent = create_deep_agent(model=langchain_model, tools=[...])
    """
    return LangChainChatModelAdapter(chat_model=chat_model)
