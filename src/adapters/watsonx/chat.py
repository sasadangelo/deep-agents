"""
WatsonX provider implementation.

Wraps ChatWatsonx from langchain_ibm to use correct authentication.
"""

from typing import AsyncIterator

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import ChatWatsonx
from pydantic import SecretStr

from backend.chat import ChatModel, ChatModelOutput, Message
from backend.types import ChatModelParameters


class WatsonxChatModel(ChatModel):
    """
    WatsonX LLM client implementation.

    Wraps ChatWatsonx from langchain_ibm for correct authentication.
    """

    def __init__(self, config: ChatModelParameters) -> None:
        """Initialize WatsonX client using LangChain's ChatWatsonx."""
        super().__init__(config)

        if not config.base_url:
            raise ValueError("base_url is required for WatsonX")

        if not config.project_id:
            raise ValueError("project_id is required for WatsonX")

        # Prepare parameters for WatsonX
        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: config.max_tokens or 800,
            GenParams.TEMPERATURE: config.temperature,
        }

        # Create LangChain ChatWatsonx instance
        self.watsonx_client = ChatWatsonx(
            model_id=config.model,
            url=SecretStr(secret_value=config.base_url),
            project_id=config.project_id,
            params=parameters,
        )

    async def generate(
        self,
        messages: list[Message],
        stream: bool | None = None,
        **kwargs
    ) -> ChatModelOutput | AsyncIterator[str]:
        """
        Generate a response using WatsonX via LangChain.

        Args:
            messages: List of messages
            stream: Whether to stream (overrides config)
            **kwargs: Additional parameters

        Returns:
            ChatModelOutput or AsyncIterator[str]
        """
        from langchain_core.messages import (AIMessage, HumanMessage,
                                             SystemMessage)

        # Convert our messages to LangChain format
        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))

        should_stream: bool = (
            stream if stream is not None else self.config.stream
        )

        if should_stream:
            return self._stream_response(lc_messages)
        else:
            return await self._non_stream_response(lc_messages)

    async def _non_stream_response(
        self, lc_messages
    ) -> ChatModelOutput:
        """Handle non-streaming response using LangChain."""
        # LangChain's invoke is sync, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self.watsonx_client.invoke, lc_messages
        )

        # Handle content that might be str or list
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        return ChatModelOutput(
            content=content,
            model=self.config.model,
            finish_reason=None,
            usage={}
        )

    async def _stream_response(self, lc_messages) -> AsyncIterator[str]:
        """Handle streaming response using LangChain."""
        import asyncio

        # Get the sync generator
        sync_gen = self.watsonx_client.stream(lc_messages)

        # Convert to async
        for chunk in sync_gen:
            # Handle content that might be str or list
            content = (
                chunk.content
                if isinstance(chunk.content, str)
                else str(chunk.content)
            )
            yield content
            await asyncio.sleep(0)  # Yield control

    async def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        stream: bool | None = None,
        **kwargs
    ) -> ChatModelOutput | AsyncIterator[str]:
        """
        Simple chat interface.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream
            **kwargs: Additional parameters

        Returns:
            ChatModelOutput or AsyncIterator[str]
        """
        messages = []

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        messages.append(Message(role="user", content=prompt))

        return await self.generate(messages, stream=stream, **kwargs)

    async def close(self):
        """Close resources (no-op for LangChain client)."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
