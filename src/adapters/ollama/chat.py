"""
Ollama provider implementation.

Implements the LLM client interface for Ollama with support for
both Ollama and OpenAI protocols.
"""

import json
from typing import AsyncIterator

import httpx
from httpx._models import Response

from backend.chat import ChatModel, ChatModelOutput, Message
from backend.types import ChatModelParameters, ProtocolType


class OllamaChatModel(ChatModel):
    """
    Ollama LLM client implementation.

    Supports both Ollama and OpenAI protocols.
    Can be used with streaming and non-streaming responses.
    """

    def __init__(self, config: ChatModelParameters):
        """Initialize Ollama client."""
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self.client = httpx.AsyncClient(timeout=300.0)

    async def generate(
        self,
        messages: list[Message],
        stream: bool | None = None,
        **kwargs
    ) -> ChatModelOutput | AsyncIterator[str]:
        """
        Generate a response using Ollama.

        Args:
            messages: List of messages
            stream: Whether to stream (overrides config)
            **kwargs: Additional parameters

        Returns:
            ChatModelOutput or AsyncIterator[str]
        """
        should_stream = (
            stream if stream is not None else self.config.stream
        )

        if self.config.protocol == ProtocolType.OPENAI:
            return await self._generate_openai(
                messages, stream=should_stream, **kwargs
            )
        else:
            return await self._generate_ollama(
                messages, stream=should_stream, **kwargs
            )

    async def _generate_ollama(
        self,
        messages: list[Message],
        stream: bool,
        **kwargs
    ) -> ChatModelOutput | AsyncIterator[str]:
        """Generate using Ollama protocol."""
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        payload = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
            }
        }

        if self.config.max_tokens:
            payload["options"]["num_predict"] = self.config.max_tokens

        payload.update(kwargs)

        url = f"{self.base_url}/api/chat"

        if stream:
            return self._stream_ollama(url, payload)
        else:
            return await self._non_stream_ollama(url, payload)

    async def _generate_openai(
        self,
        messages: list[Message],
        stream: bool,
        **kwargs
    ) -> ChatModelOutput | AsyncIterator[str]:
        """Generate using OpenAI protocol."""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        payload = {
            "model": self.config.model,
            "messages": openai_messages,
            "stream": stream,
            "temperature": self.config.temperature,
        }

        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens

        payload.update(kwargs)

        url = f"{self.base_url}/v1/chat/completions"

        if stream:
            return self._stream_openai(url, payload)
        else:
            return await self._non_stream_openai(url, payload)

    async def _non_stream_ollama(
        self, url: str, payload: dict
    ) -> ChatModelOutput:
        """Handle non-streaming Ollama response."""
        response: Response = await self.client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        return ChatModelOutput(
            content=data["message"]["content"],
            model=data["model"],
            finish_reason=data.get("done_reason"),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) +
                    data.get("eval_count", 0)
                ),
            }
        )

    async def _non_stream_openai(
        self, url: str, payload: dict
    ) -> ChatModelOutput:
        """Handle non-streaming OpenAI response."""
        response = await self.client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]

        return ChatModelOutput(
            content=choice["message"]["content"],
            model=data["model"],
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {})
        )

    async def _stream_ollama(
        self, url: str, payload: dict
    ) -> AsyncIterator[str]:
        """Handle streaming Ollama response."""
        async with self.client.stream(
            "POST", url, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        content = data["message"]["content"]
                        if content:
                            yield content

    async def _stream_openai(
        self, url: str, payload: dict
    ) -> AsyncIterator[str]:
        """Handle streaming OpenAI response."""
        async with self.client.stream(
            "POST", url, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

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
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
