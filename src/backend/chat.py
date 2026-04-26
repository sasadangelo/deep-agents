"""
Base interface for LLM clients.

Defines the abstract base class that all LLM provider implementations
must follow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Self

from pydantic import BaseModel

from .types import ChatModelParameters


class Message(BaseModel):
    """Represents a chat message."""
    role: str
    content: str


class ChatModelOutput(BaseModel):
    """Response from LLM."""
    content: str
    model: str
    finish_reason: str | None = None
    usage: dict | None = None


class ChatModel(ABC):
    """
    Abstract base class for LLM clients.

    All provider implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: ChatModelParameters) -> None:
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        stream: bool | None = None,
        **kwargs
    ) -> ChatModelOutput | AsyncIterator[str]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of messages in the conversation
            stream: Whether to stream the response (overrides config)
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatModelOutput if not streaming, AsyncIterator[str] if streaming
        """
        pass

    @abstractmethod
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
            stream: Whether to stream the response (overrides config)
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatModelOutput if not streaming, AsyncIterator[str] if streaming
        """
        pass

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config.model

    def get_provider(self) -> str:
        """Get the provider name."""
        return self.config.provider.value

    async def close(self) -> None:
        """Close any resources. Override in subclasses if needed."""
        pass

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}"
            f"(provider={self.config.provider.value}, "
            f"model={self.config.model})"
        )

    @staticmethod
    def create(parameters: ChatModelParameters) -> ChatModel:
        """
        Create a ChatModel from parameters.

        Args:
            parameters: ChatModelParameters instance

        Returns:
            ChatModel instance
        """
        if parameters.provider == "ollama":
            from adapters.ollama import OllamaChatModel
            return OllamaChatModel(parameters)
        elif parameters.provider == "watsonx":
            from adapters.watsonx import WatsonxChatModel
            return WatsonxChatModel(parameters)
        else:
            raise ValueError(f"Unsupported provider: {parameters.provider}")

    @staticmethod
    def from_name(
        name: str, protocol: str | None = None, **kwargs
    ) -> ChatModel:
        """
        Create a ChatModel from a name string.

        Args:
            name: Provider and model in format 'provider:model'
            protocol: Optional protocol override
            **kwargs: Additional configuration parameters

        Returns:
            ChatModel instance
        """
        config = ChatModelParameters.from_name(
            name, protocol=protocol, **kwargs
        )

        if config.provider == "ollama":
            from adapters.ollama import OllamaChatModel
            return OllamaChatModel(config)
        elif config.provider == "watsonx":
            from adapters.watsonx import WatsonxChatModel
            return WatsonxChatModel(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    # @staticmethod
    # def from_env(
    #     provider_env: str = "LLM_PROVIDER",
    #     model_env: str = "LLM_MODEL",
    #     **kwargs
    # ) -> "ChatModel":
    #     """
    #     Create a ChatModel from environment variables.

    #     Args:
    #         provider_env: Environment variable name for provider
    #         model_env: Environment variable name for model
    #         **kwargs: Additional configuration parameters

    #     Returns:
    #         ChatModel instance
    #     """
    #     import os

    #     provider = os.getenv(provider_env)
    #     model = os.getenv(model_env)

    #     if not provider or not model:
    #         raise ValueError(
    #             f"Environment variables {provider_env} and "
    #             f"{model_env} must be set"
    #         )

    #     name = f"{provider}:{model}"
    #     return ChatModel.from_name(name, **kwargs)

