"""
Configuration module for LLM providers.

Defines the configuration structure for different LLM providers,
including provider types, protocols, and connection settings.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    WATSONX = "watsonx"


class ProtocolType(str, Enum):
    """Supported API protocols."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    WATSONX = "watsonx"


class ChatModelParameters(BaseModel):
    """
    Configuration for LLM provider.

    Attributes:
        provider: The LLM provider to use (ollama, openai, watsonx, etc.)
        protocol: The API protocol to use (ollama, openai, watsonx)
        model: The model identifier (e.g., "llama3.1", "gpt-4", "granite-3-8b")
        base_url: Base URL for the API endpoint
        api_key: API key for authentication (optional)
        project_id: Project ID for WatsonX (optional)
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream responses
    """

    provider: ProviderType = Field(
        description="LLM provider to use"
    )
    protocol: ProtocolType = Field(
        description="API protocol to use"
    )
    model: str = Field(
        description="Model identifier"
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for the API endpoint"
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication"
    )
    project_id: str | None = Field(
        default=None,
        description="Project ID (for WatsonX)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream responses"
    )

    @classmethod
    def from_name(cls, name: str, **kwargs) -> ChatModelParameters:
        """
        Create config from a name string like 'ollama:llama3.1' or 'openai:gpt-4'.

        Args:
            name: Provider and model in format 'provider:model'
            **kwargs: Additional configuration parameters

        Returns:
            ChatModelParameters instance

        Example:
            >>> config = ChatModelParameters.from_name("ollama:llama3.1")
            >>> config = ChatModelParameters.from_name("watsonx:granite-3-8b", base_url="https://...")
        """
        if ":" not in name:
            raise ValueError(f"Invalid name format. Expected 'provider:model', got '{name}'")

        provider_str, model = name.split(sep=":", maxsplit=1)
        provider: ProviderType = ProviderType(provider_str)

        # Extract protocol from kwargs if provided, otherwise use default
        protocol_override = kwargs.pop("protocol", None)

        # Determine default protocol based on provider
        # Ollama defaults to ollama protocol, WatsonX to watsonx
        protocol_map: dict[ProviderType, ProtocolType] = {
            ProviderType.OLLAMA: ProtocolType.OLLAMA,
            ProviderType.WATSONX: ProtocolType.WATSONX,
        }

        if protocol_override:
            protocol = ProtocolType(protocol_override)
        else:
            protocol = protocol_map.get(provider, ProtocolType.OLLAMA)

        # Set default base URLs
        default_urls: dict[ProviderType, str] = {
            ProviderType.OLLAMA: "http://localhost:11434",
        }

        base_url = kwargs.pop("base_url", default_urls.get(provider))

        return cls(
            provider=provider,
            protocol=protocol,
            model=model,
            base_url=base_url,
            **kwargs
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.model_dump(exclude_none=True)
