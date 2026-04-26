"""
Tests for the LLM Proxy system.

Basic tests to verify configuration, factory, and client creation.
"""

import pytest

from src.backend import ChatModelParameters, ChatModel, ProtocolType, ProviderType
from src.adapters.ollama import OllamaChatModel
from src.adapters.watsonx import WatsonxChatModel


def test_config_from_name_ollama():
    """Test creating config from name string for Ollama."""
    config = ChatModelParameters.from_name("ollama:llama3.1")

    assert config.provider == ProviderType.OLLAMA
    assert config.protocol == ProtocolType.OLLAMA
    assert config.model == "llama3.1"
    assert config.base_url == "http://localhost:11434"


def test_config_from_name_watsonx():
    """Test creating config from name string for WatsonX."""
    config = ChatModelParameters.from_name(
        "watsonx:granite-3-8b",
        base_url="https://watsonx.example.com",
        api_key="test-key"
    )

    assert config.provider == ProviderType.WATSONX
    assert config.protocol == ProtocolType.WATSONX
    assert config.model == "granite-3-8b"
    assert config.base_url == "https://watsonx.example.com"
    assert config.api_key == "test-key"


def test_config_with_protocol_override():
    """Test overriding protocol in config."""
    config = ChatModelParameters(
        provider=ProviderType.OLLAMA,
        protocol=ProtocolType.OPENAI,
        model="llama3.1"
    )

    assert config.provider == ProviderType.OLLAMA
    assert config.protocol == ProtocolType.OPENAI


def test_factory_create_ollama():
    """Test factory creates Ollama client."""
    config = ChatModelParameters.from_name("ollama:llama3.1")
    client = ChatModel.create(config)

    assert isinstance(client, OllamaChatModel)
    assert client.get_provider() == "ollama"
    assert client.get_model_name() == "llama3.1"


def test_factory_create_watsonx():
    """Test factory creates WatsonX client."""
    config = ChatModelParameters.from_name(
        "watsonx:granite-3-8b",
        base_url="https://watsonx.example.com",
        api_key="test-key"
    )
    client = ChatModel.create(config)

    assert isinstance(client, WatsonxChatModel)
    assert client.get_provider() == "watsonx"
    assert client.get_model_name() == "granite-3-8b"


def test_factory_from_name():
    """Test factory from_name method."""
    client = ChatModel.from_name("ollama:llama3.1")

    assert isinstance(client, OllamaChatModel)
    assert client.config.protocol == ProtocolType.OLLAMA


def test_factory_from_name_with_protocol():
    """Test factory from_name with protocol override."""
    client = ChatModel.from_name(
        "ollama:llama3.1",
        protocol="openai"
    )

    assert isinstance(client, OllamaChatModel)
    assert client.config.protocol == ProtocolType.OPENAI


def test_config_temperature_validation():
    """Test temperature validation."""
    # Valid temperature
    config = ChatModelParameters(
        provider=ProviderType.OLLAMA,
        protocol=ProtocolType.OLLAMA,
        model="llama3.1",
        temperature=0.5
    )
    assert config.temperature == 0.5

    # Invalid temperature should raise validation error
    with pytest.raises(Exception):  # Pydantic ValidationError
        ChatModelParameters(
            provider=ProviderType.OLLAMA,
            protocol=ProtocolType.OLLAMA,
            model="llama3.1",
            temperature=1.5  # > 1.0
        )


def test_config_to_dict():
    """Test config serialization to dict."""
    config = ChatModelParameters.from_name(
        "ollama:llama3.1",
        temperature=0.8,
        max_tokens=100
    )

    config_dict = config.to_dict()

    assert config_dict["provider"] == "ollama"
    assert config_dict["model"] == "llama3.1"
    assert config_dict["temperature"] == 0.8
    assert config_dict["max_tokens"] == 100


def test_invalid_provider_name():
    """Test invalid provider name raises error."""
    with pytest.raises(ValueError):
        ChatModelParameters.from_name("invalid:model")


def test_invalid_name_format():
    """Test invalid name format raises error."""
    with pytest.raises(ValueError):
        ChatModelParameters.from_name("invalid-format")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
