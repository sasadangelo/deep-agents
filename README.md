# Deep Agents - LLM Proxy

A simplified LLM proxy system inspired by beeai-framework, providing a unified interface for multiple LLM providers.

## Features

- **Multiple Providers**: Support for Ollama and WatsonX
- **Multiple Protocols**: Ollama can use both native Ollama protocol and OpenAI-compatible protocol
- **Streaming Support**: Both streaming and non-streaming responses
- **Simple Configuration**: Easy setup via code, environment variables, or configuration objects
- **Async/Await**: Full async support with context managers
- **Type Safe**: Built with Pydantic for configuration validation

## Supported Providers

### Ollama
- **Protocols**: Ollama (native), OpenAI (compatible)
- **Default URL**: `http://localhost:11434`
- **Models**: Any model available in your Ollama installation (llama3.1, mistral, etc.)

### WatsonX
- **Protocol**: WatsonX (native)
- **Authentication**: API Key + Project ID
- **Models**: IBM Granite models and others available in WatsonX

## Installation

```bash
# Install dependencies
pip install httpx pydantic python-dotenv

# Or using uv
uv pip install httpx pydantic python-dotenv
```

## Quick Start

### Basic Usage with Ollama

```python
import asyncio
from src.llm_proxy import LLMFactory

async def main():
    # Create client
    client = LLMFactory.from_name("ollama:llama3.1")

    # Simple chat
    response = await client.chat(
        prompt="What is the capital of Italy?",
        system_prompt="You are a helpful assistant."
    )

    print(response.content)
    await client.close()

asyncio.run(main())
```

### Using OpenAI Protocol with Ollama

```python
client = LLMFactory.from_name(
    "ollama:llama3.1",
    protocol="openai"  # Use OpenAI-compatible protocol
)
```

### Streaming Responses

```python
config = LLMConfig.from_name("ollama:llama3.1", stream=True)
client = LLMFactory.create(config)

response = await client.chat(prompt="Tell me a story")

# Iterate over chunks
async for chunk in response:
    print(chunk, end="", flush=True)

await client.close()
```

### Using Context Manager

```python
async with LLMFactory.from_name("ollama:llama3.1") as client:
    response = await client.chat(prompt="Hello!")
    print(response.content)
```

### WatsonX Example

```python
import os

client = LLMFactory.from_name(
    "watsonx:ibm/granite-3-8b-instruct",
    base_url=os.getenv("WATSONX_URL"),
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID")
)

response = await client.chat(prompt="What is AI?")
print(response.content)
await client.close()
```

### Using Environment Variables

```python
import os

# Set environment variables
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["LLM_MODEL"] = "llama3.1"
os.environ["LLM_PROTOCOL"] = "ollama"
os.environ["LLM_TEMPERATURE"] = "0.7"

# Create client from environment
client = LLMFactory.from_env()
response = await client.chat(prompt="Hello!")
await client.close()
```

## Configuration

### LLMConfig Parameters

- `provider`: Provider type (`ollama`, `watsonx`)
- `protocol`: Protocol type (`ollama`, `openai`, `watsonx`)
- `model`: Model identifier
- `base_url`: API endpoint URL (optional)
- `api_key`: API key for authentication (optional)
- `project_id`: Project ID for WatsonX (optional)
- `temperature`: Sampling temperature (0.0-1.0, default: 0.7)
- `max_tokens`: Maximum tokens to generate (optional)
- `stream`: Enable streaming responses (default: False)

### Environment Variables

- `LLM_PROVIDER`: Provider name
- `LLM_MODEL`: Model identifier
- `LLM_PROTOCOL`: Protocol type
- `LLM_BASE_URL`: Base URL
- `LLM_API_KEY`: API key
- `LLM_PROJECT_ID`: Project ID (WatsonX)
- `LLM_TEMPERATURE`: Temperature
- `LLM_MAX_TOKENS`: Max tokens
- `LLM_STREAM`: Stream responses (true/false)

## Examples

See the `examples/` directory for more examples:

```bash
# Run basic examples
python examples/basic_usage.py
```

## Architecture

```
src/llm_proxy/
├── __init__.py          # Package exports
├── base.py              # Base LLM client interface
├── config.py            # Configuration classes
├── factory.py           # Factory for creating clients
└── providers/
    ├── __init__.py
    ├── ollama.py        # Ollama implementation
    └── watsonx.py       # WatsonX implementation
```

## Inspiration

This project is inspired by the [beeai-framework](https://github.com/i-am-bee/bee-agent-framework) backend system, which provides a comprehensive interface for multiple LLM providers. This is a simplified version focusing on the core concepts of provider abstraction and protocol flexibility.

## License

MIT License
