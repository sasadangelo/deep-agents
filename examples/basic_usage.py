"""
Basic usage examples for the LLM Proxy system.

This file demonstrates how to use the LLM proxy to interact with
different providers (Ollama, WatsonX) using various protocols.
"""

import asyncio

from src.backend import LLMConfig, LLMFactory


async def example_ollama_native():
    """Example using Ollama with native Ollama protocol."""
    print("\n=== Ollama with Native Protocol ===")

    # Create client using factory from_name method
    client = LLMFactory.from_name("ollama:llama3.1")

    try:
        # Simple chat
        response = await client.chat(
            prompt="What is the capital of Italy?",
            system_prompt="You are a helpful assistant."
        )
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
    finally:
        await client.close()


async def example_ollama_openai_protocol():
    """Example using Ollama with OpenAI protocol."""
    print("\n=== Ollama with OpenAI Protocol ===")

    # Create client with OpenAI protocol
    client = LLMFactory.from_name(
        "ollama:llama3.1",
        protocol="openai"
    )

    try:
        response = await client.chat(
            prompt="What is 2+2?",
            system_prompt="You are a math tutor."
        )
        print(f"Response: {response.content}")
    finally:
        await client.close()


async def example_ollama_streaming():
    """Example using Ollama with streaming."""
    print("\n=== Ollama with Streaming ===")

    config = LLMConfig.from_name(
        "ollama:llama3.1",
        stream=True
    )
    client = LLMFactory.create(config)

    try:
        print("Streaming response: ", end="", flush=True)
        response = await client.chat(
            prompt="Count from 1 to 5 slowly."
        )

        # response is an AsyncIterator when streaming
        async for chunk in response:
            print(chunk, end="", flush=True)
        print()  # New line after streaming
    finally:
        await client.close()


async def example_watsonx():
    """Example using WatsonX."""
    print("\n=== WatsonX ===")

    # Note: You need to set these environment variables or pass them
    # export WATSONX_URL="https://your-watsonx-url"
    # export WATSONX_API_KEY="your-api-key"
    # export WATSONX_PROJECT_ID="your-project-id"

    import os

    # Check if credentials are available
    if not os.getenv("WATSONX_URL") or not os.getenv("WATSONX_API_KEY"):
        print("Skipping WatsonX example - credentials not set")
        print("Set WATSONX_URL and WATSONX_API_KEY to run this example")
        return

    client = LLMFactory.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        base_url=os.getenv("WATSONX_URL"),
        api_key=os.getenv("WATSONX_API_KEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )

    try:
        response = await client.chat(
            prompt="What is artificial intelligence?",
            system_prompt="You are an AI expert."
        )
        print(f"Response: {response.content}")
    finally:
        await client.close()


async def example_from_env():
    """Example using environment variables."""
    print("\n=== Using Environment Variables ===")

    import os

    # Set environment variables
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = "llama3.1"
    os.environ["LLM_PROTOCOL"] = "ollama"
    os.environ["LLM_TEMPERATURE"] = "0.7"

    try:
        client = LLMFactory.from_env()

        response = await client.chat(
            prompt="Hello, how are you?"
        )
        print(f"Response: {response.content}")

        await client.close()
    except ValueError as e:
        print(f"Error: {e}")


async def example_context_manager():
    """Example using async context manager."""
    print("\n=== Using Context Manager ===")

    async with LLMFactory.from_name("ollama:llama3.1") as client:
        response = await client.chat(
            prompt="What is Python?"
        )
        print(f"Response: {response.content}")


async def main():
    """Run all examples."""
    print("LLM Proxy - Basic Usage Examples")
    print("=" * 50)

    # Run examples
    await example_ollama_native()
    await example_ollama_openai_protocol()
    await example_ollama_streaming()
    await example_watsonx()
    await example_from_env()
    await example_context_manager()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
