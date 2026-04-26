import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from tavily import TavilyClient

from adapters.langchain_adapter import (LangChainChatModelAdapter,
                                        create_langchain_adapter)
from backend import ChatModel

print("Loading environment variables...")
load_dotenv()

# Get environment variables
watsonx_url: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
watsonx_api_key: str | None = os.getenv("WATSONX_APIKEY")
project_id: str | None = os.getenv("WATSONX_PROJECT_ID")
tavily_api_key: str = os.environ["TAVILY_API_KEY"]

print("Initializing Tavily client...")
tavily_client: TavilyClient = TavilyClient(api_key=tavily_api_key)


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search and return structured results."""
    print(f"Running internet_search for query: {query}")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


research_instructions = """
You are an expert researcher. Your job is to conduct thorough research
and then write a polished report.

You have access to an internet search tool as your primary means of
gathering information.

## internet_search

Use this to run an internet search for a given query. You can specify
the max number of results to return, the topic, and whether raw content
should be included.
"""

print("Initializing Watsonx model...")
# Create our ChatModel instance
chat_model: ChatModel = ChatModel.from_name(
    name="watsonx:meta-llama/llama-3-3-70b-instruct",
    base_url=watsonx_url,
    api_key=watsonx_api_key,
    project_id=project_id,
    temperature=0.1,
    max_tokens=800,
)

# Wrap it in a LangChain-compatible adapter
model: LangChainChatModelAdapter = create_langchain_adapter(chat_model)

print("Creating deep agent...")
agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=research_instructions,
)

print("Invoking deep agent...")
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is LangGraph?"}]
})

print("Final response:")
print(result["messages"][-1].content)
