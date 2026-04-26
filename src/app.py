import os
from typing import Any, Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import ChatWatsonx
from pydantic import SecretStr
from tavily import TavilyClient

print("Loading environment variables...")
load_dotenv()

parameters: dict[str, Any] = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

project_id = os.getenv("WATSONX_PROJECT_ID")
tavily_api_key = os.environ["TAVILY_API_KEY"]

print("Initializing Tavily client...")
tavily_client = TavilyClient(api_key=tavily_api_key)


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
model = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=SecretStr("https://us-south.ml.cloud.ibm.com"),
    project_id=project_id,
    params=parameters,
)

print("Creating deep agent...")
agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=research_instructions,
)

print("Invoking deep agent...")
result = agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph?"}]})

print("Final response:")
print(result["messages"][-1].content)

# Made with Bob
