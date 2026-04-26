import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from tavily import TavilyClient

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

print("Creating deep agent with Watsonx model...")
# Deep Agents supporta direttamente il formato provider:model tramite init_chat_model
# Il provider per WatsonX è "ibm" in LangChain
agent = create_deep_agent(
    model=init_chat_model(
        model="meta-llama/llama-3-3-70b-instruct",
        model_provider="ibm",
        url=watsonx_url,
        apikey=watsonx_api_key,
        project_id=project_id,
        temperature=0.1,
        max_tokens=800,
        max_retries=10,  # Aumenta per reti instabili (default: 6)
        timeout=120,     # Aumenta timeout per connessioni lente (secondi)
    ),
    tools=[internet_search],
    system_prompt=research_instructions,
)

print("Invoking deep agent...")
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is LangGraph?"}]
})

print("Final response:")
print(result["messages"][-1].content)
