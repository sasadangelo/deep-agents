import os
from typing import Literal

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
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
    print("\n🔍 SUBAGENT TOOL CALL: internet_search")
    print(f"   Query: '{query}'")
    print(f"   Max results: {max_results}")
    print(f"   Topic: '{topic}'")

    result = tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )

    print(f"✅ SUBAGENT TOOL RESULT: Found {len(result.get('results', []))} results\n")
    return result


# Create the model instance for the subagent
subagent_model = init_chat_model(
    model="meta-llama/llama-3-3-70b-instruct",
    model_provider="ibm",
    url=watsonx_url,
    apikey=watsonx_api_key,
    project_id=project_id,
    temperature=0.1,
    max_tokens=800,
    max_retries=10,
    timeout=120,
)

# Define the research subagent
research_subagent: SubAgent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher with access to internet search. Use the internet_search tool to find accurate and up-to-date information.",
    "tools": [internet_search],
    "model": subagent_model,
}

print("Creating deep agent with subagent...")
print(f"📋 Registered subagents: [{research_subagent['name']}]")
print()

# Create main agent with research subagent
agent = create_deep_agent(
    model=init_chat_model(
        model="meta-llama/llama-3-3-70b-instruct",
        model_provider="ibm",
        url=watsonx_url,
        apikey=watsonx_api_key,
        project_id=project_id,
        temperature=0.1,
        max_tokens=800,
        max_retries=10,
        timeout=120,
    ),
    subagents=[research_subagent],
    system_prompt="You are a helpful assistant. When you need to research information, delegate to the research-agent subagent.",
)

print("=" * 80)
print("🚀 Invoking deep agent with user query...")
print("=" * 80)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is LangGraph?"}]
})

print("=" * 80)
print("✅ Agent execution completed")
print("=" * 80)
print("\n📝 Final response:")
print(result["messages"][-1].content)
