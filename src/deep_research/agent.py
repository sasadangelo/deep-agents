# flake8: noqa: E402
import os
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables FIRST, before any other imports that need them
print("Loading environment variables...")
load_dotenv()

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from deep_research.research_agent.prompts import (
    RESEARCH_WORKFLOW_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS)
from deep_research.research_agent.tools import (tavily_search, think_tool,
                                                write_todos)

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

# Get current date
current_date: str = datetime.now().strftime(format="%Y-%m-%d")

# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS: str = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

# Get environment variables
watsonx_url: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
watsonx_api_key: str | None = os.getenv("WATSONX_APIKEY")
project_id: str | None = os.getenv("WATSONX_PROJECT_ID")
tavily_api_key: str = os.environ["TAVILY_API_KEY"]


# Create research sub-agent
research_sub_agent: SubAgent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search, think_tool, write_todos],
}

model: BaseChatModel = init_chat_model(
    model="meta-llama/llama-3-3-70b-instruct",
    model_provider="ibm",
    url=watsonx_url,
    apikey=watsonx_api_key,
    project_id=project_id,
    temperature=0.0,
    max_tokens=800,
    max_retries=10,
    timeout=120,
)


print("Creating deep agent with subagent...")
print(f"📋 Registered subagents: [{research_sub_agent['name']}]")
print()

# Create main agent with research subagent
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool, write_todos],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
)

print("=" * 80)
print("🚀 Invoking deep agent with user query...")
print("=" * 80)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research context engineering approaches used to build AI agents"}]
})

print("=" * 80)
print("✅ Agent execution completed")
print("=" * 80)
print("\n📝 Final response:")
print(result["messages"][-1].content)

# Save the agent graph to a file
graph_output_path = "agent_graph.png"
with open(graph_output_path, "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())
print(f"\n📊 Agent graph saved to: {graph_output_path}")
