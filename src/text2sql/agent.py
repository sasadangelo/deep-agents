import argparse
import os
import sys
from argparse import Namespace

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools.base import BaseTool
from langchain_ibm import ChatWatsonx
from pydantic import SecretStr
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

console: Console = Console()


def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent powered by Watsonx."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    db_path: str = os.path.join(base_dir, "chinook.db")
    db: SQLDatabase = SQLDatabase.from_uri(database_uri=f"sqlite:///{db_path}", sample_rows_in_table_info=3)

    watsonx_url: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    watsonx_api_key: str | None = os.getenv("WATSONX_APIKEY")
    watsonx_project_id: str | None = os.getenv("WATSONX_PROJECT_ID")
    watsonx_model: str = os.getenv("WATSONX_MODEL", "meta-llama/llama-3-3-70b-instruct")

    if not watsonx_api_key:
        raise ValueError("WATSONX_APIKEY environment variable is required")

    if not watsonx_project_id:
        raise ValueError("WATSONX_PROJECT_ID environment variable is required")

    model: ChatWatsonx = ChatWatsonx(
        model_id=watsonx_model,
        url=SecretStr(secret_value=watsonx_url),
        apikey=SecretStr(secret_value=watsonx_api_key),
        project_id=watsonx_project_id,
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 800,
            GenParams.TEMPERATURE: 0,
        },
    )

    toolkit: SQLDatabaseToolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools: list[BaseTool] = toolkit.get_tools()

    console.print("[dim]SQLDatabaseToolkit tools:[/dim]")
    for tool in sql_tools:
        console.print(f" - [bold]{tool.name}[/bold]: {tool.description}")
    console.print()

    # Build absolute paths for memory and skills
    memory_path = os.path.join(base_dir, "AGENTS.md")
    skills_path = os.path.join(base_dir, "skills")

    agent = create_deep_agent(
        model=model,
        memory=[memory_path],
        skills=[skills_path],
        tools=sql_tools,
        subagents=[],
        backend=FilesystemBackend(root_dir=base_dir),
    )

    # Debug: verify memory file exists
    if os.path.exists(path=memory_path):
        console.print(f"[dim]✓ Memory file loaded: {memory_path}[/dim]")
    else:
        console.print(f"[bold red]✗ Memory file NOT found: {memory_path}[/bold red]")

    return agent


def main() -> None:
    """Main entry point for the SQL Deep Agent CLI."""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Deep Agent powered by LangChain Deep Agents and Watsonx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py "Which employee generated the most revenue by country?"
  python agent.py "How many customers are from Canada?"
        """,
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer using the Chinook database",
    )

    args: Namespace = parser.parse_args()

    console.print(
        Panel(renderable=f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print()

    console.print("[dim]Creating SQL Deep Agent...[/dim]")
    agent = create_sql_deep_agent()

    console.print("[dim]Processing query...[/dim]\n")

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.question}]}
        )

        final_message = result["messages"][-1]
        answer = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        console.print(
            Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green")
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
