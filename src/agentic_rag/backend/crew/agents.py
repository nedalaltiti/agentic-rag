"""CrewAI agent definitions for the agentic RAG pipeline.

Defines researcher and writer agents with prompts from Phoenix.
"""

from crewai import Agent
from langchain_openai import ChatOpenAI

from agentic_rag.shared.config import settings
from agentic_rag.shared.prompts import PromptRegistry

from .tools import DatabaseSearchTool, MemoryLookupTool

# LLM configuration for CrewAI agents
# Uses Ollama via OpenAI-compatible endpoint
llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    base_url=f"{settings.OLLAMA_BASE_URL}/v1",
    api_key="NA",  # Ollama doesn't require API key
    temperature=0.1,
)


def create_researcher_agent(session_id: str) -> Agent:
    """
    Create the researcher agent.

    Retrieves information from the knowledge base and conversation history.

    Args:
        session_id: Session ID for conversation memory

    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Senior Research Analyst",
        goal="Analyze requests and retrieve precise information from the knowledge base.",
        backstory=PromptRegistry.get_template("researcher_backstory"),
        tools=[DatabaseSearchTool(), MemoryLookupTool(session_id=session_id)],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def create_writer_agent() -> Agent:
    """
    Create the writer agent.

    Synthesizes retrieved information into coherent answers with citations.

    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Technical Content Synthesizer",
        goal="Synthesize retrieved info into clear, accurate answers with citations.",
        backstory=PromptRegistry.get_template("writer_backstory"),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
