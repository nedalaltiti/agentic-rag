"""CrewAI agent definitions for the agentic RAG pipeline.

Defines researcher and writer agents with prompts from Phoenix.
"""

from crewai import LLM, Agent

from agentic_rag.core.config import settings
from agentic_rag.core.prompts import PromptRegistry

from .tools import DatabaseSearchTool, MemoryLookupTool


def _get_llm() -> LLM:
    """Lazy LLM initialization to avoid import-time errors."""
    return LLM(
        model=f"ollama/{settings.LLM_MODEL}",
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
    )


def create_researcher_agent(session_id: str) -> Agent:
    """Create the researcher agent with knowledge base and memory tools."""
    llm = _get_llm()
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
    """Create the writer agent for synthesizing answers with citations."""
    llm = _get_llm()
    return Agent(
        role="Technical Content Synthesizer",
        goal="Synthesize retrieved info into clear, accurate answers with citations.",
        backstory=PromptRegistry.get_template("writer_backstory"),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
