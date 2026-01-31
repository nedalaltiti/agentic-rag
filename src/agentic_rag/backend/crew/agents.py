"""CrewAI agent definitions for the RAG pipeline.

Defines a researcher agent (retrieves facts) and a writer agent
(synthesizes answers with citations and hallucination guardrails).
"""

from crewai import Agent
from langchain_openai import ChatOpenAI

from agentic_rag.shared.config import settings
from .tools import DatabaseSearchTool, MemoryLookupTool

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    base_url=f"{settings.OLLAMA_BASE_URL}/v1",
    api_key="NA",
    temperature=0.1,
)


def create_researcher_agent(session_id: str) -> Agent:
    """Create the research agent with search and memory tools."""
    return Agent(
        role="Senior Research Analyst",
        goal="Analyze requests and retrieve precise information from the knowledge base.",
        backstory=(
            "You are an expert researcher. Your job is to understand the user's question, "
            "check past conversation context if the user refers to it, and query the knowledge base "
            "to find precise facts. You do not hallucinate."
        ),
        tools=[
            DatabaseSearchTool(),
            MemoryLookupTool(session_id=session_id),
        ],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def create_writer_agent() -> Agent:
    """Create the writer agent with hallucination guardrails."""
    return Agent(
        role="Technical Content Synthesizer",
        goal="Synthesize retrieved info into clear, accurate answers with citations.",
        backstory=(
            "You are a skilled technical writer. You take raw information snippets "
            "and transform them into a coherent response. "
            "You MUST cite your sources using the File Name and Page Number provided by the Researcher. "
            "If the researcher provides no sources, you must NOT answer from general knowledge; "
            "instead, state that the information is not in the documents."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
