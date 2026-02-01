"""CrewAI pipeline orchestration for query processing.

Supports two modes:
1. kickoff_with_context: Pre-retrieved context passed to writer (deterministic)
2. kickoff: Full agent pipeline with tool usage (requires capable model)
"""

import structlog
from crewai import Crew, Process, Task

from agentic_rag.shared.prompts import PromptRegistry

from .agents import create_researcher_agent, create_writer_agent

logger = structlog.get_logger()


class CrewRunner:
    """Orchestrates the research-then-write agent pipeline."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.researcher = create_researcher_agent(session_id)
        self.writer = create_writer_agent()

    def kickoff_with_context(self, query: str, knowledge_context: str) -> str:
        """
        Run CrewAI with pre-retrieved context (deterministic RAG).
        
        This bypasses the researcher's tool usage and passes documents
        directly to the writer for synthesis.
        
        Args:
            query: User's question
            knowledge_context: Pre-formatted KNOWLEDGE block from retrieval
        
        Returns:
            Synthesized response with citations
        """
        logger.info(
            "Kickstarting Crew with pre-retrieved context",
            session_id=self.session_id,
            context_length=len(knowledge_context),
        )

        # Render user prompt from template
        user_prompt = PromptRegistry.render(
            "user_prompt",
            query=query,
            context=knowledge_context,
        )

        # Single task: Writer synthesizes from pre-retrieved context
        write_task = Task(
            description=user_prompt,
            expected_output=(
                "A well-formatted PDPL compliance response "
                "with inline citations and References section."
            ),
            agent=self.writer,
        )

        # Single-agent crew for synthesis only
        synthesis_crew = Crew(
            agents=[self.writer],
            tasks=[write_task],
            process=Process.sequential,
            verbose=True,
        )

        result = synthesis_crew.kickoff()
        return str(result)

    def kickoff(self, query: str) -> str:
        """
        Run the full CrewAI pipeline with tool usage.
        
        Note: Requires a capable model (7B+) for reliable tool calling.
        For smaller models, use kickoff_with_context instead.
        """
        logger.info("Kickstarting Crew (full pipeline)", session_id=self.session_id)

        research_task = Task(
            description=(
                f"User query: '{query}'\n\n"
                "INSTRUCTIONS (YOU MUST FOLLOW):\n"
                "1. You MUST call DatabaseSearchTool at least once for every question.\n"
                "2. If the user refers to past context, ALSO use MemoryLookupTool.\n"
                "3. For each finding, preserve the EXACT text "
                "and any article/section identifiers.\n"
                "4. Include: Source ID, File Name, Section/Article "
                "(if present), and exact text passage.\n"
                "5. If DatabaseSearchTool returns no documents, "
                "explicitly state: 'No relevant PDPL information found.'"
            ),
            expected_output="Exact passages from the knowledge base with full source metadata.",
            agent=self.researcher,
        )

        write_task = Task(
            description=(
                "Using the researcher's findings, write a response following these rules:\n\n"
                "IF researcher found 'No relevant PDPL information':\n"
                "- Naturally explain it's not in the knowledge base, "
                "then suggest related PDPL questions.\n\n"
                "FOR SUBSTANTIVE ANSWERS:\n"
                "1. Answer using ONLY the provided findings - never invent information.\n"
                "2. Add inline citations [1], [2] for every factual statement.\n"
                "3. End with a References section mapping each [n] to its source.\n"
                "4. Use clear formatting: sections, bullets, **bold** for key terms.\n"
                "5. NO emojis. Professional tone.\n"
                "6. End with: 'Is there anything else I can help "
                "you with regarding PDPL compliance?'"
            ),
            expected_output=(
                "A well-formatted PDPL compliance response "
                "with inline citations and References section."
            ),
            agent=self.writer,
            context=[research_task],
        )

        rag_crew = Crew(
            agents=[self.researcher, self.writer],
            tasks=[research_task, write_task],
            process=Process.sequential,
            verbose=True,
        )

        result = rag_crew.kickoff()
        return str(result)
