"""CrewAI pipeline orchestration for query processing."""

import structlog
from crewai import Crew, Process, Task

from agentic_rag.core.prompts import PromptRegistry
from agentic_rag.core.schemas import Citation

from .agents import create_researcher_agent, create_writer_agent
from .tools import DatabaseSearchTool, MemoryLookupTool

logger = structlog.get_logger()


class CrewRunner:
    """Orchestrates the research-then-write agent pipeline."""

    def __init__(self, session_id: str, model: str | None = None):
        self.session_id = session_id
        self.db_tool = DatabaseSearchTool()
        self.mem_tool = MemoryLookupTool(session_id=session_id)
        self.researcher = create_researcher_agent(
            session_id,
            model_name=model,
            database_tool=self.db_tool,
            memory_tool=self.mem_tool,
        )
        self.writer = create_writer_agent(model_name=model)

    def kickoff_with_context(self, query: str, knowledge_context: str) -> str:
        """Run CrewAI with pre-retrieved context, bypassing researcher tool usage."""
        logger.info(
            "Kickstarting Crew with pre-retrieved context",
            session_id=self.session_id,
            context_length=len(knowledge_context),
        )

        user_prompt = PromptRegistry.render(
            "user_prompt",
            query=query,
            context=knowledge_context,
        )

        write_task = Task(
            description=user_prompt,
            expected_output=(
                "A well-formatted PDPL compliance response "
                "with inline citations and References section."
            ),
            agent=self.writer,
        )

        synthesis_crew = Crew(
            agents=[self.writer],
            tasks=[write_task],
            process=Process.sequential,
            verbose=True,
        )

        result = synthesis_crew.kickoff()
        return str(result)

    def kickoff(self, query: str) -> tuple[str, list[Citation]]:
        """Run the full CrewAI pipeline with tool usage."""
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
                "IF researcher found 'No relevant PDPL information found.':\n"
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
        citations = self.db_tool.get_last_citations()
        return str(result), citations
