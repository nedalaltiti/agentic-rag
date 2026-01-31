"""CrewAI pipeline orchestration for query processing.

Runs a sequential two-agent crew: researcher retrieves facts,
writer synthesizes the final cited response.
"""

from crewai import Crew, Process, Task
import structlog

from .agents import create_researcher_agent, create_writer_agent

logger = structlog.get_logger()


class CrewRunner:
    """Orchestrates the research-then-write agent pipeline."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.researcher = create_researcher_agent(session_id)
        self.writer = create_writer_agent()

    def kickoff(self, query: str) -> str:
        """
        Run the CrewAI pipeline synchronously.
        Should be called via asyncio.to_thread in FastAPI.
        """
        logger.info("Kickstarting Crew", session_id=self.session_id)

        research_task = Task(
            description=(
                f"Analyze the user query: '{query}'.\n"
                "1. If the user refers to past context (e.g., 'what about that?'), use MemoryLookupTool.\n"
                "2. Search the knowledge base using DatabaseSearchTool to find relevant facts.\n"
                "3. Compile a list of key facts with their Source ID, File Name, and Page Number."
            ),
            expected_output="A list of relevant facts with full source metadata.",
            agent=self.researcher,
        )

        write_task = Task(
            description=(
                "Using the researcher's findings, write a final answer to the user.\n"
                "1. Answer the question directly.\n"
                "2. Cite your sources inline (e.g., [File.pdf, p.12]).\n"
                "3. If no information is found in the tools, explicitly say "
                "'I could not find information on that in the documents.'\n"
                "4. Do not make up information."
            ),
            expected_output="A helpful, well-cited response in Markdown format.",
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
