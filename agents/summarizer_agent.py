from typing import Dict, Any
from agents.base import BaseAgent


class SummarizerAgent(BaseAgent):
    id = "summarizer"
    start_message = "Distilling key insights from each source..."

    async def run(self, context: Dict[str, Any]) -> str:
        topic = context["topic"]
        sources = context["search"]

        return await self.call_mistral(
            system="""
            You are an expert research summarizer. For each source provided, \
            write a focused 3-4 sentence academic summary that covers:
            - The main argument or primary finding
            - The evidence, data, or methodology used
            - Its specific relevance to the research topic

            Label each summary with its source number (e.g. "Source 1:", "Source 2:", etc.). \
            Be precise and analytical.
            """,
            user=f'Topic: "{topic}"\n\nSources to summarize:\n{sources}',
            max_tokens=5000,
        )
