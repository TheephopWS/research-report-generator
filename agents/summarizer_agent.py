from typing import Dict, Any
from agents.base_mistral import BaseAgent


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

            Label each summary with the author(s) and year in this exact format: \
            "Source 1 — Smith et al. (2022):", "Source 2 — Jones (2019):", etc. \
            Extract the author(s) and year directly from the source metadata provided. \
            Within each summary, use inline author-year citations (e.g. "Smith et al. (2022) argue...") \
            rather than generic labels like "(S1)" or "the source". \
            Be precise and analytical.
            """,
            user=f'Topic: "{topic}"\n\nSources to summarize:\n{sources}',
            max_tokens=5000,
        )
