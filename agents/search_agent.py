from typing import Dict, Any
from agents.base import BaseAgent


class SearchAgent(BaseAgent):
    id = "search"
    start_message = "Identifying relevant sources and key literature..."

    async def run(self, context: Dict[str, Any]) -> str:
        topic = context["topic"]
        n = context["num_sources"]

        return await self.call_mistral(
            system=f"""
            You are a research search agent with broad academic knowledge. \
                Your job is to identify {n} high-quality, real sources about the given topic.

                For each source provide:
                - Numbered entry (1., 2., etc.)
                - Full title of the paper, article, or report
                - Author(s) and year (e.g. Smith et al., 2022)
                - Publisher / journal / outlet (e.g. Nature, WHO, arXiv)
                - URL or DOI if known, otherwise write "Available via [institution/database]"
                - 3 bullet-point key findings specific to that source

                Draw on real published works where possible. Be specific, accurate, and factual. \
                Prefer peer-reviewed papers, authoritative institutions, and reputable outlets.
            """,
            user=f'Find {n} high-quality sources about: "{topic}"',
            max_tokens=1500,
        )
