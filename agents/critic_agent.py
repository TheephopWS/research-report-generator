from typing import Dict, Any
from agents.base import BaseAgent


class CriticAgent(BaseAgent):
    id = "critic"
    start_message = "Auditing quality and filling gaps..."

    async def run(self, context: Dict[str, Any]) -> str:
        topic = context["topic"]
        synthesis = context["synthesizer"]

        return await self.call_mistral(
            system="""
            You are a rigorous academic critic. Your job is to review a research \
synthesis and produce two clearly labelled sections:

CRITIQUE:
- Identify exactly 2-3 specific gaps, unsupported claims, or logical weaknesses
- Rate overall evidence quality as Strong, Moderate, or Weak — give one sentence of justification
- Note any missing perspectives or methodological concerns

REVISED SYNTHESIS:
- A strengthened, revised version of the synthesis that directly addresses every issue \
identified in the critique above
- Maintain the five-section structure (Overview, Theme A, Theme B, Agreements & Tensions, \
Synthesis Conclusion) but improve the content

Be direct and intellectually honest.""",
            user=f'Topic: "{topic}"\n\nSynthesis to critique:\n{synthesis}',
            max_tokens=5000,
        )
