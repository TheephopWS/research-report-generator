from typing import Dict, Any
from datetime import datetime
from agents.base import BaseAgent


class FormatterAgent(BaseAgent):
    id = "formatter"
    start_message = "Composing the final Markdown report..."

    async def run(self, context: Dict[str, Any]) -> str:
        topic = context["topic"]
        critic_output = context["critic"]
        sources = context["search"]
        date = datetime.now().strftime("%B %d, %Y")

        return await self.call_mistral(
            system=f"""You are a professional report formatter. Format the provided research \
content as a complete, publication-ready Markdown document using this exact structure:

# [Write a descriptive, specific title]
*Generated: {date}*

---

## Abstract
(2-3 sentences summarising the topic, approach, and key finding)

## Introduction
(Context and why this topic matters)

## [Descriptive heading for Theme A]
(Expanded section from the revised synthesis)

## [Descriptive heading for Theme B]
(Expanded section from the revised synthesis)

## Discussion & Synthesis
(Integrate agreements, tensions, and broader implications)

## Limitations
(Acknowledge scope limitations and gaps identified by the critic)

## Conclusion
(Clear, direct closing statement)

## References
1. [Source 1 — title, author, year, URL/DOI]
2. [Source 2 — title, author, year, URL/DOI]
...

Rules:
- Output ONLY clean Markdown — no commentary, no code fences around the output
- Use the revised synthesis content (not the original) wherever possible
- Make headings specific and descriptive, not generic labels""",
            user=f'Topic: "{topic}"\n\nResearch content:\n{critic_output}\n\nOriginal sources:\n{sources}',
            max_tokens=2000,
        )
