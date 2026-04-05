from typing import AsyncGenerator, Dict, Any
from agents import (
    SearchAgent,
    SummarizerAgent,
    SynthesizerAgent,
    CriticAgent,
    FormatterAgent,
)


class ResearchPipeline:
    """
    Chains five specialised agents in sequence.
    Each agent receives the accumulated context and yields
    SSE-compatible dict events for the API to stream.
    """

    def __init__(self, topic: str, num_sources: int):
        self.topic = topic
        self.num_sources = max(2, min(num_sources, 6))
        self.agents = [
            SearchAgent(),
            SummarizerAgent(),
            SynthesizerAgent(),
            CriticAgent(),
            FormatterAgent(),
        ]

    async def run(self) -> AsyncGenerator[Dict[str, Any], None]:
        context: Dict[str, Any] = {
            "topic": self.topic,
            "num_sources": self.num_sources,
        }

        for agent in self.agents:
            yield {
                "type": "agent_start",
                "agent": agent.id,
                "message": agent.start_message,
            }
            try:
                output = await agent.run(context)
                context[agent.id] = output
                yield {
                    "type": "agent_done",
                    "agent": agent.id,
                    "output": output,
                }
            except Exception as exc:
                yield {
                    "type": "agent_error",
                    "agent": agent.id,
                    "message": str(exc),
                }
                return  # Halt pipeline on error

        # Emit the final formatted report
        yield {
            "type": "report",
            "content": context.get("formatter", ""),
        }

        # Generate LaTeX summary via synthesizer tool-calling
        synthesizer = next(a for a in self.agents if a.id == "synthesizer")
        try:
            latex = await synthesizer.generate_latex(context)
            yield {"type": "latex", "content": latex}
        except Exception:
            pass  # LaTeX generation is non-critical
