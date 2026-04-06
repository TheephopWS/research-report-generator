import asyncio
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

from agents import (
    SearchAgent,
    SummarizerAgent,
    SynthesizerAgent,
    CriticAgent,
    FormatterAgent,
)


def _compile_pdf_sync(latex: str) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "report.tex"
        tex_path.write_text(latex, encoding="utf-8")
        try:
            # Run twice so TOC and hyperref cross-references resolve correctly
            for _ in range(2):
                subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-output-directory", tmpdir,
                        str(tex_path),
                    ],
                    capture_output=True,
                    timeout=120,
                )
            pdf_path = Path(tmpdir) / "report.pdf"
            if pdf_path.exists():
                return pdf_path.read_bytes()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
    return None


async def _compile_pdf(latex: str) -> bytes | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compile_pdf_sync, latex)


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
                return

        # Emit the final formatted report
        yield {
            "type": "report",
            "content": context.get("formatter", ""),
        }

        # Generate LaTeX and compile to PDF via synthesizer tool-calling
        synthesizer = next(a for a in self.agents if a.id == "synthesizer")
        try:
            latex = await synthesizer.generate_latex(context)
            yield {"type": "latex", "content": latex}
            yield {"type": "pdf_compiling"}
            pdf_bytes = await _compile_pdf(latex)
            if pdf_bytes:
                yield {
                    "type": "pdf",
                    "content": base64.b64encode(pdf_bytes).decode(),
                }
        except Exception:
            pass  # LaTeX/PDF generation is non-critical
