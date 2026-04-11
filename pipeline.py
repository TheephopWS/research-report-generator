import asyncio
import base64
import json
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

def _run_pdflatex(latex: str) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "report.tex"
        tex_path.write_text(latex, encoding="utf-8")
        try:
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
    return await loop.run_in_executor(None, _run_pdflatex, latex)


class ResearchPipeline:
    def __init__(self, topic: str, num_sources: int, filepath: str = "contexts.json"):
        self.topic = topic
        self.num_sources = max(2, min(num_sources, 10))
        self.contexts = []
        self.agents = [
            SearchAgent(),
            SummarizerAgent(),
            SynthesizerAgent(),
            CriticAgent(),
            FormatterAgent(),
        ]
        self.contexts_filepath = filepath

    def _save_contexts(self, output: str) -> None:
        """Save input, contexts, and output to a JSON file."""
        contexts_file = Path(self.contexts_filepath)
        data = {
            "input": self.topic,
            "contexts": self.contexts,
            "output": output,
        }
        with open(contexts_file, "w") as f:
            json.dump(data, f, indent=2)

    async def run(self) -> AsyncGenerator[Dict[str, Any], None]:
        if len(self.topic) < 3:
            raise ValueError("Topic is too short. Please provide topic of at least 3 characters.")
            
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
                if agent.id == "search":
                    self.contexts.append(output)
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

        yield {
            "type": "report",
            "content": context.get("formatter", ""),
        }

        # Latex and pdf
        synthesizer = next(a for a in self.agents if a.id == "synthesizer")
        try:
            yield {"type": "pdf_compiling"}
            latex = await synthesizer.generate_latex(context)
            yield {"type": "latex", "content": latex}
            pdf_bytes = await _compile_pdf(latex)
            if pdf_bytes:
                yield {
                    "type": "pdf",
                    "content": base64.b64encode(pdf_bytes).decode(),
                }
        except Exception:
            pass

        # Save contexts to file
        self._save_contexts(context.get("formatter", ""))
