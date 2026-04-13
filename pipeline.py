import asyncio
import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

logger = logging.getLogger(__name__)

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
            for i in range(2):
                result = subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-output-directory", tmpdir,
                        str(tex_path),
                    ],
                    capture_output=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    logger.warning(
                        "pdflatex pass %d exit code %d\nSTDOUT:\n%s\nSTDERR:\n%s",
                        i + 1,
                        result.returncode,
                        result.stdout.decode(errors="replace")[-3000:],
                        result.stderr.decode(errors="replace")[-1000:],
                    )
                else:
                    logger.info("pdflatex pass %d succeeded", i + 1)
            pdf_path = Path(tmpdir) / "report.pdf"
            if pdf_path.exists():
                logger.info("PDF created: %d bytes", pdf_path.stat().st_size)
                return pdf_path.read_bytes()
            else:
                logger.error("PDF file not found after pdflatex runs — check LaTeX source")
        except FileNotFoundError:
            logger.error("pdflatex not found — is TeX Live / MikTeX installed?")
        except subprocess.TimeoutExpired:
            logger.error("pdflatex timed out after 120 seconds")
        except OSError as exc:
            logger.error("OS error running pdflatex: %s", exc)
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
            logger.info("Starting LaTeX generation for topic: %s", self.topic)
            latex = await synthesizer.generate_latex(context)
            if not latex.strip():
                logger.error("LaTeX generation returned empty content")
                yield {"type": "pdf_error", "message": "LaTeX generation returned empty content"}
            else:
                logger.info("LaTeX generated (%d chars), starting PDF compilation", len(latex))
                yield {"type": "latex", "content": latex}
                pdf_bytes = await _compile_pdf(latex)
                if pdf_bytes:
                    logger.info("PDF compiled successfully (%d bytes)", len(pdf_bytes))
                    yield {
                        "type": "pdf",
                        "content": base64.b64encode(pdf_bytes).decode(),
                    }
                else:
                    logger.error("PDF compilation produced no output")
                    yield {"type": "pdf_error", "message": "PDF compilation failed — pdflatex may not be installed or the LaTeX source has errors. Check server logs."}
        except Exception as exc:
            logger.exception("Unhandled error during LaTeX/PDF generation")
            yield {"type": "pdf_error", "message": str(exc)}

        # Save contexts to file
        self._save_contexts(context.get("formatter", ""))
