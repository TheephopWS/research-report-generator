import json
import logging
import re
from typing import Dict, Any

from agents.base_mistral import BaseAgent, MODEL

logger = logging.getLogger(__name__)

_VALID_JSON_ESCAPES = re.compile(r'\\(?!["\\bfnrtu/])')


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Escape any backslash not already part of a valid JSON escape sequence
        fixed = _VALID_JSON_ESCAPES.sub(r'\\\\', s)
        return json.loads(fixed)


def get_report_template(title: str, sections: list) -> str:
    sec = "\n\n".join("\\section{" + s + "}" for s in sections)
    return "\n".join([
        "\\documentclass[12pt,a4paper]{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{lmodern}",
        "\\usepackage{geometry}",
        "\\geometry{margin=1in}",
        "\\usepackage{setspace}",
        "\\onehalfspacing",
        "\\usepackage{hyperref}",
        "\\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\usepackage{amsmath}",
        "\\usepackage{parskip}",
        "",
        "\\title{" + title + "}",
        "\\author{AI Research Report Generator}",
        "\\date{\\today}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\begin{abstract}",
        "% Insert abstract here",
        "\\end{abstract}",
        "",
        "\\tableofcontents",
        "\\newpage",
        "",
        sec,
        "",
        "\\bibliographystyle{apalike}",
        "\\begin{thebibliography}{99}",
        "% Insert references here",
        "\\end{thebibliography}",
        "",
        "\\end{document}",
    ])


LATEX_TEMPLATE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_report_template",
        "description": (
            "Returns a tailored LaTeX report template with the specified title "
            "and section headings. Call this to get the base document structure, "
            "then fill in each section with the research content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title for the research report",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of section headings for the report body",
                },
            },
            "required": ["title", "sections"],
        },
    },
}

_TOOL_FUNCTIONS = {
    "get_report_template": get_report_template,
}


class SynthesizerAgent(BaseAgent):
    id = "synthesizer"
    start_message = "Weaving sources into a coherent narrative..."

    async def run(self, context: Dict[str, Any]) -> str:
        topic = context["topic"]
        summaries = context["summarizer"]

        return await self.call_mistral(
            system="""
            You are a research synthesizer. Given summaries of multiple sources, \
            produce a coherent academic synthesis with these five clearly labelled sections:

            1. OVERVIEW — what the literature collectively says about this topic
            2. KEY THEME A — a major finding or argument supported across sources
            3. KEY THEME B — a second distinct and significant theme
            4. AGREEMENTS & TENSIONS — where sources converge and where they diverge
            5. SYNTHESIS CONCLUSION — an integrated takeaway that goes beyond any single source

            CRITICAL CITATION RULE: Every factual claim must include an inline author-year citation, \
            e.g. "Vaswani et al. (2017) demonstrate..." or "...has been shown to improve performance \
            (Brown et al., 2020)". NEVER use shorthand labels like (S1), (S2), Source 1, etc. \
            Use only the author surnames and year extracted from the source metadata.

            Write in clear, academic prose. Do not just repeat the summaries — synthesize them.
            """,
            user=f'Topic: "{topic}"\n\nSource Summaries:\n{summaries}',
            max_tokens=5000,
        )

    async def generate_latex(self, context: Dict[str, Any]) -> str:
        topic = context["topic"]
        report = context.get("formatter", context.get("critic", ""))
        logger.info("generate_latex called (report length: %d chars)", len(report))

        messages: list[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a LaTeX document generator. Produce a complete, compilable "
                    "LaTeX research report.\n\n"
                    "First, call the get_report_template tool with a descriptive title "
                    "and appropriate section headings for the research topic.\n"
                    "Then, using the returned template, fill in every section with the "
                    "provided research content and output the complete LaTeX document.\n\n"
                    "Rules:\n"
                    "- Do not put the final report in any ```latex wrapper, just plain LaTex source\n"
                    "- Output ONLY the final LaTeX source, no commentary\n"
                    "- Properly escape special LaTeX characters in content\n"
                    "- Maintain academic tone and structure\n"
                    "- Remove the references section and include all references in the bibliography section\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Generate a LaTeX report for: "{topic}"\n\n'
                    f"Research content:\n{report}"
                ),
            },
        ]

        # Get template
        logger.info("Calling LLM for template tool call")
        response = await self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[LATEX_TEMPLATE_TOOL],
            tool_choice="required",
            max_tokens=1000,
        )

        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            tc = assistant_msg.tool_calls[0]
            fn_name = tc.function.name
            logger.info("Tool call received: %s", fn_name)
            raw_args = tc.function.arguments
            fn_args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else raw_args
            args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)

            tool_fn = _TOOL_FUNCTIONS.get(fn_name)
            tool_result = tool_fn(**fn_args) if tool_fn else ""
            logger.info("Template built (%d chars), calling LLM for full LaTeX", len(tool_result))

            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": args_str},
                    }
                ],
            })
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tc.id,
            })

            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=5000,
            )
        else:
            logger.warning("No tool call in first LLM response — proceeding with raw content")

        raw = response.choices[0].message.content or ""
        # Strip markdown code fences the LLM may have added despite instructions
        latex = re.sub(r'^```(?:latex|tex)?\s*\n', '', raw.strip())
        latex = re.sub(r'\n```\s*$', '', latex)
        logger.info("LaTeX finalised (%d chars)", len(latex))
        return latex.strip()
