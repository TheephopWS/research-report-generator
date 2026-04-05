import os
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()

from pipeline import ResearchPipeline  # noqa: E402 — after load_dotenv

app = FastAPI(
    title="Research Report Generator",
    description="A 5-agent AI pipeline: Search → Summarise → Synthesise → Critique → Format",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    num_sources: int = Field(default=3, ge=2, le=6)


@app.get("/", response_class=HTMLResponse)
async def serve_demo():
    """Serve the demo front-end."""
    demo_path = Path(__file__).parent / "demo" / "index.html"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo page not found")
    return HTMLResponse(demo_path.read_text())


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """
    Stream Server-Sent Events as each agent completes.

    Event shapes:
      {"type": "agent_start", "agent": str, "message": str}
      {"type": "agent_done",  "agent": str, "output": str}
      {"type": "agent_error", "agent": str, "message": str}
      {"type": "report",      "content": str}
    """
    if not os.environ.get("MISTRAL_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="MISTRAL_API_KEY environment variable is not set.",
        )

    async def event_stream():
        pipeline = ResearchPipeline(req.topic, req.num_sources)
        async for event in pipeline.run():
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
