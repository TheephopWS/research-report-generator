import logging
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from pipeline import ResearchPipeline

app = FastAPI(
    title="Research Report Generator",
    description="",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    topic: str = Field(..., max_length=500)
    num_sources: int = Field(default=3, ge=2, le=6)


@app.get("/", response_class=HTMLResponse)
async def serve_demo():
    demo_path = Path(__file__).parent / "demo" / "index.html"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo page not found")
    return HTMLResponse(demo_path.read_text())


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if len(req.topic) < 3:
        raise ValueError('Topic must be at least 3 characters long')
        
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY environment variable is not set.",
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
