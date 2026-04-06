# Research Report Generator

Deployed on [https://research-report-generator-live.onrender.com](https://research-report-generator-live.onrender.com)

A **5-agent AI pipeline** that synthesises sources into a polished research report, powered entirely by **Mistral's free API**

```
Search Agent → Summarizer → Synthesizer → Critic → Formatter
```

---

## Quick Start

### 1. Get a free Mistral API key
Sign up at **https://console.mistral.ai** → API Keys → Create key.
The free tier includes `mistral-small-latest` with generous rate limits.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
cp .env.example .env
```

### 4. Run the server
```bash
python main.py
```

Open **http://localhost:8000** in your browser.

---

## Changing the model

Edit the `MODEL` constant in `agents/base.py`:

---

## Project Structure

```
research-report-generator/
├── main.py                    # FastAPI app — serves demo + SSE endpoint
├── pipeline.py                # Orchestrates the 5-agent chain
├── agents/
│   ├── __init__.py
│   ├── base.py                # Shared async Mistral client + BaseAgent
│   ├── search_agent.py        # Identifies N sources from model knowledge
│   ├── summarizer_agent.py    # Summarises each source academically
│   ├── synthesizer_agent.py   # Synthesises into a coherent narrative
│   ├── critic_agent.py        # Audits quality; produces revised synthesis
│   └── formatter_agent.py     # Outputs a complete Markdown report
├── demo/
│   └── index.html             # Self-contained demo front-end
├── requirements.txt
├── .env.example
└── README.md
```

---

## Note on web search

Mistral's free API does not include a live web search tool, so the Search Agent draws on the model's trained
knowledge to identify real published sources. For topics after the model's knowledge cutoff, consider pairing with a free search API such as SerpAPI or DuckDuckGo Instant
Answer API and passing results into the Search Agent's prompt.

---

## API

### `POST /api/generate`
Streams [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).

**Request body**
```json
{ "topic": "string", "num_sources": 3 }
```

**Event shapes**
| `type`         | Fields                      |
|----------------|-----------------------------|
| `agent_start`  | `agent`, `message`          |
| `agent_done`   | `agent`, `output`           |
| `agent_error`  | `agent`, `message`          |
| `report`       | `content` (Markdown string) |
