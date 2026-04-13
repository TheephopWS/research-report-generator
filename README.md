# Research Report Generator

Deployed on [https://research-report-generator-live.onrender.com](https://research-report-generator-live.onrender.com)

---

## Quick Start

### 1. Get a DeepSeek API key
Sign up at **https://platform.deepseek.com** → API Keys → Create key.
This repo will use `deepseek-chat` model.

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
│   ├── base.py                # Shared async DeepSeek/OpenAI-compatible client + BaseAgent
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
