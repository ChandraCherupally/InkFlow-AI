# InkFlow-AI

**AI-powered technical content orchestration engine for generating publication-ready engineering articles.**

InkFlow-AI is a production-grade multi-agent system built on LangGraph that autonomously researches, plans, writes, edits, and illustrates long-form technical blog posts — with full cost observability and a real-time streaming web UI.

---

## Overview

InkFlow-AI orchestrates a stateful, multi-agent pipeline that transforms a simple topic prompt into a structured, editor-polished Markdown article with contextually placed AI-generated technical illustrations.

The system is designed around three principles:

- **Separation of concerns** — each workflow stage is an isolated, independently testable node.
- **Provider agnosticism** — all LLM and image model calls route through a unified gateway with automatic fallback chains.
- **Full observability** — every node emits structured cost and latency metrics streamed live to the UI dashboard.

---

## Architecture

```
User Prompt
     │
     ▼
Input Guardrails          ← Prompt injection, length, and format validation
     │
     ▼
Router                    ← Decides: closed_book | hybrid | open_book
     │
     ├── (needs research) ──► Research Graph   ← Parallel Tavily web search
     │
     ▼
Planner                   ← Generates structured article outline (Plan schema)
     │
     ▼
Writing Graph             ← Parallel section workers + Senior Editor pass
     │
     ▼
Publishing Graph          ← Image Planner → Parallel Image Generation → Markdown Assembly
     │
     ▼
Output: blog.md + images/ + meta.json
```

### Subgraphs

| Subgraph | Responsibility |
| :--- | :--- |
| `RoutingGraph` | Classifies topic intent and determines whether external research is needed |
| `ResearchGraph` | Executes parallel Tavily searches and deduplicates evidence |
| `WritingGraph` | Dispatches parallel `Send()` section writers, then runs a Senior Editor review |
| `PublishingGraph` | Plans illustrations, generates images in parallel, assembles final Markdown |

### Node-to-Model Assignment

| Node | Primary Model | Fallbacks |
| :--- | :--- | :--- |
| Router | `gemini-2.5-flash` | `gemini-2.5-flash-lite`, `gpt-5-mini` |
| Research | `gemini-3.5-flash` | `gemini-2.5-flash`, `gpt-54-mini` |
| Planner | `gpt-54-mini` | `gemini-2.5-pro`, `gemini-3.5-flash` |
| Writer | `gemini-3.5-flash` | `gemini-2.5-pro`, `gpt-54-mini` |
| Editor | `gemini-2.5-pro` | `gpt-54-mini` |
| Image Planner | `gemini-3.5-flash` | `gemini-2.5-pro`, `gpt-54-mini` |
| Image Generator | `gemini-flash-image` | `imagen-3`, `gpt-image-1` |

---

## Features

- **Multi-agent LangGraph workflow** with stateful `BlogState` passed across all nodes and subgraphs.
- **Intelligent routing** — classifies topics and selects between `closed_book`, `hybrid`, or `open_book` research modes.
- **Parallel section writing** via LangGraph `Send()` for concurrent section generation.
- **Senior editorial review** — a dedicated editor node polishes the full draft for publication quality.
- **Contextual image placement** — semantic keyword scoring ensures diagrams appear in their relevant body sections, never in concluding blocks.
- **Sequential figure renumbering** — figure captions are always normalized to `Figure 1:`, `Figure 2:`, etc., in document order.
- **Context-aware article endings** — article conclusion type is deterministically chosen based on article category (`## Key Takeaways`, `## Choosing the Right Approach`, `## Final Thoughts`, etc.).
- **Input guardrails** — deterministic regex-based prompt injection and jailbreak detection.
- **Real-time SSE streaming** — node-by-node progress streamed to the browser dashboard.
- **Full cost observability** — per-node token counts, latency, estimated costs, and model metadata tracked and displayed live.
- **Custom image pricing** — custom pricing table for newly released image models not covered by LiteLLM.
- **Run history** — all outputs persisted to `outputs/<run_id>/` with `blog.md`, `images/`, and `meta.json`.

---

## Project Structure

```
InkFlow-AI/
├── app.py                        # FastAPI server, SSE streaming, run management
├── src/
│   ├── config/
│   │   └── settings.py           # Environment, paths, model defaults, logging config
│   ├── graph/
│   │   └── main_graph.py         # Root LangGraph StateGraph with all subgraph wiring
│   ├── subgraphs/
│   │   ├── routing_graph.py      # Topic classification subgraph
│   │   ├── research_graph.py     # Parallel Tavily research subgraph
│   │   ├── writing_graph.py      # Parallel section writer + editor subgraph
│   │   └── publishing_graph.py   # Image planning, generation, and Markdown assembly
│   ├── nodes/
│   │   ├── router.py             # Topic routing node
│   │   ├── research.py           # Tavily web search node
│   │   ├── planner.py            # Article outline generation node
│   │   ├── worker.py             # Individual section writer node
│   │   ├── editor.py             # Senior editorial review node
│   │   ├── image_planner.py      # Illustration planning node
│   │   ├── image_generator.py    # AI image generation node
│   │   ├── merge.py              # Section merge node
│   │   └── formatter.py          # Markdown formatting node
│   ├── models/
│   │   ├── gateway.py            # Unified LLM & image gateway with fallback chains
│   │   ├── registry.py           # Node-to-model registry with capability validation
│   │   ├── providers.py          # ModelProfile definitions per provider
│   │   └── types.py              # NodeType enum and ModelProfile dataclass
│   ├── schemas/
│   │   ├── models.py             # Pydantic domain schemas (Plan, Task, ImageSpec, etc.)
│   │   └── state.py              # LangGraph BlogState TypedDict
│   ├── prompts/
│   │   └── prompts.py            # SystemPrompts for all LLM nodes
│   ├── tools/
│   │   ├── markdown.py           # MarkdownBuilder: image placement and figure renumbering
│   │   └── image_generator.py    # Image generation tool wrapping the gateway
│   ├── observability/
│   │   └── cost_tracker.py       # Per-node cost, token, and latency metric tracking
│   └── guardrails/
│       └── input_guardrails.py   # Input validation and prompt injection detection
├── tests/
│   └── test_app.py               # Core pytest suite (API, Markdown, Cost Tracker)
├── outputs/                      # Generated run outputs (auto-created)
│   └── run_<id>/
│       ├── blog.md
│       ├── images/
│       └── meta.json
├── static/                       # Frontend static assets
├── templates/                    # Jinja2 HTML templates
├── pyproject.toml
└── requirements.txt
```

---

## Requirements

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or pip

### API Keys Required

| Key | Purpose |
| :--- | :--- |
| `GOOGLE_API_KEY` | Gemini LLM and image generation (primary provider) |
| `OPENAI_API_KEY` | OpenAI LLM and image fallbacks |
| `TAVILY_API_KEY` | Web research for `hybrid` and `open_book` routing modes |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ChandraCherupally/InkFlow-AI.git
cd InkFlow-AI
```

### 2. Create a virtual environment and install dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional model overrides
DEFAULT_LLM=gemini-3.1-flash
DEFAULT_IMAGE_MODEL=gemini-3.1-flash-image
```

### 4. Start the server

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## Usage

### Web UI

1. Enter a technical topic (e.g., `Implementing RAG with LangGraph`, `PCA for Dimensionality Reduction`).
2. Click **Generate** — the workflow streams live progress node-by-node.
3. Review the generated article, cost dashboard, and AI illustrations.
4. Browse previous runs from the **Run History** panel.

### API

**Generate an article:**

```bash
curl -X POST http://127.0.0.1:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Hybrid Search: BM25 and Vector Search"}'
```

**Health check:**

```bash
curl http://127.0.0.1:8000/api/health
```

**Run history:**

```bash
curl http://127.0.0.1:8000/api/history
```

---

## Output Structure

Each completed run is saved to `outputs/run_<timestamp>_<id>/`:

```
outputs/run_20260803_152439_fd76433a/
├── blog.md          # Final publication-ready Markdown article
├── images/
│   ├── diagram_1.png
│   └── diagram_2.png
└── meta.json        # Execution metadata: models used, costs, latency, image specs
```

---

## Running Tests

```bash
uv pip install pytest httpx
python -m pytest tests/ -v
```

The test suite covers:

| Test | What It Verifies |
| :--- | :--- |
| `test_api_endpoints` | `/api/health` and `/api/history` return correct responses |
| `test_markdown_builder` | Article assembly, image placement, and sequential figure renumbering |
| `test_cost_tracker_summary` | Cost aggregation, token totals, and most-expensive-node calculation |

---

## Cost Observability

InkFlow-AI tracks execution costs at node granularity. After each run, `meta.json` contains:

- Per-node: model used, provider, prompt/completion tokens, latency (ms), estimated cost, fallback status
- Workflow summary: total cost, total tokens, images generated, most expensive node, unique models used

Image model costs use a custom pricing table (independent of LiteLLM) to support newly released models not yet covered by LiteLLM pricing.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Orchestration | LangGraph 1.x |
| LLM Abstraction | LangChain + LiteLLM |
| LLM Providers | Google Gemini (primary), OpenAI (fallback) |
| Image Generation | Google GenAI SDK (Vertex AI), Imagen 3, GPT Image 1 |
| Web Research | Tavily |
| API Server | FastAPI + Uvicorn |
| Frontend Streaming | Server-Sent Events (SSE) |
| Schemas | Pydantic v2 |
| Testing | pytest |
| Runtime | Python 3.12 |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
