# InkFlow-AI

> **Production-grade, AI-powered technical content engine built with LangGraph, FastAPI, Docker, and AWS CI/CD.**

InkFlow-AI autonomously researches, plans, writes, edits, and illustrates publication-ready engineering articles with full cost transparency, real-time streaming, and robust fallback models.

---

## 🚀 Key Features

- **Multi-Agent Orchestration**: Built on **LangGraph** with stateful execution across specialized subgraphs (`Routing`, `Research`, `Writing`, and `Publishing`).
- **Dynamic Research Modes**: Classifies topics automatically into `closed_book`, `hybrid`, or `open_book` research pipelines.
- **Parallel Execution**: Spawns concurrent section writers and AI diagram generators using LangGraph's `Send()` framework.
- **Senior Editorial Pass**: Dedicates an editorial node to polish content, refine code snippets, and ensure smooth transitions.
- **AI Visual Diagrams**: Generates contextual technical diagrams, renumbers captions sequentially (`Figure 1:`, `Figure 2:`), and embeds them automatically.
- **Real-Time SSE Streaming**: Streams node progress and live generation logs to a clean web interface.
- **Cost & Latency Observability**: Tracks token usage, execution latency (ms), model pricing, and most expensive nodes per run.
- **Production CI/CD & Containerization**: Fully containerized with Docker and deployed to AWS EC2 via GitHub Actions.

---

## 🏗️ Architecture Overview

```
                      ┌────────────────────────┐
                      │   Topic Prompt / API   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │    Input Guardrails    │  ← Security & injection check
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │     Routing Graph      │  ← Determines research depth
                      └───────────┬────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │ (If web research required)  │
                   ▼                             │
        ┌────────────────────┐                   │
        │   Research Graph   │                   │  ← Tavily search engine
        └──────────┬─────────┘                   │
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │     Planner Node       │  ← Generates structured outline
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │     Writing Graph      │  ← Parallel writers + Senior Editor
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │    Publishing Graph    │  ← Image planning → Image gen → Markdown assembly
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │  Final Output Files    │  ← blog.md + images/ + meta.json
                      └────────────────────────┘
```

👉 *For detailed subgraph breakdown, state schema definitions, and model fallback chains, see the [Architecture Documentation](docs/ARCHITECTURE.md).*

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- API Keys: `GEMINI_API_KEY`, `OPENAI_API_KEY`, and `TAVILY_API_KEY`

### 2. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/ChandraCherupally/InkFlow-AI.git
cd InkFlow-AI

# Create virtual environment and install dependencies using uv
uv venv
uv sync
```

### 3. Environment Variables
Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Run the Server
```bash
uv run uvicorn app:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser to launch the Web UI.

---

## 🐳 Docker Deployment

InkFlow-AI is fully containerized for seamless deployment.

```bash
# Run with Docker Compose
docker compose -f docker/docker-compose.yml up -d --build
```

The containerized service runs on `http://localhost:8000` with output files persisted locally in `./outputs`.

---

## 🔄 CI/CD & AWS Infrastructure

InkFlow-AI utilizes a two-tier sequential **GitHub Actions** CI/CD pipeline targeting **AWS EC2**:

1. **Continuous Integration (`ci.yml`)**:
   - Triggers on push and pull requests.
   - Executes `pytest` unit tests and validates Docker BuildKit layer compilation.
2. **Continuous Deployment (`cd.yml`)**:
   - Triggered automatically when CI succeeds on `main`.
   - Runs on a **self-hosted AWS EC2 runner**.
   - Builds & pushes tagged images to Docker Hub.
   - Restarts containers using `docker compose` on AWS EC2.
   - Performs production health checks against `/api/health`.

👉 *For step-by-step setup guides, secret configuration, and Twelve-Factor security practices, see the [Deployment Guide](docs/DEPLOYMENT.md).*

---

## 📡 API Usage

### Generate an Article
```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Building Multi-Agent Systems with LangGraph"}'
```

### Check System Health
```bash
curl http://localhost:8000/api/health
```

### View Run History
```bash
curl http://localhost:8000/api/history
```

---

## 📁 Output Structure

Every run creates a dedicated output folder inside `outputs/run_<timestamp>_<id>/`:

```
outputs/run_20260803_152439_fd76433a/
├── blog.md          # Final publication-ready Markdown article
├── images/          # Generated technical diagrams (diagram_1.png, diagram_2.png)
└── meta.json        # Execution metadata: models used, costs, latency, tokens
```

---

## 🧪 Running Tests

```bash
uv run pytest tests/ -v
```

The test suite covers API health/history endpoints, Markdown builder renumbering logic, and cost tracking metric aggregations.

---

## 📖 Documentation Index

- [Architecture Guide](docs/ARCHITECTURE.md): Multi-agent LangGraph workflows, routing logic, model gateway, fallback chains, and cost tracking.
- [Vertical Architecture Diagram](docs/architecture.excalidraw): Editable Excalidraw visual diagram of the system pipeline in vertical orientation.
- [Deployment & CI/CD Guide](docs/DEPLOYMENT.md): Local environment setup, Docker Compose, AWS EC2 deployment, GitHub Actions workflows, and Twelve-Factor security.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
