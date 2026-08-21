# InkFlow-AI — System Architecture

Welcome to the **InkFlow-AI** architecture guide! This document explains how InkFlow-AI turns a simple text prompt into a fully researched, structured, edited, and illustrated technical article.

---

## 📌 Overview

InkFlow-AI is built using **LangGraph**, a framework for building stateful, multi-agent LLM applications. Unlike traditional single-prompt generators, InkFlow-AI breaks article creation into specialized stages executed by discrete agent nodes.

### Core Architecture Highlights
- **Stateful Workflow**: A central state object (`BlogState`) carries article data, research notes, sections, and metadata through every node.
- **Parallel Execution**: Uses LangGraph's `Send()` API to write multiple sections and generate multiple diagrams concurrently.
- **Multi-Model Gateway**: Routes LLM and image calls through a unified provider with automatic fallback chains (e.g., Gemini → OpenAI).
- **Cost & Token Tracking**: Logs token counts, model latency, and cost estimates for every node in real time.

---

## 📐 High-Level Architecture

> 🎨 **Interactive Diagram**: You can open and edit the visual diagram file at [`docs/architecture.excalidraw`](file:///c:/Users/cheru/OneDrive/Desktop/GitHub_Projects/InkFlow-AI/docs/architecture.excalidraw) directly in [Excalidraw](https://excalidraw.com).

```
                       ┌───────────────────────┐
                       │   User Prompt / API   │
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   Input Guardrails    │  ← Security check (regex / injection filter)
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │     Routing Graph     │  ← Determines research mode (Closed/Hybrid/Open)
                       └───────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │ (If external research needed) │
                    ▼                             │
         ┌─────────────────────┐                  │
         │   Research Graph    │                  │
         │ (Tavily Web Search) │                  │
         └──────────┬──────────┘                  │
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │        Planner        │  ← Generates structured outline (Plan schema)
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │     Writing Graph     │  ← Parallel section workers + Senior Editor
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   Publishing Graph    │  ← Image planning → Image gen → Markdown assembly
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │  Final Output Files   │  ← blog.md + images/ + meta.json
                       └───────────┘
```

---

## 🧩 Subgraph Breakdown

InkFlow-AI splits complex tasks into four modular subgraphs:

### 1. Routing Subgraph (`RoutingGraph`)
- **Purpose**: Analyzes the topic prompt to decide how much external research is required.
- **Modes**:
  - `closed_book`: Concept-focused topics (e.g., "Explain Binary Search Trees") using LLM knowledge.
  - `hybrid`: Topics requiring recent context or benchmarking.
  - `open_book`: Deep technical research topics needing extensive web search.

### 2. Research Subgraph (`ResearchGraph`)
- **Purpose**: Gathers up-to-date documentation and code samples.
- **Operation**: Executes parallel web queries via the **Tavily API**, deduplicates search results, and aggregates structured research summaries into `BlogState`.

### 3. Writing Subgraph (`WritingGraph`)
- **Purpose**: Transforms the outline into a cohesive long-form article.
- **Operation**:
  - **Planner Node**: Creates section headings and target word counts.
  - **Worker Nodes**: Uses `Send()` to spawn concurrent section writers.
  - **Merge Node**: Combines section drafts in chronological order.
  - **Senior Editor Node**: Reviews the draft for clarity, tone, code formatting, and smooth transitions.

### 4. Publishing Subgraph (`PublishingGraph`)
- **Purpose**: Adds technical diagrams and formats final Markdown.
- **Operation**:
  - **Image Planner Node**: Identifies key concepts needing visual diagrams.
  - **Image Generator Node**: Generates images concurrently using AI image models.
  - **Formatter Node**: Renumbers figures sequentially (`Figure 1:`, `Figure 2:`), embeds images at relevant sections, and writes output files (`blog.md`, `meta.json`, `images/`).

---

## 🤖 Model Gateway & Fallback System

InkFlow-AI implements a resilient **Model Gateway** (`src/models/gateway.py`) to prevent service disruptions:

```
Primary Call (e.g., Gemini 3.5 Flash)
    │
    ├──► Success ──► Return Result
    │
    └──► Error / Timeout / Rate Limit
            │
            ▼
   Fallback Call (e.g., Gemini 2.5 Pro or GPT-4o-mini)
            │
            ├──► Success ──► Return Result
            └──► Error ──► Fail-safe error handler
```

### Node Model Mapping

| Node | Primary Model | Fallback Model |
| :--- | :--- | :--- |
| **Router** | Gemini 2.5 Flash | Gemini 2.5 Flash Lite / GPT-4o Mini |
| **Research** | Gemini 3.5 Flash | Gemini 2.5 Flash / GPT-4o Mini |
| **Planner** | GPT-4o Mini | Gemini 2.5 Pro / Gemini 3.5 Flash |
| **Writer** | Gemini 3.5 Flash | Gemini 2.5 Pro / GPT-4o Mini |
| **Editor** | Gemini 2.5 Pro | GPT-4o Mini |
| **Image Generator** | Gemini Flash Image | Imagen 3 / GPT Image 1 |

---

## 📊 Cost Observability & Guardrails

### 1. Cost & Latency Tracking (`CostTracker`)
Every node records execution metadata streamed back to the user interface:
- Prompt tokens & completion tokens.
- Node execution latency in milliseconds.
- Calculated cost based on exact per-token pricing tables.
- Stored per run in `outputs/run_<id>/meta.json`.

### 2. Input Guardrails
Before entering the LangGraph state machine, input prompts pass through `src/guardrails/input_guardrails.py`:
- Checks for prompt injection and jailbreak patterns.
- Validates prompt length and formatting.
- Rejects harmful or out-of-scope requests early to save API costs.

---

## 📁 Directory Reference

```
src/
├── config/          # Environment configuration & settings
├── graph/           # Main LangGraph entrypoint (main_graph.py)
├── subgraphs/       # Subgraph definitions (routing, research, writing, publishing)
├── nodes/           # Isolated node logic (router, planner, worker, editor, etc.)
├── models/          # Gateway, model registry, and provider definitions
├── schemas/         # Pydantic domain models and BlogState definition
├── guardrails/      # Input security and prompt sanitization
└── observability/   # Per-node token, latency, and cost tracking
```
