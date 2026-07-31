from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import logging
from pathlib import Path
from typing import Any, Generator
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

import time
from src.observability.cost_tracker import cost_tracker

# ---------------------------------------------------------
# Import the compiled LangGraph workflow from src/
# ---------------------------------------------------------
from src.graph.main_graph import build_main_graph as build_graph
from src.schemas.state import BlogState

workflow = build_graph()


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
IMAGES_DIR = BASE_DIR / "data" / "images"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("inkflow")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------
app = FastAPI(
    title="InkFlow-AI",
    description="AI-powered technical content orchestration using LangGraph.",
    version="1.0.0",
)


# ---------------------------------------------------------
# Static files
# ---------------------------------------------------------
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

app.mount(
    "/images",
    StaticFiles(directory=IMAGES_DIR),
    name="images",
)


# ---------------------------------------------------------
# Templates
# ---------------------------------------------------------
templates = Jinja2Templates(
    directory=TEMPLATES_DIR,
)


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------
class AgentRunRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The technical blog topic.",
    )


# ---------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------
def make_serializable(value: Any) -> Any:
    """
    Recursively convert Pydantic models, dataclasses, and other values
    into JSON-compatible Python values.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return make_serializable(asdict(value))

    if hasattr(value, "model_dump"):
        return make_serializable(value.model_dump())

    if hasattr(value, "dict"):
        return make_serializable(value.dict())

    if isinstance(value, dict):
        return {
            str(key): make_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_serializable(item)
            for item in value
        ]

    return value


def create_sse_event(
    payload: dict[str, Any],
    event_name: str | None = None,
) -> str:
    """
    Convert a dictionary to a Server-Sent Event message.
    """
    encoded_payload = json.dumps(
        jsonable_encoder(payload),
        ensure_ascii=False,
    )

    lines: list[str] = []

    if event_name:
        lines.append(f"event: {event_name}")

    lines.append(f"data: {encoded_payload}")

    return "\n".join(lines) + "\n\n"


def normalize_stream_chunk(
    chunk: Any,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """
    Normalize LangGraph streaming chunks.
    """
    if (
        isinstance(chunk, tuple)
        and len(chunk) == 2
        and isinstance(chunk[1], dict)
    ):
        raw_namespace = chunk[0] or ()
        namespace = tuple(str(item) for item in raw_namespace)
        return namespace, chunk[1]

    if isinstance(chunk, dict):
        return (), chunk

    return (), {}


def get_plan_task_map(
    plan: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """
    Create a task lookup using each task ID.
    """
    task_map: dict[int, dict[str, Any]] = {}
    tasks = plan.get("tasks", [])

    if not isinstance(tasks, list):
        return task_map

    for task in tasks:
        if not isinstance(task, dict):
            continue

        try:
            task_id = int(task["id"])
        except (KeyError, TypeError, ValueError):
            continue

        task_map[task_id] = task

    return task_map


def save_final_markdown(
    run_id: str,
    markdown: str,
) -> Path:
    """
    Save a copy of the generated Markdown output.
    """
    run_directory = OUTPUTS_DIR / run_id
    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = run_directory / "blog.md"

    output_file.write_text(
        markdown,
        encoding="utf-8",
    )

    return output_file


# ---------------------------------------------------------
# LangGraph streaming
# ---------------------------------------------------------
def stream_workflow(
    topic: str,
    run_id: str,
) -> Generator[str, None, None]:
    """
    Run the InkFlow-AI LangGraph workflow and stream execution updates.
    """
    config = {
        "configurable": {
            "thread_id": run_id,
        }
    }

    workflow_input = BlogState(topic=topic)
    start_time_total = time.perf_counter()

    task_map: dict[int, dict[str, Any]] = {}
    completed_task_ids: set[int] = set()

    final_markdown = ""

    yield create_sse_event(
        {
            "type": "run_started",
            "run_id": run_id,
            "topic": topic,
        }
    )

    yield create_sse_event(
        {
            "type": "stage",
            "id": "router",
            "label": "Route User Request",
            "status": "running",
            "detail": (
                "Determining whether the topic requires "
                "current web research."
            ),
        }
    )

    try:
        stream = workflow.stream(
            workflow_input,
            config=config,
            stream_mode="updates",
            subgraphs=True,
        )

        for raw_chunk in stream:
            namespace, updates = normalize_stream_chunk(raw_chunk)

            if not updates:
                continue

            for node_name, raw_node_update in updates.items():
                node_update = make_serializable(raw_node_update)

                if not isinstance(node_update, dict):
                    node_update = {}

                # =================================================
                # Router
                # =================================================
                if node_name in ("router", "routing", "route_after_router"):
                    mode = str(
                        node_update.get("routing_mode")
                        or node_update.get("mode")
                        or "closed_book"
                    )

                    needs_research = bool(
                        node_update.get("needs_research", False)
                    )

                    queries = (
                        node_update.get("search_queries")
                        or node_update.get("queries")
                        or []
                    )

                    yield create_sse_event(
                        {
                            "type": "routing",
                            "mode": mode,
                            "needs_research": needs_research,
                            "queries": queries,
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "router",
                            "label": "Route User Request",
                            "status": "completed",
                            "detail": f"Selected {mode.replace('_', ' ')} mode.",
                        }
                    )

                    if needs_research:
                        yield create_sse_event(
                            {
                                "type": "stage",
                                "id": "research",
                                "label": "Retrieve Supporting Evidence",
                                "status": "running",
                                "detail": (
                                    "Searching the web and preparing "
                                    "a deduplicated evidence pack."
                                ),
                            }
                        )
                    else:
                        yield create_sse_event(
                            {
                                "type": "stage",
                                "id": "orchestrator",
                                "label": "Build Structured Article Plan",
                                "status": "running",
                                "detail": (
                                    "Creating the article structure, "
                                    "goals and writing tasks."
                                ),
                            }
                        )

                # =================================================
                # Research
                # =================================================
                elif node_name in ("research", "merge_research", "tavily_worker"):
                    raw_evidence = node_update.get("evidence", {})
                    if isinstance(raw_evidence, dict):
                        evidence_list = raw_evidence.get("evidence", [])
                    elif isinstance(raw_evidence, list):
                        evidence_list = raw_evidence
                    else:
                        evidence_list = []

                    if evidence_list:
                        yield create_sse_event(
                            {
                                "type": "research_complete",
                                "count": len(evidence_list),
                                "evidence": evidence_list[:12],
                            }
                        )

                        yield create_sse_event(
                            {
                                "type": "stage",
                                "id": "research",
                                "label": "Retrieve Supporting Evidence",
                                "status": "completed",
                                "detail": f"Prepared {len(evidence_list)} deduplicated sources.",
                            }
                        )

                        yield create_sse_event(
                            {
                                "type": "stage",
                                "id": "orchestrator",
                                "label": "Build Structured Article Plan",
                                "status": "running",
                                "detail": (
                                    "Creating sections, goals, bullets "
                                    "and target word counts."
                                ),
                            }
                        )

                # =================================================
                # Planner / Orchestrator
                # =================================================
                elif node_name in ("planner", "orchestrator"):
                    plan = node_update.get("plan", {})

                    if not isinstance(plan, dict):
                        plan = {}

                    task_map = get_plan_task_map(plan)

                    yield create_sse_event(
                        {
                            "type": "plan",
                            "plan": plan,
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "orchestrator",
                            "label": "Build Structured Article Plan",
                            "status": "completed",
                            "detail": f"Created {len(task_map)} article sections.",
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "workers",
                            "label": "Generate Content Sections",
                            "status": "running",
                            "detail": "Section workers are writing the article.",
                        }
                    )

                # =================================================
                # Writer / Worker
                # =================================================
                elif node_name in ("writer", "worker", "worker_section", "writing", "assemble_sections"):
                    blog_markdown = str(node_update.get("blog_markdown", ""))
                    if blog_markdown:
                        final_markdown = blog_markdown

                    tasks = list(task_map.values())
                    for idx, task in enumerate(tasks, 1):
                        task_id = int(task.get("id", idx))
                        if task_id not in completed_task_ids:
                            completed_task_ids.add(task_id)
                            yield create_sse_event(
                                {
                                    "type": "section_complete",
                                    "task_id": task_id,
                                    "title": task.get("title", f"Section {task_id}"),
                                    "markdown": "",
                                    "completed": len(completed_task_ids),
                                    "total": len(task_map),
                                }
                            )

                    if blog_markdown or (task_map and len(completed_task_ids) >= len(task_map)):
                        yield create_sse_event(
                            {
                                "type": "stage",
                                "id": "workers",
                                "label": "Generate Content Sections",
                                "status": "completed",
                                "detail": f"Completed all {len(task_map)} sections.",
                            }
                        )

                        yield create_sse_event(
                            {
                                "type": "stage",
                                "id": "reducer",
                                "label": "Assemble Final Markdown",
                                "status": "running",
                                "detail": "Merging sections and planning visuals.",
                            }
                        )

                # =================================================
                # Editorial Review & Markdown Formatter
                # =================================================
                elif node_name in ("editor", "markdown_formatter"):
                    blog_markdown = str(node_update.get("blog_markdown", ""))
                    if blog_markdown:
                        final_markdown = blog_markdown

                    label = "Senior Editorial Review" if node_name == "editor" else "Standardizing Markdown Presentation"
                    yield create_sse_event(
                        {
                            "type": "substage",
                            "id": node_name,
                            "label": label,
                            "status": "completed",
                            "namespace": list(namespace),
                        }
                    )

                # =================================================
                # Image Planner
                # =================================================
                elif node_name in ("image_planner", "decide_images"):
                    image_plan = node_update.get("image_plan", {})
                    if isinstance(image_plan, dict):
                        images = image_plan.get("images", [])
                    else:
                        images = []

                    yield create_sse_event(
                        {
                            "type": "images_planned",
                            "count": len(images),
                            "images": images,
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "substage",
                            "id": "decide_images",
                            "label": "Plan Technical Illustrations",
                            "status": "completed",
                            "namespace": list(namespace),
                        }
                    )

                # =================================================
                # Image Generator / Publishing
                # =================================================
                elif node_name in ("image_generator", "image_worker", "assemble_publishing", "publishing", "generate_and_place_images", "reducer", "merge_content"):
                    generated_final = (
                        node_update.get("final_markdown")
                        or node_update.get("blog_markdown")
                        or node_update.get("final")
                    )

                    if generated_final:
                        final_markdown = str(generated_final)

                    if node_name in ("image_generator", "publishing", "generate_and_place_images"):
                        yield create_sse_event(
                            {
                                "type": "substage",
                                "id": "generate_images",
                                "label": "Generate Technical Images",
                                "status": "completed",
                                "namespace": list(namespace),
                            }
                        )

        # -----------------------------------------------------
        # Final output check
        # -----------------------------------------------------
        if not final_markdown:
            snapshot = workflow.get_state(config)
            state_values = getattr(snapshot, "values", {})

            if is_dataclass(state_values):
                final_markdown = getattr(state_values, "final_markdown", "") or getattr(state_values, "blog_markdown", "")
            elif isinstance(state_values, dict):
                final_markdown = str(
                    state_values.get("final_markdown")
                    or state_values.get("blog_markdown")
                    or state_values.get("final")
                    or ""
                )

        if not final_markdown:
            raise RuntimeError("The workflow completed but did not return final Markdown.")

        save_final_markdown(
            run_id=run_id,
            markdown=final_markdown,
        )

        yield create_sse_event(
            {
                "type": "stage",
                "id": "reducer",
                "label": "Assemble Final Markdown",
                "status": "completed",
                "detail": "The final Markdown article is ready.",
            }
        )

        # Calculate & Stream Real-Time Execution Summary & Metrics
        duration_seconds = time.perf_counter() - start_time_total
        snapshot = workflow.get_state(config)
        state_values = getattr(snapshot, "values", {})

        if is_dataclass(state_values):
            metrics = getattr(state_values, "metrics", [])
            plan_obj = getattr(state_values, "plan", None)
            evidence_obj = getattr(state_values, "evidence", None)
        elif isinstance(state_values, dict):
            metrics = state_values.get("metrics", [])
            plan_obj = state_values.get("plan")
            evidence_obj = state_values.get("evidence")
        else:
            metrics = []
            plan_obj = None
            evidence_obj = None

        if isinstance(plan_obj, dict):
            sections_count = len(plan_obj.get("tasks", []))
        elif plan_obj and hasattr(plan_obj, "tasks"):
            sections_count = len(plan_obj.tasks)
        else:
            sections_count = 0

        if isinstance(evidence_obj, dict):
            sources_count = len(evidence_obj.get("evidence", []))
        elif evidence_obj and hasattr(evidence_obj, "evidence"):
            sources_count = len(evidence_obj.evidence)
        else:
            sources_count = 0

        deduped_metrics = cost_tracker.deduplicate_metrics(metrics or [])
        exec_summary = cost_tracker.calculate_summary(
            metrics=deduped_metrics,
            duration_seconds=duration_seconds,
            sections_count=sections_count,
            sources_count=sources_count,
        )

        try:
            workflow.update_state(config, {"execution_summary": exec_summary})
        except Exception:
            pass

        yield create_sse_event(
            {
                "type": "summary",
                "summary": exec_summary,
                "metrics": deduped_metrics,
            }
        )

        yield create_sse_event(
            {
                "type": "final",
                "run_id": run_id,
                "markdown": final_markdown,
                "download_url": f"/api/runs/{run_id}/download",
            }
        )

        yield create_sse_event(
            {
                "type": "done",
                "run_id": run_id,
            }
        )

    except GeneratorExit:
        logger.info("Browser disconnected from run %s", run_id)
        raise

    except Exception as error:
        logger.exception("Workflow run %s failed", run_id)
        yield create_sse_event(
            {
                "type": "error",
                "run_id": run_id,
                "message": str(error),
            }
        )


# ---------------------------------------------------------
# Page endpoint
# ---------------------------------------------------------
@app.get(
    "/",
    response_class=HTMLResponse,
)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "page_title": "InkFlow-AI",
        },
    )


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "workflow": "loaded",
    }


# ---------------------------------------------------------
# Agent execution endpoint
# ---------------------------------------------------------
@app.post("/api/run")
def run_agent(request_data: AgentRunRequest):
    topic = request_data.topic.strip()

    if len(topic) < 3:
        raise HTTPException(
            status_code=422,
            detail="Please provide a valid topic.",
        )

    run_id = uuid.uuid4().hex

    return StreamingResponse(
        stream_workflow(
            topic=topic,
            run_id=run_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------
# Markdown download endpoint
# ---------------------------------------------------------
@app.get("/api/runs/{run_id}/download")
def download_markdown(run_id: str):
    safe_run_id = "".join(
        character
        for character in run_id
        if character.isalnum()
        or character in {"-", "_"}
    )

    if safe_run_id != run_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid run ID.",
        )

    output_file = OUTPUTS_DIR / safe_run_id / "blog.md"

    if not output_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Generated Markdown file was not found.",
        )

    return FileResponse(
        path=output_file,
        media_type="text/markdown",
        filename=f"inkflow-blog-{safe_run_id[:8]}.md",
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )