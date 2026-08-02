from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
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


def save_final_run_data(
    run_id: str,
    topic: str,
    markdown: str,
    summary: dict[str, Any] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> Path:
    """
    Save Markdown output and execution metadata.json.
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

    title = topic
    for line in markdown.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("# "):
            title = line_clean[2:].strip()
            break

    meta_file = run_directory / "meta.json"
    meta_payload = {
        "run_id": run_id,
        "topic": topic,
        "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_duration": (summary or {}).get("execution_duration", "0.0s"),
        "total_cost": (summary or {}).get("total_cost", 0.0),
        "total_tokens": (summary or {}).get("total_tokens", 0),
        "summary": summary or {},
        "metrics": metrics or [],
    }

    try:
        meta_file.write_text(
            json.dumps(meta_payload, indent=2),
            encoding="utf-8",
        )
    except Exception as err:
        logger.warning("Could not write meta.json for run %s: %s", run_id, err)

    return output_file


# ---------------------------------------------------------
# LangGraph streaming
# ---------------------------------------------------------
def stream_workflow(topic: str, run_id: str,) -> Generator[str, None, None]:
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
            "id": "input_guardrails",
            "label": "Input Guardrails Validation",
            "status": "running",
            "detail": (
                "Validating user prompt for security, "
                "safety, and structural compliance."
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
                # Input Guardrails
                # =================================================
                if node_name == "input_guardrails":
                    error = node_update.get("error")
                    if error:
                        yield create_sse_event(
                            {
                                "type": "stage",
                                "id": "input_guardrails",
                                "label": "Input Guardrails Validation",
                                "status": "failed",
                                "detail": f"Failed: {error}",
                            }
                        )
                    else:
                        yield create_sse_event(
                            {
                                "type": "stage",
                                "id": "input_guardrails",
                                "label": "Input Guardrails Validation",
                                "status": "completed",
                                "detail": "Input prompt passed security & format checks.",
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

                # =================================================
                # Router
                # =================================================
                elif node_name in ("router", "routing", "route_after_router"):
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
                                "id": "editor",
                                "label": "Senior Editorial Review",
                                "status": "running",
                                "detail": "Senior Editorial Review in progress...",
                            }
                        )

                # =================================================
                # Editorial Review
                # =================================================
                elif node_name == "editor":
                    blog_markdown = str(node_update.get("blog_markdown", ""))
                    if blog_markdown:
                        final_markdown = blog_markdown

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "editor",
                            "label": "Senior Editorial Review",
                            "status": "completed",
                            "detail": "Senior Editorial Review completed.",
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "output_guardrails",
                            "label": "Output Guardrails Validation",
                            "status": "running",
                            "detail": "Validating generated markdown for placeholders, leaks, and secrets...",
                        }
                    )

                # =================================================
                # Output Guardrails
                # =================================================
                elif node_name == "output_guardrails":
                    warnings = node_update.get("guardrail_warnings", [])
                    errors = node_update.get("guardrail_errors", [])
                    detail_msg = "Output validation completed cleanly."
                    if errors or warnings:
                        detail_msg = f"Output validation completed with {len(errors)} error(s), {len(warnings)} warning(s)."

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "output_guardrails",
                            "label": "Output Guardrails Validation",
                            "status": "completed",
                            "detail": detail_msg,
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "formatter",
                            "label": "Markdown Formatter",
                            "status": "running",
                            "detail": "Standardizing Markdown presentation...",
                        }
                    )

                # =================================================
                # Markdown Formatter
                # =================================================
                elif node_name in ("markdown_formatter", "formatter"):
                    blog_markdown = str(node_update.get("blog_markdown", ""))
                    if blog_markdown:
                        final_markdown = blog_markdown

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "formatter",
                            "label": "Markdown Formatter",
                            "status": "completed",
                            "detail": "Markdown presentation standardized.",
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "image_planner",
                            "label": "Image Planner",
                            "status": "running",
                            "detail": "Planning technical visual illustrations...",
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
                            "type": "stage",
                            "id": "image_planner",
                            "label": "Image Planner",
                            "status": "completed",
                            "detail": f"Planned {len(images)} technical visual illustration(s).",
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "image_generator",
                            "label": "Generate Images",
                            "status": "running",
                            "detail": "Generating technical images in parallel...",
                        }
                    )

                # =================================================
                # Image Generator / Publishing
                # =================================================
                elif node_name in ("image_generator", "image_worker", "assemble_publishing", "publishing", "generate_and_place_images"):
                    generated_final = (
                        node_update.get("final_markdown")
                        or node_update.get("blog_markdown")
                        or node_update.get("final")
                    )

                    if generated_final:
                        final_markdown = str(generated_final)

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "image_generator",
                            "label": "Generate Images",
                            "status": "completed",
                            "detail": "Technical images generated.",
                        }
                    )

                    yield create_sse_event(
                        {
                            "type": "stage",
                            "id": "completed",
                            "label": "Completed",
                            "status": "running",
                            "detail": "Finalizing Markdown article...",
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
            g_warn = getattr(state_values, "guardrail_warnings", []) or []
            g_err = getattr(state_values, "guardrail_errors", []) or []
        elif isinstance(state_values, dict):
            metrics = state_values.get("metrics", [])
            plan_obj = state_values.get("plan")
            evidence_obj = state_values.get("evidence")
            g_warn = state_values.get("guardrail_warnings", []) or []
            g_err = state_values.get("guardrail_errors", []) or []
        else:
            metrics = []
            plan_obj = None
            evidence_obj = None
            g_warn = []
            g_err = []

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

        guardrail_violations = len(g_warn) + len(g_err)

        deduped_metrics = cost_tracker.deduplicate_metrics(metrics or [])
        exec_summary = cost_tracker.calculate_summary(
            metrics=deduped_metrics,
            duration_seconds=duration_seconds,
            sections_count=sections_count,
            sources_count=sources_count,
            guardrail_violations=guardrail_violations,
        )

        save_final_run_data(
            run_id=run_id,
            topic=topic,
            markdown=final_markdown,
            summary=exec_summary,
            metrics=deduped_metrics,
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
@app.get("/",response_class=HTMLResponse,)
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


# ---------------------------------------------------------
# History Endpoints (Disk-based output folder enumeration)
# ---------------------------------------------------------
@app.get("/api/history")
def list_history():
    """
    List all previously generated blog runs from data/outputs.
    """
    history_items: list[dict[str, Any]] = []

    if not OUTPUTS_DIR.exists():
        return history_items

    for run_dir in OUTPUTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        blog_file = run_dir / "blog.md"
        meta_file = run_dir / "meta.json"

        if not blog_file.exists():
            continue

        title = "Untitled Workflow Run"
        created_at = datetime.fromtimestamp(
            blog_file.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        execution_duration = "—"
        total_cost = 0.0
        total_tokens = 0

        # Extract title from blog.md H1 header
        try:
            markdown_content = blog_file.read_text(encoding="utf-8")
            for line in markdown_content.splitlines():
                clean_line = line.strip()
                if clean_line.startswith("# "):
                    title = clean_line[2:].strip()
                    break
        except Exception:
            pass

        # Load metadata if present
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    title = meta.get("title") or meta.get("topic") or title
                    created_at = meta.get("created_at") or created_at
                    execution_duration = meta.get("execution_duration") or execution_duration
                    total_cost = meta.get("total_cost", 0.0)
                    total_tokens = meta.get("total_tokens", 0)
            except Exception:
                pass

        history_items.append(
            {
                "run_id": run_id,
                "title": title,
                "created_at": created_at,
                "execution_duration": execution_duration,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
            }
        )

    # Sort descending by creation timestamp
    history_items.sort(key=lambda x: x["created_at"], reverse=True)
    return history_items


@app.get("/api/history/{run_id}")
def get_history_item(run_id: str):
    """
    Load stored markdown, images, summary, and details metrics for a historical run.
    """
    safe_run_id = "".join(
        ch for ch in run_id if ch.isalnum() or ch in {"-", "_"}
    )
    if safe_run_id != run_id:
        raise HTTPException(status_code=400, detail="Invalid run ID.")

    run_dir = OUTPUTS_DIR / safe_run_id
    blog_file = run_dir / "blog.md"
    meta_file = run_dir / "meta.json"

    if not blog_file.exists():
        raise HTTPException(status_code=404, detail="Run output not found.")

    markdown = blog_file.read_text(encoding="utf-8")
    summary: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []

    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                summary = meta.get("summary", {})
                metrics = meta.get("metrics", [])
        except Exception:
            pass

    return {
        "run_id": safe_run_id,
        "markdown": markdown,
        "summary": summary,
        "metrics": metrics,
        "download_url": f"/api/runs/{safe_run_id}/download",
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )