"""
Core Test Suite for InkFlow-AI.

Tests:
1. test_api_endpoints: Verify FastAPI /api/health and /api/history endpoints.
2. test_markdown_builder: Verify markdown assembly, image placement, and sequential figure renumbering.
3. test_cost_tracker_summary: Verify workflow cost aggregation and metric summary calculations.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from src.observability.cost_tracker import cost_tracker
from src.schemas.models import GeneratedImage, Plan
from src.tools.markdown import markdown_builder

client = TestClient(app)


# ---------------------------------------------------------
# Test 1: API Endpoints Health & History
# ---------------------------------------------------------
def test_api_endpoints() -> None:
    """Verify that /api/health and /api/history respond cleanly with status 200."""
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    health_json = health_res.json()
    assert health_json.get("status") == "ok"
    assert health_json.get("workflow") == "loaded"

    history_res = client.get("/api/history")
    assert history_res.status_code == 200
    assert isinstance(history_res.json(), list)


# ---------------------------------------------------------
# Test 2: Markdown Builder & Figure Renumbering
# ---------------------------------------------------------
def test_markdown_builder() -> None:
    """Verify markdown builder populates article sections and normalizes figure captions sequentially."""
    plan = Plan(
        blog_title="Test Production Article",
        subtitle="A test subtitle explaining the system architecture in simple words.",
        audience="Engineers",
        tone="Technical",
        blog_kind="guide",
    )

    sections = [
        "## Architectural Overview\n\nBuilding microservices requires decoupling.",
        "## System Design Mechanics\n\nMessaging queues provide backpressure handling.",
    ]

    images = [
        GeneratedImage(
            filename="arch_diagram.png",
            path="images/arch_diagram.png",
            alt="System architecture diagram",
            caption="Figure 2: Comprehensive system architecture blueprint",
        ),
        GeneratedImage(
            filename="queue_flow.png",
            path="images/queue_flow.png",
            alt="Queue flow diagram",
            caption="Figure 1: Message bus routing pattern",
        ),
    ]

    final_markdown = markdown_builder.build(plan=plan, sections=sections, images=images)

    # Assert title and subtitle are present
    assert "# Test Production Article" in final_markdown
    assert "*A test subtitle explaining the system architecture in simple words.*" in final_markdown

    # Assert figure numbers were renumbered sequentially (Figure 1 first, Figure 2 second)
    assert "*Figure 1: Comprehensive system architecture blueprint*" in final_markdown
    assert "*Figure 2: Message bus routing pattern*" in final_markdown


# ---------------------------------------------------------
# Test 3: Cost Tracker Metrics Summary
# ---------------------------------------------------------
def test_cost_tracker_summary() -> None:
    """Verify workflow cost aggregation and summary metrics calculation."""
    dummy_metrics = [
        {
            "node_name": "router",
            "model": "gemini-2.5-flash",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "latency_ms": 1200.0,
            "estimated_cost": 0.0010,
            "status": "completed",
            "images_generated": 0,
            "is_fallback": False,
        },
        {
            "node_name": "image_generator",
            "model": "gemini-3-pro-image",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency_ms": 15000.0,
            "estimated_cost": 0.1340,
            "status": "completed",
            "images_generated": 1,
            "is_fallback": False,
        },
    ]

    summary = cost_tracker.calculate_summary(metrics=dummy_metrics, duration_seconds=16.2)

    assert summary["workflow_status"] == "completed"
    assert summary["total_cost"] == 0.135
    assert summary["total_tokens"] == 150
    assert summary["images_generated"] == 1
    assert summary["unique_models_used"] == 2
    assert summary["most_expensive_node"] == "image_generator"
