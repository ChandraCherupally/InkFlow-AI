"""
Lightweight Cost Tracking & Observability for InkFlow-AI.

Responsibilities:
- Create standardized node execution metric records.
- Extract token counts and costs from LiteLLM/LangChain responses.
- Aggregate workflow execution summary metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Production-ready Lightweight Cost Tracking & Observability Collector.
    """

    @staticmethod
    def create_metric(
        node_name: str,
        provider: str,
        model: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float = 0.0,
        estimated_cost: float | None = None,
        status: str = "completed",
        images_generated: int = 0,
        resolution: str | None = None,
        is_fallback: bool = False,
    ) -> dict[str, Any]:
        """
        Create a single standardized node execution metric record.
        """
        return {
            "node_name": node_name,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
            "estimated_cost": round(estimated_cost, 6) if estimated_cost is not None else 0.0,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "images_generated": images_generated,
            "resolution": resolution,
            "is_fallback": is_fallback,
        }

    @staticmethod
    def deduplicate_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Deduplicate metric records by unique execution signature to eliminate
        subgraph state reducer duplication artifacts.
        """
        seen: set[tuple] = set()
        deduped: list[dict[str, Any]] = []

        for m in metrics:
            if not isinstance(m, dict):
                continue
            sig = (
                m.get("node_name"),
                m.get("provider"),
                m.get("model"),
                m.get("prompt_tokens"),
                m.get("completion_tokens"),
                m.get("total_tokens"),
                m.get("latency_ms"),
                m.get("timestamp"),
                m.get("images_generated"),
                m.get("resolution"),
            )
            if sig not in seen:
                seen.add(sig)
                deduped.append(m)

        return deduped

    @staticmethod
    def extract_llm_metrics(
        response: Any,
        node_name: str,
        provider: str,
        model: str,
        latency_ms: float,
        is_fallback: bool = False,
    ) -> dict[str, Any]:
        """
        Extract token usage and cost metadata from an LLM response object.
        """
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        cost = 0.0005  # Base estimate

        res_obj = response
        if isinstance(response, dict) and "raw" in response:
            res_obj = response.get("raw")

        # 1. Try usage_metadata from LangChain BaseMessage / AIMessage
        if hasattr(res_obj, "usage_metadata") and res_obj.usage_metadata:
            um = res_obj.usage_metadata
            prompt_tokens = um.get("input_tokens") or um.get("prompt_tokens")
            completion_tokens = um.get("output_tokens") or um.get("completion_tokens")
            total_tokens = um.get("total_tokens")

        # 2. Try response_metadata from LiteLLM / LangChain
        if not total_tokens and hasattr(res_obj, "response_metadata") and res_obj.response_metadata:
            rm = res_obj.response_metadata
            token_usage = rm.get("token_usage") or rm.get("usage") or {}
            prompt_tokens = prompt_tokens or token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
            completion_tokens = completion_tokens or token_usage.get("completion_tokens") or token_usage.get("output_tokens")
            total_tokens = total_tokens or token_usage.get("total_tokens")
            if "_response_cost" in rm and rm["_response_cost"] is not None:
                cost = float(rm["_response_cost"])

        # 3. Try direct usage attribute
        if not total_tokens and hasattr(res_obj, "usage") and res_obj.usage:
            u = res_obj.usage
            if isinstance(u, dict):
                prompt_tokens = prompt_tokens or u.get("prompt_tokens") or u.get("input_tokens")
                completion_tokens = completion_tokens or u.get("completion_tokens") or u.get("output_tokens")
                total_tokens = total_tokens or u.get("total_tokens")
            elif hasattr(u, "prompt_tokens"):
                prompt_tokens = prompt_tokens or getattr(u, "prompt_tokens", None)
                completion_tokens = completion_tokens or getattr(u, "completion_tokens", None)
                total_tokens = total_tokens or getattr(u, "total_tokens", None)

        if not total_tokens and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        # Calculate lightweight estimated cost if total_tokens is known
        if total_tokens and cost == 0.0005:
            # Approx $0.00015 / 1K tokens for Flash tier
            cost = round((total_tokens / 1000.0) * 0.00015, 6)

        return CostTracker.create_metric(
            node_name=node_name,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost=cost,
            status="completed",
            is_fallback=is_fallback,
        )

    @staticmethod
    def calculate_summary(
        metrics: list[dict[str, Any]],
        duration_seconds: float = 0.0,
        sections_count: int = 0,
        sources_count: int = 0,
    ) -> dict[str, Any]:
        """
        Automatically aggregate individual node metrics into an Execution Summary.
        """
        metrics = CostTracker.deduplicate_metrics(metrics or [])
        if not metrics:
            return {
                "workflow_status": "completed",
                "total_cost": 0.0,
                "total_tokens": 0,
                "average_latency": 0.0,
                "most_expensive_node": "N/A",
                "most_expensive_model": "N/A",
                "slowest_node": "N/A",
                "sections_generated": sections_count,
                "images_generated": 0,
                "sources_retrieved": sources_count,
                "unique_models_used": 0,
                "fallback_count": 0,
                "guardrail_violations": 0,
                "execution_duration": f"{duration_seconds:.1f}s",
                "successful_nodes": 0,
                "failed_nodes": 0,
            }

        total_cost = sum(m.get("estimated_cost") or 0.0 for m in metrics)
        total_tokens = sum(m.get("total_tokens") or 0 for m in metrics if m.get("total_tokens"))
        latencies = [m.get("latency_ms") or 0.0 for m in metrics]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Most expensive node calculation
        node_costs: dict[str, float] = {}
        for m in metrics:
            n = m.get("node_name", "unknown")
            node_costs[n] = node_costs.get(n, 0.0) + (m.get("estimated_cost") or 0.0)
        most_expensive_node = max(node_costs, key=lambda k: node_costs[k]) if node_costs else "N/A"

        # Most expensive model calculation
        model_costs: dict[str, float] = {}
        for m in metrics:
            mod = m.get("model", "unknown")
            model_costs[mod] = model_costs.get(mod, 0.0) + (m.get("estimated_cost") or 0.0)
        most_expensive_model = max(model_costs, key=lambda k: model_costs[k]) if model_costs else "N/A"

        # Slowest node calculation
        slowest_metric = max(metrics, key=lambda x: x.get("latency_ms", 0.0)) if metrics else {}
        slowest_node = slowest_metric.get("node_name", "N/A")

        # Key counts
        images_generated = sum(m.get("images_generated", 0) for m in metrics)
        unique_models = len({m.get("model") for m in metrics if m.get("model")})
        fallback_count = sum(1 for m in metrics if m.get("is_fallback"))
        successful_nodes = sum(1 for m in metrics if m.get("status") == "completed")
        failed_nodes = sum(1 for m in metrics if m.get("status") == "failed")

        return {
            "workflow_status": "failed" if failed_nodes > 0 else "completed",
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "average_latency": round(avg_latency, 1),
            "most_expensive_node": most_expensive_node,
            "most_expensive_model": most_expensive_model,
            "slowest_node": slowest_node,
            "sections_generated": sections_count,
            "images_generated": images_generated,
            "sources_retrieved": sources_count,
            "unique_models_used": unique_models,
            "fallback_count": fallback_count,
            "guardrail_violations": 0,
            "execution_duration": f"{duration_seconds:.1f}s",
            "successful_nodes": successful_nodes,
            "failed_nodes": failed_nodes,
        }


cost_tracker = CostTracker()
