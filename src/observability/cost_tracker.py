"""
Lightweight Cost Tracking & Observability for InkFlow-AI.

Responsibilities:
- Create standardized node execution metric records.
- Extract token counts and costs from LiteLLM/LangChain responses.
- Calculate deterministic custom costs for image generation models.
- Aggregate workflow execution summary metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from litellm import completion_cost
from litellm.types.utils import ModelResponse, Usage

logger = logging.getLogger(__name__)

# Official / Custom Image Model Pricing Table per Resolution Tier
IMAGE_PRICING: dict[str, dict[str, float]] = {
    "gemini-3-pro-image": {
        "1k": 0.134,
        "2k": 0.134,
        "4k": 0.240,
    },
    "gemini-3.1-flash-image": {
        "1k": 0.045,
        "2k": 0.045,
        "4k": 0.080,
    },
    "imagen-3": {
        "1k": 0.030,
        "2k": 0.030,
        "4k": 0.060,
    },
    "dall-e-3": {
        "1k": 0.040,
        "2k": 0.080,
        "4k": 0.120,
    },
    "dall-e-2": {
        "1k": 0.020,
        "2k": 0.020,
        "4k": 0.020,
    },
}


def get_image_resolution_tier(size: str) -> str:
    """
    Map resolution size string (e.g. '1024x1024', '2560x1440') to pricing tier ('1k', '2k', '4k').
    """
    if not size or not isinstance(size, str):
        return "2k"

    s = size.strip().lower()

    # Exact / common resolution mappings
    if s in ("1024x1024", "1024*1024", "1k", "512x512", "768x768"):
        return "1k"
    if s in ("2048x2048", "2560x1440", "1920x1080", "1792x1024", "2k"):
        return "2k"
    if s in ("3840x2160", "4096x4096", "4k", "4096x2160"):
        return "4k"

    # Dimension parsing fallback (e.g. max side dimension check)
    try:
        if "x" in s:
            parts = s.split("x")
            max_dim = max(int(parts[0]), int(parts[1]))
            if max_dim <= 1280:
                return "1k"
            elif max_dim <= 2880:
                return "2k"
            else:
                return "4k"
    except Exception:
        pass

    return "2k"


def calculate_image_cost(
    model_name: str,
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1,
) -> float:
    """
    Calculate deterministic USD cost for image generation models using custom pricing table.
    Bypasses LiteLLM completion_cost for image models.
    """
    if not model_name or "placeholder" in model_name.lower():
        return 0.0

    raw_model = str(model_name).lower().strip()
    if "/" in raw_model:
        raw_model = raw_model.split("/")[-1]

    # Map model name string to IMAGE_PRICING key
    matched_key = None
    for key in IMAGE_PRICING:
        if key in raw_model:
            matched_key = key
            break

    if not matched_key:
        logger.warning("Unknown image pricing for model: %s", model_name)
        return 0.0

    tier = get_image_resolution_tier(size)
    tier_prices = IMAGE_PRICING[matched_key]
    price_per_image = tier_prices.get(tier, tier_prices.get("2k", 0.0))

    return round(price_per_image * n, 6)


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
    def infer_provider(model_name: str, default_provider: str = "openai") -> str:
        """
        Infer standard provider identifier from model name string so provider and model
        always match accurately when fallbacks occur.
        """
        if not model_name:
            return default_provider
        m = str(model_name).lower()
        if "/" in m:
            prefix = m.split("/")[0]
            if prefix in ("vertex_ai", "gemini", "google"):
                return "vertex_ai"
            if prefix in ("openai", "azure"):
                return "openai"
            if prefix in ("anthropic", "claude"):
                return "anthropic"
            if prefix in ("tavily", "web_search"):
                return "tavily"
            if prefix in ("local", "placeholder"):
                return "local"
            return prefix
        if "gemini" in m or "imagen" in m:
            return "vertex_ai"
        if "gpt" in m or "dall-e" in m:
            return "openai"
        if "claude" in m:
            return "anthropic"
        return default_provider

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
        Extract token usage and calculate exact cost using LiteLLM completion_cost.
        Detects actual model used and fallback status from response metadata.
        """
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        actual_model = model

        res_obj = response
        if isinstance(response, dict) and "raw" in response:
            res_obj = response.get("raw")

        # Extract actual responding model from response metadata
        if hasattr(res_obj, "response_metadata") and res_obj.response_metadata:
            rm = res_obj.response_metadata
            resp_model = rm.get("model_name") or rm.get("model")
            if resp_model and isinstance(resp_model, str):
                actual_model = resp_model
                if actual_model != model:
                    is_fallback = True

        actual_provider = CostTracker.infer_provider(actual_model, default_provider=provider)

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
            if hasattr(token_usage, "prompt_tokens"):
                prompt_tokens = prompt_tokens or getattr(token_usage, "prompt_tokens", None)
                completion_tokens = completion_tokens or getattr(token_usage, "completion_tokens", None)
                total_tokens = total_tokens or getattr(token_usage, "total_tokens", None)
            elif isinstance(token_usage, dict):
                prompt_tokens = prompt_tokens or token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                completion_tokens = completion_tokens or token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                total_tokens = total_tokens or token_usage.get("total_tokens")

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

        # Calculate exact USD cost using litellm completion_cost
        cost = 0.0
        if total_tokens or prompt_tokens or completion_tokens:
            try:
                res_wrapper = ModelResponse()
                res_wrapper.model = actual_model
                res_wrapper.usage = Usage(
                    prompt_tokens=prompt_tokens or 0,
                    completion_tokens=completion_tokens or 0,
                    total_tokens=total_tokens or ((prompt_tokens or 0) + (completion_tokens or 0)),
                )
                calc_cost = completion_cost(completion_response=res_wrapper)
                if calc_cost is not None and calc_cost > 0:
                    cost = float(calc_cost)
            except Exception as err:
                logger.debug("completion_cost calculation failed for %s: %s", actual_model, err)

        if cost == 0.0 and total_tokens:
            cost = round((total_tokens / 1000.0) * 0.00015, 6)

        return CostTracker.create_metric(
            node_name=node_name,
            provider=actual_provider,
            model=actual_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost=cost,
            status="completed",
            is_fallback=is_fallback,
        )

    @staticmethod
    def extract_image_metrics(
        node_name: str,
        provider: str,
        model: str,
        latency_ms: float,
        images_generated: int = 1,
        resolution: str = "2560x1440",
        is_fallback: bool = False,
        status: str = "completed",
    ) -> dict[str, Any]:
        """
        Extract metrics and exact USD cost for image generation calls.
        """
        actual_provider = CostTracker.infer_provider(model, default_provider=provider)
        cost = calculate_image_cost(
            model_name=model,
            size=resolution,
            n=images_generated,
        ) if status == "completed" else 0.0

        return CostTracker.create_metric(
            node_name=node_name,
            provider=actual_provider,
            model=model,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=latency_ms,
            estimated_cost=cost,
            status=status,
            images_generated=images_generated,
            resolution=resolution,
            is_fallback=is_fallback,
        )

    @staticmethod
    def calculate_summary(
        metrics: list[dict[str, Any]],
        duration_seconds: float = 0.0,
        sections_count: int = 0,
        sources_count: int = 0,
        guardrail_violations: int = 0,
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
                "guardrail_violations": guardrail_violations,
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
            "guardrail_violations": guardrail_violations,
            "execution_duration": f"{duration_seconds:.1f}s",
            "successful_nodes": successful_nodes,
            "failed_nodes": failed_nodes,
        }


cost_tracker = CostTracker()

