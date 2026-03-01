from __future__ import annotations

from typing import Any, Dict

from utils.mcp_parsing import parse_mcp_result
from utils.mcp_registry import ToolRegistry
from utils.node_helpers import get_node_guidance
from utils.state_types import DiagnosticState


async def locus_boost_node(state: DiagnosticState, registry: ToolRegistry, llm_model: str) -> Dict[str, Any]:
    tool = await registry.get_tool("locus_boost_rerank")

    guidance = get_node_guidance(state, "locus_boost")
    focus = guidance.get("focus") or ""
    if focus:
        print(f"  → Controller guidance: '{focus}'")

    annotate_payload = state.get("annotate_payload") or {}
    ranked = state.get("golden_genes") or []

    payload = {
        "annotate_payload": annotate_payload,
        "ranked_genes": ranked,
        "clinical_note": state["clinical_note"],
        "syndromes": state.get("syndromes", []) or [],
        "blackboard": state.get("blackboard") or {},
        "llm_model": llm_model,
    }

    result = await tool.ainvoke(payload)
    data = parse_mcp_result(result)

    reranked = data.get("reranked_genes") or data.get("reranked_results") or ranked
    evidence = data.get("evidence", {}) or {}

    return {
        "candidate_genes": reranked,
        "locus_reranked_genes": reranked,
        "locus_raw_genes": reranked,
        "locus_evidence": evidence,
        "steps_completed": ["locus_boost"],
    }