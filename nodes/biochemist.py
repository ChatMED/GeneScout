from __future__ import annotations

from typing import Any, Dict, List

from utils.mcp_parsing import parse_mcp_result
from utils.mcp_registry import ToolRegistry
from utils.node_helpers import get_node_guidance, gene_symbol
from utils.state_types import DiagnosticState


async def biochemist_node(state: DiagnosticState, registry: ToolRegistry) -> Dict[str, Any]:
    tool = await registry.get_tool("refine_biochemical_candidates")

    guidance = get_node_guidance(state, "biochemist")
    boost_genes: List[str] = guidance.get("boost_genes") or []
    focus = guidance.get("focus") or ""
    if focus:
        print(f"  → Controller guidance: '{focus}'")

    base_genes = (
        state.get("golden_genes") or
        state.get("locus_reranked_genes") or
        state.get("candidate_genes") or []
    )
    if not base_genes:
        print("  → No candidate genes to refine → skipping")
        return {
            "biochem_reranked_genes": [],
            "biochem_reasoning": {"note": "skipped — no input candidates"},
            "steps_completed": ["biochem_refiner_skipped"],
        }

    if boost_genes:
        print(f"  → Controller boosted genes: {boost_genes}")
        base_genes = sorted(base_genes, key=lambda g: 0 if gene_symbol(g) in boost_genes else 1)

    payload = {
        "biochemical_findings": state["biochemical_findings"],
        "candidate_genes": base_genes,
        "include_reasoning": True,
        "reasoning_top_n": 25,
        "blackboard": state.get("blackboard") or {},
    }

    result = await tool.ainvoke(payload)
    data = parse_mcp_result(result)

    return {
        "biochem_reranked_genes": data.get("reranked_genes", []) or [],
        "biochem_reasoning": data.get("reasoning", {}) or {},
        "steps_completed": ["biochem_refiner"],
    }