from __future__ import annotations

from typing import Any, Dict, List

from utils.mcp_parsing import parse_mcp_result
from utils.mcp_registry import ToolRegistry
from utils.node_helpers import get_node_guidance, gene_symbol
from utils.state_types import DiagnosticState


async def geneticist_node(
    state: DiagnosticState,
    registry: ToolRegistry,
    reasoning_model: str,
) -> Dict[str, Any]:
    tool = await registry.get_tool("hpo_gene_ranker_with_reasoning")

    guidance = get_node_guidance(state, "geneticist")
    bb = state.get("blackboard") or {}
    nc = (bb.get("node_config") or {}).get("geneticist") or {}

    locus_ran = "locus_boost" in (state.get("steps_completed") or [])
    restrict = guidance.get("restrict", nc.get("restrict", locus_ran))
    confidence = guidance.get("confidence", "medium")
    focus = guidance.get("focus", "")
    boost_genes: List[str] = guidance.get("boost_genes") or []

    top_n = int(nc.get("top_n", 200 if not restrict else 150))
    if confidence == "low":
        top_n = max(top_n, 300)

    if focus:
        print(f"  → Controller guidance: focus='{focus}' restrict={restrict} confidence={confidence} top_n={top_n}")

    previous = state.get("locus_reranked_genes") or state.get("candidate_genes") or []
    if boost_genes and previous:
        previous = sorted(previous, key=lambda g: 0 if gene_symbol(g) in boost_genes else 1)

    payload = {
        "hpo_ids": state["hpo_ids"],
        "previous_candidates": previous,
        "restrict_to_previous": restrict,
        "top_n": top_n,
        "prune": False,
        "similarity_method": nc.get("similarity_method", "resnik"),
        "similarity_combine": nc.get("similarity_combine", "funSimMax"),
        "include_reasoning": nc.get("include_reasoning", False),
        "reasoning_top_n": int(nc.get("reasoning_top_n", 25)),
        "reasoning_model": reasoning_model,
    }

    result = await tool.ainvoke(payload)
    data = parse_mcp_result(result)
    ranked = data.get("candidate_genes", []) or []

    if restrict and len(ranked) == 0 and previous:
        print("  → Restricted returned 0 genes → falling back to full HPO enrichment")
        payload["restrict_to_previous"] = False
        payload["top_n"] = max(top_n, 250)
        result = await tool.ainvoke(payload)
        data = parse_mcp_result(result)
        ranked = data.get("candidate_genes", []) or []

    print(f"  → Final HPO ranked genes: {len(ranked)}")

    return {
        "golden_genes": ranked,
        "golden_reasoning": data.get("reasoning", {}) or {},
        "steps_completed": ["geneticist"],
    }