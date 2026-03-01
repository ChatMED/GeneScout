from __future__ import annotations

from typing import Any, Dict

from utils.mcp_parsing import parse_mcp_result
from utils.mcp_registry import ToolRegistry
from utils.node_helpers import get_node_guidance
from utils.state_types import DiagnosticState


async def validator_node(state: DiagnosticState, registry: ToolRegistry, llm_model: str) -> Dict[str, Any]:
    tool = await registry.get_tool("search_literature_nuanced")

    guidance = get_node_guidance(state, "validator")
    confidence = guidance.get("confidence", "medium")
    focus = guidance.get("focus") or ""
    if focus:
        print(f"  → Controller guidance: '{focus}'")

    base_genes = (
        state.get("biochem_reranked_genes") or
        state.get("golden_genes") or
        state.get("locus_reranked_genes") or []
    )

    bb = state.get("blackboard") or {}
    nc = (bb.get("node_config") or {}).get("validator") or {}
    top_n = int(nc.get("search_top_n", 300))
    if confidence == "low":
        top_n = max(top_n, 300)

    top_genes = base_genes[:top_n]
    gene_list = [g.get("symbol") for g in top_genes if g.get("symbol")]

    result = await tool.ainvoke({
        "genes": gene_list,
        "ranked_genes": base_genes,
        "clinical_note": state["clinical_note"],
        "use_llm_query": True,
        "llm_model": llm_model,
        "phenotypes": state.get("hpo_names", []) or [],
        "biochemicals": state.get("biochemical_findings", []) or [],
        "syndromes": state.get("syndromes", []) or [],
        "keywords": state.get("keywords", []) or [],
        "additional_context": (state.get("additional_context", []) + ([focus] if focus else [])),
        "query_terms": state.get("query_terms", {}) or {},
        "noise_terms": state.get("noise_terms", []) or [],
        "blackboard": bb,
    })

    data = parse_mcp_result(result)
    meta_genes = ((data.get("meta") or {}).get("genes") or {})

    return {
        "literature_reranked_genes": data.get("reranked_results", []) or [],
        "pubmed_evidence": meta_genes,
        "steps_completed": ["validator"],
    }