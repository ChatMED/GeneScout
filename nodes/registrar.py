from __future__ import annotations
import os
print(os.getenv("OPENAI_API_KEY"))
from typing import Any, Dict

from utils.shared_config import DEFAULT_BLACKBOARD

from utils.mcp_parsing import parse_mcp_result
from utils.mcp_registry import ToolRegistry
from utils.state_types import DiagnosticState


async def registrar_node(state: DiagnosticState, registry: ToolRegistry) -> Dict[str, Any]:
    tool = await registry.get_tool("annotate_case")
    result = await tool.ainvoke({"text": state["clinical_note"]})
    data = parse_mcp_result(result)

    norm = data.get("normalization", {}) or {}
    syndromes = data.get("syndromes") or norm.get("syndromes") or []
    keywords = data.get("keywords") or norm.get("keywords") or []
    additional_context = data.get("additional_context") or norm.get("additional_context") or []
    query_terms = data.get("query_terms") or norm.get("query_terms") or {}
    noise_terms = data.get("noise_terms") or norm.get("noise_terms") or []

    return {
        "annotate_payload": data,
        "hpo_ids": data.get("hpo_ids", []),
        "hpo_names": data.get("hpo_names", []),
        "biochemical_findings": norm.get("biochemical", []),
        "negatives": norm.get("negatives", []),
        "syndromes": syndromes,
        "keywords": keywords,
        "additional_context": additional_context,
        "query_terms": query_terms,
        "noise_terms": noise_terms,
        "steps_completed": ["registrar"],
        "blackboard": state.get("blackboard") or DEFAULT_BLACKBOARD.copy(),
        "merged_genes": state.get("merged_genes", []) or [],
    }