from __future__ import annotations

import json
from typing import Any, Dict, List

from utils.shared_config import DEFAULT_BLACKBOARD

from utils.mcp_parsing import parse_mcp_result
from utils.mcp_registry import ToolRegistry
from utils.state_types import DiagnosticState


def _log_controller(last_step: str, next_actions: List[Dict[str, Any]], weights: Dict[str, Any], node_cfg: Dict[str, Any]) -> None:
    tools = [a.get("tool") for a in next_actions]
    print(f"  → Controller next_actions: {tools}")
    print(
        f"  → Weights: hpo={float(weights.get('hpo', 0)):.2f}  "
        f"locus={float(weights.get('locus', 0)):.2f}  "
        f"biochem={float(weights.get('biochem', 0)):.2f}  "
        f"literature={float(weights.get('literature', 0)):.2f}"
    )
    if node_cfg:
        for a in next_actions:
            tool = a.get("tool", "")
            cfg = node_cfg.get(tool, {})
            if cfg:
                print(f"  → [{tool}] node_config: {json.dumps(cfg, ensure_ascii=False)}")


async def controller_node(state: DiagnosticState, registry: ToolRegistry, last_step: str, controller_model: str) -> Dict[str, Any]:
    tool = await registry.get_tool("update_blackboard")

    locus_top = (state.get("locus_reranked_genes") or [])[:20]
    hpo_top = (state.get("golden_genes") or [])[:20]
    bio_top = (state.get("biochem_reranked_genes") or [])[:20]
    lit_top = (state.get("literature_reranked_genes") or [])[:20]
    completed = state.get("steps_completed", []) or []

    payload = {
        "clinical_note": state["clinical_note"],
        "last_step": last_step,
        "blackboard": state.get("blackboard", {}) or {},
        "syndromes": state.get("syndromes", []) or [],
        "hpo_ids": state.get("hpo_ids", []) or [],
        "biochemical_findings": state.get("biochemical_findings", []) or [],
        "locus_top": locus_top,
        "hpo_top": hpo_top,
        "biochem_top": bio_top,
        "literature_top": lit_top,
        "model": controller_model,
        "valid_next_tools": ["locus_boost", "geneticist", "biochemist", "validator", "fhir_export"],
        "current_completed_steps": completed,
    }

    result = await tool.ainvoke(payload)
    data = parse_mcp_result(result)

    bb = data.get("blackboard", {}) or {}
    weights = bb.get("weights") or DEFAULT_BLACKBOARD["weights"]
    node_cfg = bb.get("node_config") or {}
    na_out = data.get("next_actions")
    if not isinstance(na_out, list):
        na_out = []

    bb["next_actions"] = na_out
    _log_controller(last_step, na_out, weights, node_cfg)

    if last_step == "locus_boost":
        merged = state.get("locus_reranked_genes") or []
    elif last_step == "geneticist":
        merged = state.get("golden_genes") or []
    elif last_step == "biochemist":
        merged = state.get("biochem_reranked_genes") or []
    elif last_step == "validator":
        merged = state.get("literature_reranked_genes") or []
    else:
        merged = state.get("merged_genes") or []

    return {
        "blackboard": bb,
        "merged_genes": merged,
        "steps_completed": [f"controller_after_{last_step}"],
    }