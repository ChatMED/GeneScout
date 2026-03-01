from __future__ import annotations

from typing import Any, Dict, List


def safe_topk_merged(final_state: Dict[str, Any], k: int = 20) -> List[Dict[str, Any]]:
    arr = final_state.get("merged_genes")
    if isinstance(arr, list) and arr:
        return arr[:k]
    return []


def gene_symbol(g: Dict[str, Any]) -> str:
    return (g.get("symbol") or g.get("gene") or g.get("name") or "").strip()


def gene_score(g: Dict[str, Any]) -> Any:
    v = g.get("final_score")
    return v if v is not None else ""


def get_node_guidance(state: Dict[str, Any], node_name: str) -> dict:
    bb = state.get("blackboard") or {}
    next_actions = bb.get("next_actions") or []
    for action in next_actions:
        if action.get("tool") == node_name:
            return action.get("guidance") or {}
    return {}