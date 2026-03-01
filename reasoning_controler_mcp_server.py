from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from shared_config import DEFAULT_BLACKBOARD

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# IMPORTANT: never print in an MCP stdio server (stdout must stay JSON-RPC only)
logger = logging.getLogger("reasoning_controller")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)  # goes to stderr

mcp = FastMCP("reasoning_controller")

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

mcp = FastMCP("reasoning_controller")


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_json_load(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(text[a : b + 1])
        raise


def _get_openai_client(api_key: Optional[str] = None) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install `openai`.")
    key = "OPENAI_API_KEY"
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)


def _normalize_weights_exact(w: Dict[str, Any]) -> Dict[str, float]:
    def f(k: str, d: float) -> float:
        try:
            return float(w.get(k, d))
        except Exception:
            return d

    wh = f("hpo", 0.50)
    wl = f("locus", 0.30)
    wb = f("biochem", 0.10)
    wi = f("literature", 0.10)

    vals = [ _clip(wh), _clip(wl), _clip(wb), _clip(wi) ]
    s = sum(vals)

    if s <= 0:
        vals = [0.50, 0.30, 0.10, 0.10]
        s = 1.0

    vals = [v / s for v in vals]

    vals[3] = 1.0 - (vals[0] + vals[1] + vals[2])

    if vals[3] < 0.0:
        deficit = -vals[3]
        vals[3] = 0.0
        j = max(range(3), key=lambda i: vals[i])
        vals[j] = max(0.0, vals[j] - deficit)
        # renormalize again and correct
        s2 = vals[0] + vals[1] + vals[2] + vals[3]
        if s2 <= 0:
            vals = [0.50, 0.30, 0.10, 0.10]
        else:
            vals = [v / s2 for v in vals]
            vals[3] = 1.0 - (vals[0] + vals[1] + vals[2])

    return {"hpo": vals[0], "locus": vals[1], "biochem": vals[2], "literature": vals[3]}


def _compact(lst: Optional[List[Dict[str, Any]]], n: int = 10) -> List[Dict[str, Any]]:
    if not lst:
        return []
    out = []
    for g in (lst or [])[:n]:
        if not isinstance(g, dict):
            continue
        try:
            score = float(g.get("final_score", g.get("score", 0)) or 0)
        except Exception:
            score = 0.0
        out.append(
            {
                "symbol": str(g.get("symbol", "") or g.get("gene", "") or "").strip(),
                "score": round(score, 4),
                "match_type": str(g.get("match_type", "") or "").strip(),
            }
        )
    return out


def _score_stats(lst: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    scores: List[float] = []
    for g in (lst or []):
        if not isinstance(g, dict):
            continue
        try:
            scores.append(float(g.get("final_score", g.get("score", 0)) or 0))
        except Exception:
            continue
    if not scores:
        return {"count": 0, "max": 0.0, "min": 0.0, "mean": 0.0, "spread": 0.0}
    mx, mn = max(scores), min(scores)
    mean = sum(scores) / len(scores)
    return {
        "count": len(scores),
        "max": round(mx, 4),
        "min": round(mn, 4),
        "mean": round(mean, 4),
        "spread": round(mx - mn, 4),
    }


def _infer_next_actions(
    completed: List[str],
    valid: List[str],
    syndromes: List[str],
    biochem: List[str],
) -> List[Dict[str, Any]]:

    completed_set = set(completed or [])
    has_biochem = any(str(x).strip() for x in (biochem or []))
    focus = (syndromes[0] if syndromes else "").strip()

    # Common step aliases (so you don't get stuck if your completed_steps uses a different name)
    registrar_done = any(s in completed_set for s in ("registrar", "phenotype_annotator", "annotate_case"))
    locus_done = "locus_boost" in completed_set
    geneticist_done = "geneticist" in completed_set
    biochem_done = "biochemist" in completed_set
    validator_done = "validator" in completed_set
    fhir_done = any(s in completed_set for s in ("fhir_export", "fhir_exporter", "export"))

    if registrar_done and (not locus_done) and bool(syndromes) and ("locus_boost" in valid):
        return [
            {
                "tool": "locus_boost",
                "reason": "Syndrome signal present; refine candidates with locus/disease associations.",
                "priority": "high",
                "guidance": {
                    "focus": focus,
                    "restrict": False,
                    "confidence": "medium",
                    "boost_genes": [],
                    "notes": "",
                },
            }
        ]

    if (registrar_done or locus_done) and (not geneticist_done) and ("geneticist" in valid):
        restrict = bool(locus_done)
        return [
            {
                "tool": "geneticist",
                "reason": "Run phenotype-to-gene ranking from HPO signal.",
                "priority": "high",
                "guidance": {
                    "focus": focus,
                    "restrict": restrict,
                    "confidence": "medium",
                    "boost_genes": [],
                    "notes": "",
                },
            }
        ]

    if geneticist_done and (not biochem_done) and has_biochem and ("biochemist" in valid):
        return [
            {
                "tool": "biochemist",
                "reason": "Biochemical abnormalities present; refine/boost metabolic candidates.",
                "priority": "medium",
                "guidance": {
                    "focus": "",
                    "restrict": True,
                    "confidence": "medium",
                    "boost_genes": [],
                    "notes": "",
                },
            }
        ]

    if geneticist_done and (not validator_done) and ("validator" in valid):
        return [
            {
                "tool": "validator",
                "reason": "Literature validation to confirm top candidates and differentiate close ranks.",
                "priority": "high",
                "guidance": {
                    "focus": focus,
                    "restrict": False,
                    "confidence": "medium",
                    "boost_genes": [],
                    "notes": "",
                },
            }
        ]

    if validator_done and (not fhir_done) and ("fhir_export" in valid):
        return [
            {
                "tool": "fhir_export",
                "reason": "All evidence streams complete; export final report artifacts.",
                "priority": "high",
                "guidance": {},
            }
        ]

    return []


@mcp.tool()
def update_blackboard(
    *,
    clinical_note: str,
    last_step: str,
    blackboard: Optional[Dict[str, Any]] = None,
    syndromes: Optional[List[str]] = None,
    hpo_ids: Optional[List[str]] = None,
    biochemical_findings: Optional[List[str]] = None,
    locus_top: Optional[List[Dict[str, Any]]] = None,
    hpo_top: Optional[List[Dict[str, Any]]] = None,
    biochem_top: Optional[List[Dict[str, Any]]] = None,
    literature_top: Optional[List[Dict[str, Any]]] = None,
    valid_next_tools: Optional[List[str]] = None,
    current_completed_steps: Optional[List[str]] = None,
    model: str = "gpt-5-mini",
    api_key: Optional[str] = "OPENAI_API_KEY"
) -> Dict[str, Any]:

    bb = blackboard or {}
    syndromes = syndromes or []
    hpo_ids = hpo_ids or []
    biochemical_findings = biochemical_findings or []
    valid_next_tools = valid_next_tools or ["locus_boost", "geneticist", "biochemist", "validator", "fhir_export"]
    current_completed_steps = current_completed_steps or []

    client = _get_openai_client(api_key=api_key)

    system = (
        "You are a Clinical Geneticist orchestrator inside a multi-agent gene prioritization system.\n"
        "Your role is to analyze structured signals, maintain the blackboard state, update diagnostic hypotheses,\n"
        "set evidence weights, and decide EXACTLY ONE next tool to execute.\n\n"
        "You DO NOT invent genes, syndromes, scores, or facts.\n"
        "You reason ONLY using provided evidence.\n\n"
        f"Valid next tool names: {', '.join(valid_next_tools)}.\n\n"
        "=== WORKFLOW RULES (STRICT) ===\n"
        "1) NEVER suggest a tool in current_completed_steps.\n"
        "2) Suggest EXACTLY ONE tool in next_actions.\n"
        "3) After locus_boost  → MUST suggest geneticist.\n"
        "4) After geneticist   → biochemist if has_biochem else validator.\n"
        "5) After biochemist   → validator.\n"
        "6) After validator    → fhir_export.\n"
        "7) guidance.restrict must be grounded in actual locus score analysis.\n"
        "8) boost_genes must be grounded in clinical_note text (do not guess).\n\n"
        "Return STRICT JSON only. No text outside JSON."
    )

    user = {
        "last_step": last_step,
        "current_completed_steps": current_completed_steps,
        "valid_next_tools": valid_next_tools,
        "signals": {
            "syndromes": syndromes[:6],
            "n_hpo": len(hpo_ids),
            "has_biochem": any(str(x).strip() for x in biochemical_findings),
            "biochemical_findings": biochemical_findings[:8],
        },
        "blackboard_prev": {
            "hypotheses": bb.get("hypotheses", []),
            "weights": bb.get("weights", {}),
            "signals": bb.get("signals", {}),
            "node_config": bb.get("node_config", {}),
        },
        "agent_output_analysis": {
            "locus_stats": _score_stats(locus_top),
            "hpo_stats": _score_stats(hpo_top),
            "biochem_stats": _score_stats(biochem_top),
            "literature_stats": _score_stats(literature_top),
            "locus_top": _compact(locus_top),
            "hpo_top": _compact(hpo_top),
            "biochem_top": _compact(biochem_top),
            "literature_top": _compact(literature_top),
        },
        "clinical_note_excerpt": (clinical_note or "")[:2500],
        "default_node_config": DEFAULT_BLACKBOARD.get("node_config", {}),
        "required_output_schema": {
            "blackboard": {
                "case_summary": "string (≤30 words)",
                "signals": {"has_syndrome": "bool", "has_biochem": "bool", "n_hpo": "int"},
                "hypotheses": [{"dx": "string", "confidence": "float 0-1", "why": ["strings"]}],
                "weights": {
                    "hpo": "float",
                    "locus": "float",
                    "biochem": "float",
                    "literature": "float",
                    "__constraint": "must sum EXACTLY to 1.0; streams not yet run = 0",
                },
                "node_config": "object (merge with defaults)",
                "merge_policy": "object (merge with defaults)",
            },
            "next_actions": [
                {
                    "tool": "exactly one of valid_next_tools, never a completed step",
                    "reason": "string — evidence-grounded",
                    "priority": "high|medium|low",
                    "guidance": {
                        "focus": "string",
                        "boost_genes": ["gene symbols (only if explicitly supported)"],
                        "restrict": "bool",
                        "confidence": "low|medium|high",
                        "notes": "string",
                    },
                }
            ],
        },
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
    )

    out = _safe_json_load(resp.choices[0].message.content or "{}")
    if not isinstance(out, dict):
        out = {}

    bb_out = out.get("blackboard") if isinstance(out.get("blackboard"), dict) else {}

    bb_out["weights"] = _normalize_weights_exact(
        bb_out.get("weights") if isinstance(bb_out.get("weights"), dict) else {}
    )

    sig = bb_out.get("signals") if isinstance(bb_out.get("signals"), dict) else {}
    sig.setdefault("has_syndrome", bool(syndromes))
    sig.setdefault("has_biochem", any(str(x).strip() for x in biochemical_findings))
    sig.setdefault("n_hpo", len(hpo_ids))
    bb_out["signals"] = sig

    nc_out = bb_out.get("node_config") if isinstance(bb_out.get("node_config"), dict) else {}
    defaults_nc = DEFAULT_BLACKBOARD.get("node_config", {}) or {}
    for node_name, node_defaults in defaults_nc.items():
        llm_node = nc_out.get(node_name) if isinstance(nc_out.get(node_name), dict) else {}
        merged = {**(node_defaults or {}), **{k: v for k, v in (llm_node or {}).items() if v is not None}}
        nc_out[node_name] = merged
    bb_out["node_config"] = nc_out

    mp_defaults = DEFAULT_BLACKBOARD.get("merge_policy", {}) or {}
    mp_llm = bb_out.get("merge_policy") if isinstance(bb_out.get("merge_policy"), dict) else {}
    bb_out["merge_policy"] = {**mp_defaults, **{k: v for k, v in (mp_llm or {}).items() if v is not None}}

    na_raw = out.get("next_actions") if isinstance(out.get("next_actions"), list) else []
    completed_set = set(current_completed_steps)
    na_out: List[Dict[str, Any]] = []

    for action in na_raw:
        if not isinstance(action, dict):
            continue
        tool_name = str(action.get("tool", "")).strip()
        if not tool_name or tool_name in completed_set or tool_name not in valid_next_tools:
            continue
        if not isinstance(action.get("guidance"), dict):
            action["guidance"] = {}
        na_out.append(action)
        break

    if not na_out:
        na_out = _infer_next_actions(current_completed_steps, valid_next_tools, syndromes, biochemical_findings)

    if len(na_out) > 1:
        na_out = na_out[:1]

    bb_out["next_actions"] = na_out
    out["blackboard"] = bb_out
    out["next_actions"] = na_out
    return out


if __name__ == "__main__":
    mcp.run()