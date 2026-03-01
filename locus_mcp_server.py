from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import requests
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from shared_config import get_node_config

mcp = FastMCP("locus_boost")
OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"


def _get_session_with_retries(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=retries, backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    return session


_session = _get_session_with_retries()


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 35) -> Dict[str, Any]:
    r = _session.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ot_find_disease_ids_by_name(name: str, size: int = 5) -> List[Tuple[str, str, float]]:
    q = """
    query SearchDisease($q: String!, $size: Int!) {
      search(queryString: $q, entityNames: ["disease"], page: { index: 0, size: $size }) {
        hits { id entity name score }
      }
    }
    """
    try:
        data = _post_json(OT_GRAPHQL, {"query": q, "variables": {"q": name, "size": int(size)}})
        hits = data.get("data", {}).get("search", {}).get("hits", []) or []
        out: List[Tuple[str, str, float]] = []
        for h in hits:
            if h.get("entity") == "disease" and h.get("id") and h.get("name"):
                out.append((str(h["id"]), str(h["name"]), float(h.get("score") or 0.0)))
        out.sort(key=lambda x: x[2], reverse=True)
        return out
    except Exception:
        return []


def ot_disease_associated_genes(disease_id: str, size: int = 300) -> Dict[str, float]:
    q = """
    query DiseaseTargets($diseaseId: String!, $size: Int!) {
      disease(efoId: $diseaseId) {
        associatedTargets(page: { index: 0, size: $size }) {
          rows { target { approvedSymbol } score }
        }
      }
    }
    """
    try:
        data = _post_json(OT_GRAPHQL, {"query": q, "variables": {"diseaseId": disease_id, "size": int(size)}})
        rows = data.get("data", {}).get("disease", {}).get("associatedTargets", {}).get("rows", []) or []
        out: Dict[str, float] = {}
        for r in rows:
            sym = (r.get("target") or {}).get("approvedSymbol", "")
            sym = str(sym).strip()
            if not sym:
                continue
            sc = float(r.get("score") or 0.0)
            out[sym] = max(sc, out.get(sym, 0.0))
        return out
    except Exception:
        return {}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _get_gene_symbol(g: Dict[str, Any]) -> str:
    for k in ("symbol", "gene", "name"):
        val = str(g.get(k) or "").strip()
        if val:
            return val
    return ""


def _get_base_score(g: Dict[str, Any]) -> float:
    for k in ("final_score", "score", "adjusted_score", "base_score"):
        try:
            v = g.get(k)
            if v is None:
                continue
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _delta_from_signal(signal_0_1: float, max_delta: float) -> float:
    s = _clip(signal_0_1, 0.0, 1.0)
    return max_delta * (s ** 0.75)


def _call_llm_gene_fit_scores(
    *,
    clinical_note: str,
    syndromes: List[str],
    gene_symbols: List[str],
    model: str,
    api_key: Optional[str],
) -> Dict[str, float]:
    if not api_key or not clinical_note.strip() or not gene_symbols:
        return {}
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "Score how well each provided gene explains the case. "
        "Return ONLY JSON mapping the EXACT provided gene symbols to numbers in [0,1]. Consider even the rarest cases"
        "Include ALL provided genes. No extra keys. No text."
    )
    user_content = json.dumps({
        "clinical_summary":   clinical_note[:4500].strip(),
        "suspected_syndromes": syndromes[:6],
        "candidate_genes":    gene_symbols,
    }, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
        raw    = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        allowed = set(gene_symbols)
        out: Dict[str, float] = {}
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                sym = str(k).strip()
                if sym in allowed:
                    try:
                        out[sym] = _clip(float(v))
                    except (TypeError, ValueError):
                        continue
        return out
    except Exception:
        return {}


@mcp.tool()
def locus_boost_rerank(
    annotate_payload: Dict[str, Any],
    ranked_genes: List[Dict[str, Any]],
    clinical_note: str,
    *,
    syndromes: Optional[List[str]] = None,
    blackboard: Optional[Dict[str, Any]] = None,
    llm_model: str = "gpt-5-mini",
    llm_api_key: str = "OPENAI_API_KEY"
) -> Dict[str, Any]:
    cfg = get_node_config(blackboard or {}, "locus_boost")
    max_diseases        = int(cfg["max_diseases"])
    ot_gene_cap         = int(cfg["ot_gene_cap"])
    max_delta           = float(cfg["max_delta"])
    inject_new          = bool(cfg["inject_new"])
    inject_base_score   = float(cfg["inject_base_score"])
    min_inject_llm_score = float(cfg["min_inject_llm_score"])
    use_llm             = bool(cfg["use_llm"])
    llm_weight          = float(cfg["llm_weight"])
    ap            = annotate_payload or {}
    locus_signals = (ap.get("locus_signals") or {}) if isinstance(ap.get("locus_signals"), dict) else {}
    norm          = (ap.get("normalization") or {}) if isinstance(ap.get("normalization"), dict) else {}

    if syndromes is None:
        syndromes = locus_signals.get("suspected_syndromes") or norm.get("syndromes") or []
    syndromes = [str(s).strip() for s in (syndromes or []) if str(s).strip()]

    if not syndromes:
        return {
            "triggered":      False,
            "reason":         "No syndromes provided.",
            "reranked_genes": ranked_genes or [],
            "evidence":       {"locus_signals": locus_signals},
        }

    disease_matches: List[Dict[str, Any]] = []
    ot_gene_scores:  Dict[str, float]     = {}

    for syndrome in syndromes[:max_diseases]:
        hits = ot_find_disease_ids_by_name(syndrome, size=4)
        if not hits:
            disease_matches.append({"query": syndrome, "disease_id": None, "disease_name": None,
                                     "search_score": 0.0, "top_hits": []})
            continue

        best = None
        for hid, hname, hscore in hits:
            if str(hid).startswith("HP_"):
                continue
            best = (str(hid), str(hname), float(hscore))
            break

        if not best:
            disease_matches.append({"query": syndrome, "disease_id": None, "disease_name": None,
                                     "search_score": 0.0, "top_hits": hits[:3]})
            continue

        did, dname, dscore = best
        disease_matches.append({"query": syndrome, "disease_id": did, "disease_name": dname,
                                  "search_score": dscore, "top_hits": hits[:3]})

        gene_map = ot_disease_associated_genes(did, size=ot_gene_cap)
        for sym, sc in gene_map.items():
            ot_gene_scores[sym] = max(sc, ot_gene_scores.get(sym, 0.0))

    if not ot_gene_scores:
        return {
            "triggered":      False,
            "reason":         "No disease-associated genes found in Open Targets.",
            "reranked_genes": ranked_genes or [],
            "evidence":       {"disease_matches": disease_matches, "syndromes_used": syndromes[:max_diseases]},
        }

    llm_used   = False
    llm_scores: Dict[str, float] = {}

    if use_llm and llm_api_key:
        existing_syms = {_get_gene_symbol(g) for g in (ranked_genes or []) if _get_gene_symbol(g)}
        candidates    = sorted(
            [s for s in (existing_syms or ot_gene_scores.keys()) if s in ot_gene_scores],
            key=lambda s: ot_gene_scores[s], reverse=True
        )[:400]
        if candidates:
            llm_scores = _call_llm_gene_fit_scores(
                clinical_note=clinical_note, syndromes=syndromes,
                gene_symbols=candidates, model=llm_model, api_key=llm_api_key,
            )
            llm_used = bool(llm_scores)

    ot_w  = _clip(1.0 - llm_weight)
    llm_w = _clip(llm_weight)

    reranked: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for g in (ranked_genes or []):
        sym = _get_gene_symbol(g)
        if not sym:
            continue
        seen.add(sym)
        base   = _get_base_score(g)
        ot_sc  = ot_gene_scores.get(sym, 0.0)
        if llm_used and sym in llm_scores:
            llm_sc   = llm_scores[sym]
            combined = ot_w * ot_sc + llm_w * llm_sc
        else:
            llm_sc   = None
            combined = ot_sc
        delta     = _delta_from_signal(combined, max_delta)
        new_score = base + delta
        gg        = dict(g)
        gg.update({
            "locus_ot_score":    round(ot_sc, 4),
            "locus_clinical_fit": round(llm_sc, 4) if llm_sc is not None else None,
            "locus_combined":    round(combined, 4),
            "locus_delta":       round(delta, 6),
            "score":             round(new_score, 6),
            "final_score":       round(new_score, 6),
        })
        reranked.append(gg)

    injected: List[Dict[str, Any]] = []
    if inject_new:
        inject_candidates: List[Dict[str, Any]] = []
        for sym, ot_sc in ot_gene_scores.items():
            if sym in seen:
                continue
            llm_sc = llm_scores.get(sym, 0.5) if llm_used else 0.5
            if llm_used and sym in llm_scores and llm_sc < min_inject_llm_score:
                continue
            combined = ot_w * ot_sc + llm_w * llm_sc
            delta    = _delta_from_signal(combined, max_delta)
            score    = inject_base_score + delta
            inject_candidates.append({
                "symbol":             sym,
                "match_type":         "Locus Injection" + (" + LLM" if llm_used else ""),
                "locus_ot_score":     round(ot_sc, 4),
                "locus_clinical_fit": round(llm_sc, 4) if llm_used else None,
                "locus_combined":     round(combined, 4),
                "locus_delta":        round(delta, 6),
                "score":              round(score, 6),
                "final_score":        round(score, 6),
            })
        inject_candidates.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
        injected = inject_candidates[:int(cfg.get("inject_cap", 400))]
        reranked.extend(injected)

    if len(reranked) < 20:
        existing_symbols = {_get_gene_symbol(g) for g in reranked}
        fallback = sorted(
            [
                {"symbol": sym, "match_type": "Locus Fallback",
                 "locus_ot_score": round(ot_sc, 4), "locus_clinical_fit": None,
                 "locus_combined": round(ot_sc, 4), "locus_delta": 0.0,
                 "score": round(0.001 + ot_sc * 0.001, 6),
                 "final_score": round(0.001 + ot_sc * 0.001, 6),
                 "note": "Low-confidence OT gene (fallback)"}
                for sym, ot_sc in ot_gene_scores.items() if sym not in existing_symbols
            ],
            key=lambda x: x["locus_ot_score"], reverse=True,
        )
        reranked.extend(fallback[:max(0, 20 - len(reranked))])

    reranked.sort(key=lambda x: (-float(x.get("final_score", 0.0)), _get_gene_symbol(x)))

    return {
        "triggered":      True,
        "reranked_genes": reranked,
        "evidence": {
            "disease_matches": disease_matches,
            "syndromes_used":  syndromes[:max_diseases],
            "llm_used":        llm_used,
            "llm_model":       llm_model if llm_used else None,
            "injected_count":  len(injected),
            "ot_genes_found":  len(ot_gene_scores),
            "config_used":     cfg,
        },
    }


if __name__ == "__main__":
    mcp.run()