from __future__ import annotations
import json
import math
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import requests
from mcp.server.fastmcp import FastMCP
from pyhpo import HPOSet, Ontology
from pyhpo.stats import EnrichmentModel
from shared_config import get_node_config

mcp = FastMCP("gene_extraction")
_ = Ontology()

IC_FALLBACK = 0.1
P_BOOST_CAP = 18.0


@lru_cache(maxsize=200_000)
def _ic(hpo_id: str) -> float:
    term = Ontology.get_hpo_object(hpo_id)
    if term is None:
        return IC_FALLBACK
    try:
        ic_val = term.information_content.omim
        if ic_val is not None and ic_val > 0:
            return float(ic_val)
    except (AttributeError, TypeError):
        pass
    return IC_FALLBACK


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-12:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _stable_tiebreak(symbol: str) -> float:
    s = sum((i + 1) * ord(c) for i, c in enumerate(symbol or ""))
    return (s % 1000) * 1e-9


def _safe_neg_log10_p(pval: float) -> float:
    p = max(float(pval or 1.0), 1e-300)
    return -math.log10(p)


def _json_loads_loose(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(text[a : b + 1])
        raise


def _post_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    *,
    timeout_s: int = 180,
    tries: int = 4,
):
    last_err: Optional[Exception] = None
    for attempt in range(max(1, int(tries))):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=(20, timeout_s))
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(min(2**attempt + random.random(), 10))
    raise last_err


def _clean_hpo_ids(hpo_ids: List[str]) -> List[str]:
    out, seen = [], set()
    for h in hpo_ids or []:
        h = str(h or "").strip()
        if h and h not in seen:
            out.append(h)
            seen.add(h)
    return out


def prune_redundant_hpos(hpo_ids: List[str]) -> List[str]:
    terms = [Ontology.get_hpo_object(h) for h in hpo_ids]
    terms = [t for t in terms if t is not None]
    term_set = set(terms)

    keep_ids: List[str] = []
    for t in terms:
        redundant = any((other != t and t in other.all_parents) for other in term_set)
        if not redundant:
            keep_ids.append(t.id)

    out, seen = [], set()
    for h in hpo_ids:
        h = str(h).strip()
        if h in keep_ids and h not in seen:
            out.append(h)
            seen.add(h)
    return out


def gene_to_hposet(gene) -> Optional[HPOSet]:
    if hasattr(gene, "hpo_set") and callable(gene.hpo_set):
        try:
            gs = gene.hpo_set()
            if gs and len(gs) > 0:
                return gs
        except Exception:
            pass

    raw = getattr(gene, "hpo", None)
    if raw:
        try:
            ids = [t.id if hasattr(t, "id") else str(t).strip() for t in raw]
            ids = [i for i in ids if i]
            if ids:
                return HPOSet.from_queries(ids)
        except Exception:
            pass
    return None


def _compute_coverage_and_ic_coverage(patient_set: HPOSet, gene_set: HPOSet) -> Tuple[float, float, List[str]]:
    if len(patient_set) == 0 or len(gene_set) == 0:
        return 0.0, 0.0, []

    gene_terms = list(gene_set)
    covered_patient_ids: set[str] = set()

    for p_term in patient_set:
        for g_term in gene_terms:
            if p_term.id == g_term.id:
                covered_patient_ids.add(p_term.id)
                break
            if g_term in p_term.all_parents:  # gene broader → patient more specific
                covered_patient_ids.add(p_term.id)
                break
            if p_term in g_term.all_parents:  # patient broader → gene narrower (still relevant)
                covered_patient_ids.add(p_term.id)
                break

    coverage = len(covered_patient_ids) / max(1, len(patient_set))
    total_ic = sum(_ic(t.id) for t in patient_set)
    covered_ic = sum(_ic(t.id) for t in patient_set if t.id in covered_patient_ids)
    ic_coverage = (covered_ic / total_ic) if total_ic > 0 else coverage
    matched = [t.id for t in patient_set if t.id in covered_patient_ids]
    return float(coverage), float(ic_coverage), matched


def disease_expanded_gene_prior(
    patient_set: HPOSet,
    top_diseases: int = 30,
    top_genes_per_disease: int = 200,
) -> Dict[str, float]:
    prior: Dict[str, float] = {}
    try:
        omim_model = EnrichmentModel("omim")
        gene_model = EnrichmentModel("gene")
        omim_hits = omim_model.enrichment(method="hypergeom", hposet=patient_set)[: int(top_diseases)]
    except Exception:
        return prior

    for hit in omim_hits:
        disease = hit.get("item")
        if disease is None or not hasattr(disease, "hpo_set"):
            continue
        try:
            dhpos = disease.hpo_set()
            g_hits = gene_model.enrichment(method="hypergeom", hposet=dhpos)[: int(top_genes_per_disease)]
        except Exception:
            continue

        for gh in g_hits:
            g = gh.get("item")
            if g is None:
                continue
            sym = getattr(g, "name", None)
            if not sym:
                continue
            p = float(gh.get("p_value", 1.0))
            prior[sym] = prior.get(sym, 0.0) + _safe_neg_log10_p(p)

    return prior


def _extract_locus_prior(previous_candidates: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
    prior: Dict[str, float] = {}
    if not previous_candidates:
        return prior

    def pick_raw(d: Dict[str, Any]) -> Optional[float]:
        for k in ("locus_combined", "locus_ot_score", "final_score", "score", "locus_delta"):
            v = d.get(k)
            try:
                if v is None:
                    continue
                fv = float(v)
                if math.isfinite(fv):
                    return fv
            except Exception:
                continue
        return None

    for g in previous_candidates:
        sym = str(g.get("symbol", "")).strip()
        if not sym:
            continue
        raw = pick_raw(g)
        if raw is not None:
            prior[sym] = raw

    return prior


def llm_reason_gene_ranking(
    query_hpos: List[str],
    pruned_hpos: List[str],
    weights: Dict[str, float],
    ranked_genes: List[Dict[str, Any]],
    *,
    model: str = "gpt-4o-mini",
    endpoint: str = "https://api.openai.com/v1/chat/completions",
    timeout: int = 180,
    top_n: int = 25,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    # Keep behavior: use provided api_key if present; else fall back to env; else raise.
    key = (api_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("No API key provided for reasoning (api_key param empty and OPENAI_API_KEY not set).")

    top_genes = ranked_genes[: max(1, int(top_n))]

    prompt = f"""
You are a clinical genetics assistant.
Explain briefly why the top genes were ranked highly for the patient's HPO set.
Ground reasoning ONLY in: semantic_similarity, coverage, ic_coverage, p_value boost,
disease prior boost, locus prior boost.

Return STRICT JSON only:
{{
  "overview": {{
    "query_hpos": [],
    "pruned_hpos": [],
    "weighting": {json.dumps({k: 0 for k in weights.keys()})},
    "how_to_read": "1-2 sentences"
  }},
  "top_gene_explanations": [
    {{
      "symbol": "GENE",
      "final_rank": 1,
      "why": "1-2 sentences referencing the numeric signals",
      "signals": {{
        "semantic_similarity": 0.0,
        "coverage": 0.0,
        "ic_coverage": 0.0,
        "p_value": 1.0,
        "disease_prior": 0.0,
        "locus_prior": 0.0
      }}
    }}
  ],
  "sanity_checks": {{"limitations": "1 sentence"}}
}}

query_hpos: {json.dumps(query_hpos, indent=2)}
pruned_hpos: {json.dumps(pruned_hpos, indent=2)}
weights: {json.dumps(weights, indent=2)}
top_genes: {json.dumps(top_genes, indent=2)}
""".strip()

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    r = _post_with_retries(endpoint, headers, payload, timeout_s=int(timeout), tries=3)
    content = r.json()["choices"][0]["message"]["content"]
    return _json_loads_loose(content)


@dataclass(frozen=True)
class _Settings:
    top_n: int
    max_previous_to_consider: int
    prune: bool
    similarity_method: str
    similarity_combine: str
    include_reasoning: bool
    reasoning_top_n: int
    reasoning_model: str
    restrict_to_previous: bool
    top_diseases: int
    top_genes_per_disease: int


@dataclass(frozen=True)
class _Weights:
    w_semantic: float
    w_ic_coverage: float
    w_p_boost: float
    w_disease_boost: float
    w_locus_boost: float


def _read_cfg(
    blackboard: Optional[Dict[str, Any]],
    *,
    top_n: int,
    max_previous_to_consider: int,
    prune: bool,
    similarity_method: str,
    similarity_combine: str,
    include_reasoning: bool,
    reasoning_top_n: int,
    reasoning_model: str,
    restrict_to_previous: bool,
) -> Tuple[_Settings, _Weights]:
    cfg = get_node_config(blackboard or {}, "geneticist")

    settings = _Settings(
        top_n=int(cfg.get("top_n", top_n)),
        max_previous_to_consider=int(cfg.get("max_previous_to_consider", max_previous_to_consider)),
        prune=bool(cfg.get("prune", prune)),
        similarity_method=str(cfg.get("similarity_method", similarity_method)),
        similarity_combine=str(cfg.get("similarity_combine", similarity_combine)),
        include_reasoning=bool(cfg.get("include_reasoning", include_reasoning)),
        reasoning_top_n=int(cfg.get("reasoning_top_n", reasoning_top_n)),
        reasoning_model=str(cfg.get("reasoning_model", reasoning_model)),
        restrict_to_previous=bool(cfg.get("restrict", restrict_to_previous)),
        top_diseases=int(cfg.get("top_diseases", 30)),
        top_genes_per_disease=int(cfg.get("top_genes_per_disease", 200)),
    )

    weights = _Weights(
        w_semantic=float(cfg.get("w_semantic", 6.0)),
        w_ic_coverage=float(cfg.get("w_ic_coverage", 5.0)),
        w_p_boost=float(cfg.get("w_p_boost", 1.0)),
        w_disease_boost=float(cfg.get("w_disease_boost", 1.0)),
        w_locus_boost=float(cfg.get("w_locus_boost", 10.0)),
    )

    return settings, weights


def _select_gene_symbols_to_score(
    *,
    hpo_symbols: List[str],
    prev_symbols: List[str],
    max_previous_to_consider: int,
    restrict_to_previous: bool,
) -> List[str]:
    prev_symbol_set = set(prev_symbols)

    if restrict_to_previous and prev_symbol_set:
        return list(prev_symbol_set)[: max_previous_to_consider]

    keep = list(prev_symbol_set)
    for s in hpo_symbols:
        if s not in prev_symbol_set:
            keep.append(s)
        if len(keep) >= max_previous_to_consider:
            break
    return keep


def _score_and_trim(
    ranked: List[Dict[str, Any]],
    *,
    weights: _Weights,
    top_n: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not ranked:
        return ranked, {}

    sem_n = _minmax_norm([r["_semantic_raw"] for r in ranked])
    icc_n = _minmax_norm([r["_ic_cov_raw"] for r in ranked])
    pb_n = _minmax_norm([r["_p_boost_raw"] for r in ranked])
    db_n = _minmax_norm([r["_d_boost_raw"] for r in ranked])
    lb_n = _minmax_norm([r["_l_boost_raw"] for r in ranked])

    locus_has_signal = any(r["_l_boost_raw"] > 0 for r in ranked)
    w_lb_eff = weights.w_locus_boost if locus_has_signal else 0.0

    w_sum = (weights.w_semantic + weights.w_ic_coverage + weights.w_p_boost + weights.w_disease_boost + w_lb_eff) or 1.0

    effective_weights = {
        "w_semantic": weights.w_semantic,
        "w_ic_coverage": weights.w_ic_coverage,
        "w_p_boost": weights.w_p_boost,
        "w_disease_boost": weights.w_disease_boost,
        "w_locus_boost": w_lb_eff,
    }

    for i, r in enumerate(ranked):
        score = (
            (weights.w_semantic / w_sum) * sem_n[i]
            + (weights.w_ic_coverage / w_sum) * icc_n[i]
            + (weights.w_p_boost / w_sum) * pb_n[i]
            + (weights.w_disease_boost / w_sum) * db_n[i]
            + (w_lb_eff / w_sum) * lb_n[i]
        )
        r["score"] = float(score + _stable_tiebreak(r["symbol"]))
        r["final_score"] = r["score"]

        for k in ("_semantic_raw", "_ic_cov_raw", "_p_boost_raw", "_d_boost_raw", "_l_boost_raw"):
            r.pop(k, None)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked[: int(top_n)]
    return ranked, effective_weights


@mcp.tool()
def hpo_gene_ranker_with_reasoning(
    hpo_ids: List[str],
    previous_candidates: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 200,
    max_previous_to_consider: int = 400,
    *,
    prune: bool = True,
    similarity_method: str = "resnik",
    similarity_combine: str = "funSimMax",
    include_reasoning: bool = False,
    reasoning_top_n: int = 30,
    reasoning_model: str = "gpt-4o-mini",
    api_key: str = "OPENAI_API_KEY",
    restrict_to_previous: bool = False,
    blackboard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    settings, weights = _read_cfg(
        blackboard,
        top_n=top_n,
        max_previous_to_consider=max_previous_to_consider,
        prune=prune,
        similarity_method=similarity_method,
        similarity_combine=similarity_combine,
        include_reasoning=include_reasoning,
        reasoning_top_n=reasoning_top_n,
        reasoning_model=reasoning_model,
        restrict_to_previous=restrict_to_previous,
    )

    cleaned_hpos = _clean_hpo_ids(hpo_ids)
    if not cleaned_hpos:
        return {"error": "No valid HPO terms.", "candidate_genes": []}

    used_hpos = prune_redundant_hpos(cleaned_hpos) if settings.prune else cleaned_hpos
    if not used_hpos:
        return {"error": "No HPO terms after pruning.", "candidate_genes": []}

    patient_set = HPOSet.from_queries(used_hpos)

    disease_prior = disease_expanded_gene_prior(
        patient_set,
        top_diseases=settings.top_diseases,
        top_genes_per_disease=settings.top_genes_per_disease,
    )

    locus_prior_raw = _extract_locus_prior(previous_candidates)

    gene_obj_by_symbol: Dict[str, Any] = {}
    pval_by_symbol: Dict[str, float] = {}

    gene_enrich_model = EnrichmentModel("gene")
    try:
        results = gene_enrich_model.enrichment(method="hypergeom", hposet=patient_set)
    except Exception as e:
        results = []
        print(f"  Full HPO enrichment failed: {e}")

    for r in results:
        g = r.get("item")
        sym = getattr(g, "name", None)
        if not sym:
            continue
        gene_obj_by_symbol[sym] = g
        pval_by_symbol[sym] = float(r.get("p_value", 1.0))

    hpo_symbols = list(gene_obj_by_symbol.keys())
    prev_symbols = [
        str(x.get("symbol", "")).strip()
        for x in (previous_candidates or [])
        if str(x.get("symbol", "")).strip()
    ]
    prev_symbol_set = set(prev_symbols)

    gene_symbols_to_score = _select_gene_symbols_to_score(
        hpo_symbols=hpo_symbols,
        prev_symbols=prev_symbols,
        max_previous_to_consider=settings.max_previous_to_consider,
        restrict_to_previous=settings.restrict_to_previous,
    )

    ranked: List[Dict[str, Any]] = []

    for sym in gene_symbols_to_score:
        sym = str(sym).strip()
        if not sym:
            continue

        entry: Dict[str, Any] = {
            "symbol": sym,
            "matched_patient_hpos": [],
            "n_matched": 0,
            "semantic_similarity": 0.0,
            "coverage": 0.0,
            "ic_coverage": 0.0,
            "p_value": 1.0,
            "gene_hpos_preview": [],
            "disease_prior": float(disease_prior.get(sym, 0.0)),
            "locus_prior": float(locus_prior_raw.get(sym, 0.0)),
            "source": ("previous" if sym in prev_symbol_set else ("hpo" if sym in gene_obj_by_symbol else "unknown")),
            # Raw values before normalisation
            "_semantic_raw": 0.0,
            "_ic_cov_raw": 0.0,
            "_p_boost_raw": 0.0,
            "_d_boost_raw": float(disease_prior.get(sym, 0.0)),
            "_l_boost_raw": float(locus_prior_raw.get(sym, 0.0)),
        }

        gene = gene_obj_by_symbol.get(sym)
        gene_set = gene_to_hposet(gene) if gene is not None else None

        if gene_set and len(gene_set) > 0:
            semantic = patient_set.similarity(gene_set, method=settings.similarity_method, combine=settings.similarity_combine)
            coverage, ic_cov, matched = _compute_coverage_and_ic_coverage(patient_set, gene_set)
            pval = float(pval_by_symbol.get(sym, 1.0))
            p_boost = _clip(_safe_neg_log10_p(pval), 0.0, P_BOOST_CAP)

            entry.update(
                {
                    "matched_patient_hpos": matched,
                    "n_matched": len(matched),
                    "semantic_similarity": round(float(semantic), 3),
                    "coverage": round(float(coverage), 3),
                    "ic_coverage": round(float(ic_cov), 3),
                    "p_value": pval,
                    "gene_hpos_preview": [t.id for t in gene_set][:30],
                    "_semantic_raw": float(semantic),
                    "_ic_cov_raw": float(ic_cov),
                    "_p_boost_raw": float(p_boost),
                }
            )
        else:
            entry["note"] = "No HPO annotations in PyHPO (kept for locus/previous signal)."

        ranked.append(entry)

    ranked, effective_weights = _score_and_trim(ranked, weights=weights, top_n=settings.top_n)

    reasoning = None
    if settings.include_reasoning and ranked:
        reasoning = llm_reason_gene_ranking(
            query_hpos=cleaned_hpos,
            pruned_hpos=used_hpos,
            weights=effective_weights,
            ranked_genes=ranked,
            model=settings.reasoning_model,
            top_n=settings.reasoning_top_n,
            api_key=api_key
        )

    locus_prior_loaded = bool(previous_candidates) and any(
        float(x.get("locus_combined", 0) or 0) > 0 or float(x.get("locus_ot_score", 0) or 0) > 0
        for x in (previous_candidates or [])
    )

    return {
        "query_hpos": cleaned_hpos,
        "used_hpos": used_hpos,
        "candidate_genes": ranked,
        "reasoning": reasoning,
        "summary": f"Ranked {len(ranked)} genes (HPO + disease prior + locus prior).",
        "weights_used": effective_weights,
        "settings": {
            "prune": settings.prune,
            "similarity_method": settings.similarity_method,
            "similarity_combine": settings.similarity_combine,
            "restrict_to_previous": settings.restrict_to_previous,
            "previous_candidates_count": len(previous_candidates) if previous_candidates else 0,
            "locus_prior_loaded": locus_prior_loaded,
            "top_diseases": settings.top_diseases,
            "top_genes_per_disease": settings.top_genes_per_disease,
            "config_source": "blackboard" if blackboard else "defaults",
        },
    }


if __name__ == "__main__":
    mcp.run()