import json
import math
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import mygene
import requests
from genopyc import get_associations, get_closest_genes, get_variants_info, geneId_mapping
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from shared_config import get_node_config

mcp = FastMCP("biochemical_refiner")
mg = mygene.MyGeneInfo()
KEGG_BASE = "https://rest.kegg.jp"


def _minmax_norm_map(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def _stable_tiebreak(symbol: str) -> float:
    s = sum((i + 1) * ord(c) for i, c in enumerate(symbol or ""))
    return (s % 1000) * 1e-9


def _idf_like(pathway_gene_count: int) -> float:
    return 1.0 / math.log(2.0 + max(0, int(pathway_gene_count)))


class BiochemClassified(BaseModel):
    metabolites: List[str] = Field(default_factory=list)
    enzymes: List[str] = Field(default_factory=list)
    parameters: List[str] = Field(default_factory=list)
    ignored: List[str] = Field(default_factory=list)


def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(4, (2**attempt)) + random.random())


def post_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int, tries: int = 4):
    last_err = None
    for attempt in range(tries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            _sleep_backoff(attempt)
    raise last_err


def kegg_lines(path: str, timeout: int = 20, tries: int = 3) -> List[str]:
    url = KEGG_BASE + path
    headers = {"User-Agent": "bioinformaticsmas/1.0"}
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 400:
                return []
            r.raise_for_status()
            txt = (r.text or "").strip()
            if not txt:
                return []
            return [ln for ln in txt.splitlines() if ln.strip()]
        except (requests.Timeout, requests.ConnectionError):
            _sleep_backoff(attempt)
        except requests.HTTPError:
            return []
    return []


def _parse_kegg_link(lines: List[str]) -> List[tuple]:
    out = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) == 2:
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def find_kegg_ids(kind: str, query: str, max_hits: int = 5) -> List[str]:
    q = quote(query)
    lines = kegg_lines(f"/find/{kind}/{q}")
    out = []
    for ln in lines[:max_hits]:
        parts = ln.split("\t")
        if parts and parts[0].strip():
            out.append(parts[0].strip())
    return out


def convert_hsa_to_symbols(hsa_ids: List[str]) -> List[str]:
    if not hsa_ids:
        return []
    hsa_ids = list({str(x) for x in hsa_ids if isinstance(x, str) and x.startswith("hsa:")})
    if not hsa_ids:
        return []
    symbols: Set[str] = set()

    def _convert_batch(batch: List[str]) -> None:
        batch_str = "+".join(batch)
        conv_lines = kegg_lines(f"/conv/ncbi-geneid/{batch_str}")
        ncbi_ids = []
        for ln in conv_lines:
            if "\t" in ln:
                ncbi_ids.append(ln.split("\t")[1].replace("ncbi-geneid:", ""))
        if not ncbi_ids:
            return
        info = mg.querymany(ncbi_ids, scopes="entrezgene", fields="symbol", species="human", verbose=False)
        for rec in info or []:
            sym = rec.get("symbol")
            if sym:
                symbols.add(sym)

    batches = [hsa_ids[i : i + 50] for i in range(0, len(hsa_ids), 50)]
    with ThreadPoolExecutor(max_workers=min(8, len(batches) or 1)) as ex:
        list(ex.map(_convert_batch, batches))
    return sorted(symbols)


def kegg_gene_symbol_to_hsa_id(symbol: str) -> Optional[str]:
    lines = kegg_lines(f"/find/genes/{quote(symbol)}")
    for ln in lines[:25]:
        kid = ln.split("\t")[0].strip()
        if kid.startswith("hsa:"):
            return kid
    return None


def get_all_genes_from_kegg_pathway(pathway_id: str) -> List[str]:
    if not pathway_id.startswith("path:"):
        pathway_id = f"path:{pathway_id}"
    gene_links = _parse_kegg_link(kegg_lines(f"/link/hsa/{pathway_id}"))
    hsa_ids = [tgt for _, tgt in gene_links if tgt.startswith("hsa:")]
    return convert_hsa_to_symbols(hsa_ids)


def pathways_for_compound(cpd_id: str) -> List[str]:
    links = _parse_kegg_link(kegg_lines(f"/link/pathway/{cpd_id}"))
    return sorted({tgt for _, tgt in links if tgt.startswith("path:hsa")})


def pathways_for_ec(ec_id: str) -> List[str]:
    links = _parse_kegg_link(kegg_lines(f"/link/pathway/{ec_id}"))
    return sorted({tgt for _, tgt in links if tgt.startswith("path:hsa")})


def pathways_for_hsa_gene(hsa_id: str) -> List[str]:
    links = _parse_kegg_link(kegg_lines(f"/link/pathway/{hsa_id}"))
    return sorted({tgt for _, tgt in links if tgt.startswith("path:hsa")})


def _llm_json(
    *,
    api_key: str,
    model: str,
    endpoint: str,
    prompt: str,
    timeout: int,
    temperature: float,
) -> Any:
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Provide api_key or set OPENAI_API_KEY.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
    }
    r = post_with_retries(endpoint, headers, payload, timeout=timeout, tries=4)
    content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise ValueError(f"LLM returned non-JSON: {content}")


def llm_classify_biochemical_findings(
    biochemical_findings: List[str],
    *,
    api_key: str = "OPENAI_API_KEY",
    model: str = "gpt-4o-mini",
    endpoint: str = "https://api.openai.com/v1/chat/completions",
    timeout: int = 30,
) -> BiochemClassified:
    prompt = f"""
You are an expert clinical biochemist.
Classify each input string into exactly ONE of: 'metabolites', 'enzymes', 'parameters', 'ignored'.
Output STRICT JSON only:
{{"metabolites":[],"enzymes":[],"parameters":[],"ignored":[]}}
INPUT: {json.dumps(biochemical_findings, indent=2)}
""".strip()
    data = _llm_json(api_key=api_key, model=model, endpoint=endpoint, prompt=prompt, timeout=timeout, temperature=0.0)
    return BiochemClassified.model_validate(data)


def get_genes_from_metabolite_pathways(metabolite: str, cap_pathways: int = 25) -> Dict[str, Any]:
    trace = {"query": metabolite, "compound_id": None, "pathways": [], "pathway_gene_counts": {}, "symbols": []}
    cpd_ids = find_kegg_ids("compound", metabolite, max_hits=3)
    if not cpd_ids:
        return trace
    cpd = cpd_ids[0]
    trace["compound_id"] = cpd
    pws = pathways_for_compound(cpd)[:cap_pathways]
    trace["pathways"] = pws
    genes: Set[str] = set()
    for pw in pws:
        syms = get_all_genes_from_kegg_pathway(pw)
        trace["pathway_gene_counts"][pw] = len(syms)
        genes.update(syms)
    trace["symbols"] = sorted(genes)
    return trace


def get_genes_from_metabolite_reaction_chain(metabolite: str) -> Dict[str, Any]:
    trace = {"query": metabolite, "compound_id": None, "reactions": [], "ecs": [], "hsa_ids": [], "symbols": []}
    cpd_ids = find_kegg_ids("compound", metabolite, max_hits=3)
    if not cpd_ids:
        return trace
    cpd = cpd_ids[0]
    trace["compound_id"] = cpd
    rxn_links = _parse_kegg_link(kegg_lines(f"/link/reaction/{cpd}"))
    rxns = sorted({tgt for _, tgt in rxn_links if tgt.startswith("rn:")})
    trace["reactions"] = rxns
    ecs: Set[str] = set()
    for rn in rxns[:50]:
        ec_links = _parse_kegg_link(kegg_lines(f"/link/enzyme/{rn}"))
        ecs.update({tgt for _, tgt in ec_links if tgt.startswith("ec:")})
    ecs_list = sorted(ecs)
    trace["ecs"] = ecs_list
    hsa_ids: Set[str] = set()
    for ec in ecs_list[:200]:
        gene_links = _parse_kegg_link(kegg_lines(f"/link/hsa/{ec}"))
        hsa_ids.update({tgt for _, tgt in gene_links if tgt.startswith("hsa:")})
    hsa_list = sorted(hsa_ids)
    trace["hsa_ids"] = hsa_list
    trace["symbols"] = convert_hsa_to_symbols(hsa_list)
    return trace


def _normalize_enzyme_phrase(x: str) -> str:
    x = (x or "").strip().lower()
    x = re.sub(r"\b(low|reduced|decreased|elevated|high|increased|normal)\b", "", x)
    x = re.sub(r"\b(activity|level|levels|assay)\b", "", x)
    return re.sub(r"\s+", " ", x).strip()


def uniprot_genes_for_enzyme(enzyme_phrase: str, size: int = 25) -> List[str]:
    term = _normalize_enzyme_phrase(enzyme_phrase)
    if not term:
        return []
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"({term}) AND organism_id:9606",
        "fields": "accession,protein_name,gene_names",
        "format": "json",
        "size": int(size),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    symbols: Set[str] = set()
    for entry in data.get("results", []) or []:
        for g in entry.get("genes", []) or []:
            sym = (g.get("geneName") or {}).get("value")
            if sym:
                symbols.add(sym)
    return sorted(symbols)


def get_genes_from_enzyme_pathways(enzyme_phrase: str, cap_pathways: int = 25) -> Dict[str, Any]:
    trace = {"query": enzyme_phrase, "ec_ids": [], "pathways": [], "pathway_gene_counts": {}, "symbols": []}
    ec_ids = [x for x in find_kegg_ids("enzyme", enzyme_phrase, max_hits=5) if x.startswith("ec:")]
    if not ec_ids:
        return trace
    trace["ec_ids"] = ec_ids
    all_pws: List[str] = []
    for ec in ec_ids:
        all_pws.extend(pathways_for_ec(ec))
    pws = list(dict.fromkeys(all_pws))[:cap_pathways]
    trace["pathways"] = pws
    genes: Set[str] = set()
    for pw in pws:
        syms = get_all_genes_from_kegg_pathway(pw)
        trace["pathway_gene_counts"][pw] = len(syms)
        genes.update(syms)
    trace["symbols"] = sorted(genes)
    return trace


def search_efo_id(query: str) -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None
    for base in ("https://www.ebi.ac.uk/ols4/api", "https://www.ebi.ac.uk/ols/api"):
        try:
            url = f"{base}/search"
            params = {"q": q, "ontology": "efo", "rows": 10, "exact": "false"}
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            docs = ((r.json().get("response") or {}).get("docs")) or []
            for d in docs:
                sf = d.get("short_form")
                if isinstance(sf, str) and sf.startswith("EFO_"):
                    return sf
            for d in docs:
                iri = d.get("iri")
                if isinstance(iri, str):
                    tail = iri.rsplit("/", 1)[-1]
                    if tail.startswith("EFO_"):
                        return tail
            for d in docs:
                obo_id = d.get("obo_id")
                if isinstance(obo_id, str) and obo_id.startswith("EFO:"):
                    return obo_id.replace("EFO:", "EFO_")
        except Exception:
            continue
    return None


def _extract_rsid_list(df) -> List[str]:
    if df is None or getattr(df, "empty", True):
        return []
    for col in ["rsid", "rs_id", "variant", "variant_id", "snp", "snps", "SNP", "SNPs"]:
        if hasattr(df, "columns") and col in df.columns:
            vals = df[col].dropna().astype(str).tolist()
            rs = [v for v in vals if v.startswith("rs")]
            return rs if rs else vals
    rs: List[str] = []
    if hasattr(df, "columns"):
        for col in df.columns:
            vals = df[col].dropna().astype(str).tolist()
            rs.extend([v for v in vals if v.startswith("rs")])
    return list(dict.fromkeys(rs))


def get_genes_from_parameter_gwas_and_pathways(param: str, cap_pathways: int = 25) -> Dict[str, Any]:
    trace = {
        "query": param,
        "efo_id": None,
        "rsids": [],
        "gwas_proximity_symbols": [],
        "expanded_pathways": [],
        "pathway_gene_counts": {},
        "symbols": [],
        "note": "",
    }
    efo_id = search_efo_id(param)
    trace["efo_id"] = efo_id
    if not efo_id:
        trace["note"] = "No EFO ID found"
        return trace
    try:
        gwas_df = get_associations(efo_id)
        rsids = _extract_rsid_list(gwas_df)[:10]
        trace["rsids"] = rsids
        prox_symbols: Set[str] = set()
        for rs in rsids:
            v_info = get_variants_info(rs)
            if rs not in v_info:
                continue
            mapping = (v_info[rs].get("mappings") or [{}])[0]
            ch, pos = mapping.get("seq_region_name"), mapping.get("start")
            if not ch or not pos:
                continue
            _, up, down = get_closest_genes(ch=ch, position=pos, window_size=500_000, position_id=rs)
            raw_ids = [i for i in (up, down) if i]
            if raw_ids:
                syms = geneId_mapping(raw_ids, source="ensembl", target="symbol") or []
                for s in syms:
                    if isinstance(s, str) and s.strip():
                        prox_symbols.add(s.strip())
        trace["gwas_proximity_symbols"] = sorted(prox_symbols)
        all_pws: List[str] = []
        for sym in prox_symbols:
            hsa_id = kegg_gene_symbol_to_hsa_id(sym)
            if hsa_id:
                all_pws.extend(pathways_for_hsa_gene(hsa_id))
        pws = list(dict.fromkeys(all_pws))[:cap_pathways]
        trace["expanded_pathways"] = pws
        genes: Set[str] = set(prox_symbols)
        for pw in pws:
            syms = get_all_genes_from_kegg_pathway(pw)
            trace["pathway_gene_counts"][pw] = len(syms)
            genes.update(syms)
        trace["symbols"] = sorted(genes)
        return trace
    except Exception as e:
        trace["note"] = f"GenoPyc/GWAS error: {e}"
        return trace


def llm_reason_biochemical_refinement(
    biochemical_findings: List[str],
    classified: Dict[str, Any],
    extracted: Dict[str, Any],
    reranked_genes: List[Dict[str, Any]],
    *,
    api_key: str = "OPENAI_API_KEY",
    model: str = "gpt-4o-mini",
    endpoint: str = "https://api.openai.com/v1/chat/completions",
    timeout: int = 60,
    top_n: int = 25,
) -> Dict[str, Any]:
    top_genes = reranked_genes[: max(1, int(top_n))]
    union_syms = extracted.get("union_gene_symbols", [])
    extracted_summary = {
        "summary": extracted.get("summary", {}),
        "union_gene_symbols_count": len(union_syms) if isinstance(union_syms, list) else None,
        "metabolites_keys": list((extracted.get("details", {}).get("metabolites") or {}).keys()),
        "enzymes_keys": list((extracted.get("details", {}).get("enzymes") or {}).keys()),
        "parameters_keys": list((extracted.get("details", {}).get("parameters") or {}).keys()),
    }
    prompt = f"""
You are an expert clinical biochemist + neurogenomics specialist.
Explain why genes were boosted or injected using biochemical evidence.
Return STRICT JSON only:
{{
  "overview":{{"biochemical_findings_used":[],"key_drivers":[],"notes":""}},
  "top_gene_explanations":[{{"symbol":"","final_rank":1,"match_type":"","why":"","evidence":{{"metabolite_support":[],"enzyme_support":[],"parameter_support":[],"evidence_types":[]}}}}],
  "sanity_checks":{{"do_not_overinterpret":""}}
}}
INPUTS:
biochemical_findings: {json.dumps(biochemical_findings, indent=2)}
classified: {json.dumps(classified, indent=2)}
extracted_summary: {json.dumps(extracted_summary, indent=2)}
top_reranked_genes: {json.dumps(top_genes, indent=2)}
""".strip()
    return _llm_json(
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        prompt=prompt,
        timeout=timeout,
        temperature=0.1,
    )


def extract_biochem_pathway_genes(
    biochemical_findings: List[str],
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
    endpoint: str = "https://api.openai.com/v1/chat/completions",
    cap_pathways: int = 25,
    include_metabolite_reaction_chain: bool = True,
    include_uniprot_enzyme_genes: bool = True,
) -> Dict[str, Any]:
    classified = llm_classify_biochemical_findings(
        biochemical_findings,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
    )
    logs: Dict[str, Any] = {"classified": classified.model_dump()}
    union_genes: Set[str] = set()

    def _process_metabolite(met: str):
        pw_trace = get_genes_from_metabolite_pathways(met, cap_pathways=cap_pathways)
        result = {"pathway_expansion": pw_trace}
        syms = set(pw_trace.get("symbols", []))
        if include_metabolite_reaction_chain:
            rc_trace = get_genes_from_metabolite_reaction_chain(met)
            result["reaction_chain"] = rc_trace
            syms.update(rc_trace.get("symbols", []))
        return met, result, syms

    logs["metabolites"] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(classified.metabolites) or 1)) as ex:
        for met, result, syms in ex.map(_process_metabolite, classified.metabolites):
            logs["metabolites"][met] = result
            union_genes.update(syms)

    def _process_enzyme(enz: str):
        pw_trace = get_genes_from_enzyme_pathways(enz, cap_pathways=cap_pathways)
        result = {"pathway_expansion": pw_trace}
        syms = set(pw_trace.get("symbols", []))
        if include_uniprot_enzyme_genes:
            up_syms = uniprot_genes_for_enzyme(enz)
            result["uniprot_direct_symbols"] = up_syms
            syms.update(up_syms)
        return enz, result, syms

    logs["enzymes"] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(classified.enzymes) or 1)) as ex:
        for enz, result, syms in ex.map(_process_enzyme, classified.enzymes):
            logs["enzymes"][enz] = result
            union_genes.update(syms)

    def _process_param(param: str):
        p_trace = get_genes_from_parameter_gwas_and_pathways(param, cap_pathways=cap_pathways)
        return param, p_trace, set(p_trace.get("symbols", []))

    logs["parameters"] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(classified.parameters) or 1)) as ex:
        for param, p_trace, syms in ex.map(_process_param, classified.parameters):
            logs["parameters"][param] = p_trace
            union_genes.update(syms)

    return {
        "summary": {
            "inputs": len(biochemical_findings),
            "metabolites": len(classified.metabolites),
            "enzymes": len(classified.enzymes),
            "parameters": len(classified.parameters),
            "union_genes": len(union_genes),
        },
        "details": logs,
        "union_gene_symbols": sorted(union_genes),
    }


@mcp.tool()
def refine_biochemical_candidates(
    biochemical_findings: Optional[List[str]] = None,
    candidate_genes: Optional[List[Dict[str, Any]]] = None,
    model: str = "gpt-4o-mini",
    api_key: str = "OPENAI_API_KEY",
    include_reasoning: bool = True,
    reasoning_top_n: int = 25,
    blackboard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    biochemical_findings = biochemical_findings or []
    candidate_genes = candidate_genes or []

    cfg = get_node_config(blackboard or {}, "biochemist")
    cap_pathways = int(cfg["cap_pathways"])
    eps = float(cfg["eps"])
    discovery_base = float(cfg["discovery_base"])
    w_direct_rxn = float(cfg["w_direct_rxn"])
    w_uniprot = float(cfg["w_uniprot"])
    w_gwas_prox = float(cfg["w_gwas_prox"])
    w_pathway = float(cfg["w_pathway"])
    inject_new = bool(cfg.get("inject_new", True))
    _include_reason = bool(cfg.get("include_reasoning", include_reasoning))
    _reason_top_n = int(cfg.get("reasoning_top_n", reasoning_top_n))

    extracted = extract_biochem_pathway_genes(
        biochemical_findings=biochemical_findings,
        api_key=api_key,
        model=model,
        cap_pathways=cap_pathways,
        include_metabolite_reaction_chain=True,
        include_uniprot_enzyme_genes=True,
    )
    details = extracted.get("details", {}) or {}

    gene_bio_raw: Dict[str, float] = defaultdict(float)
    gene_bio_direct_hits: Dict[str, int] = defaultdict(int)
    gene_bio_path_hits: Dict[str, int] = defaultdict(int)

    for met_payload in (details.get("metabolites") or {}).values():
        rc = met_payload.get("reaction_chain") or {}
        for sym in (rc.get("symbols") or []):
            sym = str(sym).strip()
            if sym:
                gene_bio_raw[sym] += w_direct_rxn
                gene_bio_direct_hits[sym] += 1
        pw = met_payload.get("pathway_expansion") or {}
        pw_counts = pw.get("pathway_gene_counts") or {}
        pw_syms = pw.get("symbols") or []
        if pw_syms and pw_counts:
            avg_idf = sum(_idf_like(c) for c in pw_counts.values()) / max(1, len(pw_counts))
            inc = w_pathway * avg_idf
            for sym in pw_syms:
                sym = str(sym).strip()
                if sym:
                    gene_bio_raw[sym] += inc
                    gene_bio_path_hits[sym] += 1

    for enz_payload in (details.get("enzymes") or {}).values():
        for sym in (enz_payload.get("uniprot_direct_symbols") or []):
            sym = str(sym).strip()
            if sym:
                gene_bio_raw[sym] += w_uniprot
                gene_bio_direct_hits[sym] += 1
        pw = enz_payload.get("pathway_expansion") or {}
        pw_counts = pw.get("pathway_gene_counts") or {}
        pw_syms = pw.get("symbols") or []
        if pw_syms and pw_counts:
            avg_idf = sum(_idf_like(c) for c in pw_counts.values()) / max(1, len(pw_counts))
            inc = w_pathway * avg_idf
            for sym in pw_syms:
                sym = str(sym).strip()
                if sym:
                    gene_bio_raw[sym] += inc
                    gene_bio_path_hits[sym] += 1

    for p_trace in (details.get("parameters") or {}).values():
        for sym in (p_trace.get("gwas_proximity_symbols") or []):
            sym = str(sym).strip()
            if sym:
                gene_bio_raw[sym] += w_gwas_prox
                gene_bio_direct_hits[sym] += 1
        pw_counts = p_trace.get("pathway_gene_counts") or {}
        pw_syms = p_trace.get("symbols") or []
        if pw_syms and pw_counts:
            avg_idf = sum(_idf_like(c) for c in pw_counts.values()) / max(1, len(pw_counts))
            inc = w_pathway * avg_idf
            for sym in pw_syms:
                sym = str(sym).strip()
                if sym:
                    gene_bio_raw[sym] += inc
                    gene_bio_path_hits[sym] += 1

    gene_bio_norm = _minmax_norm_map(dict(gene_bio_raw))

    def _phenotype_score(g: Dict[str, Any]) -> float:
        for key in ("adjusted_score", "score", "base_score"):
            v = g.get(key, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    return 0.0
        return 0.0

    ph_scores = [_phenotype_score(g) for g in candidate_genes] or [0.0]
    s_min, s_max = min(ph_scores), max(ph_scores)
    denom = s_max - s_min

    def _norm_hpo(s: float) -> float:
        if denom <= 1e-12:
            return 0.0
        return (s - s_min) / denom

    reranked: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    boosted_existing = 0

    for g in candidate_genes:
        sym = str(g.get("symbol", "")).strip()
        if not sym:
            continue
        seen.add(sym)
        s_i = _phenotype_score(g)
        hpo_norm = _norm_hpo(s_i)
        bio_n = float(gene_bio_norm.get(sym, 0.0))
        direct_hits = int(gene_bio_direct_hits.get(sym, 0))
        path_hits = int(gene_bio_path_hits.get(sym, 0))
        if direct_hits > 0:
            match_type = "Direct Biochemical Match"
        elif path_hits > 0:
            match_type = "Pathway Neighborhood"
        else:
            match_type = "Phenotype Only"
        if bio_n > 0:
            boosted_existing += 1
        final_score = hpo_norm + (eps * bio_n) + _stable_tiebreak(sym)
        reranked.append(
            {
                **g,
                "match_type": match_type,
                "prior_score": g.get("score", None),
                "phenotype_score_raw": s_i,
                "hpo_norm": round(hpo_norm, 6),
                "biochem_norm": round(bio_n, 6),
                "score": round(final_score, 6),
                "final_score": round(final_score, 6),
                "score_components": {
                    "input_score": s_i,
                    "hpo_norm": round(hpo_norm, 6),
                    "biochem_raw": round(float(gene_bio_raw.get(sym, 0.0)), 6),
                    "biochem_norm": round(bio_n, 6),
                    "eps": eps,
                    "direct_hits": direct_hits,
                    "path_hits": path_hits,
                    "match_type": match_type,
                },
            }
        )

    injected = 0
    if inject_new:
        for sym, bio_n in sorted(gene_bio_norm.items()):
            if sym in seen:
                continue
            final_score = discovery_base + (0.06 * float(bio_n)) + _stable_tiebreak(sym)
            reranked.append(
                {
                    "symbol": sym,
                    "match_type": "Biochem Discovery",
                    "phenotype_score_raw": None,
                    "hpo_norm": discovery_base,
                    "biochem_norm": round(float(bio_n), 6),
                    "score": round(final_score, 6),
                    "final_score": round(final_score, 6),
                    "note": "Bio-driven candidate (not in HPO pool)",
                    "score_components": {
                        "input_score": None,
                        "hpo_norm": discovery_base,
                        "biochem_raw": round(float(gene_bio_raw.get(sym, 0.0)), 6),
                        "biochem_norm": round(float(bio_n), 6),
                    },
                }
            )
            injected += 1

    reranked.sort(key=lambda x: (-float(x.get("final_score", 0.0)), str(x.get("symbol", ""))))
    for i, gg in enumerate(reranked, start=1):
        gg["final_rank"] = i

    reasoning = None
    if _include_reason:
        classified_dump = details.get("classified", {})
        reasoning = llm_reason_biochemical_refinement(
            biochemical_findings=biochemical_findings,
            classified=classified_dump,
            extracted=extracted,
            reranked_genes=reranked,
            api_key=api_key,
            model=model,
            top_n=_reason_top_n,
        )

    return {
        "summary": (
            f"Analyzed {len(candidate_genes)} phenotype-ranked genes. "
            f"Boosted {boosted_existing} existing genes via biochemistry. "
            f"Injected {injected} additional bio-only candidates."
        ),
        "reranked_genes": reranked,
        "reasoning": reasoning,
        "biochem_debug": {
            "union_genes": (extracted.get("summary", {}) or {}).get("union_genes"),
            "bio_score_nonzero": sum(1 for v in gene_bio_norm.values() if v > 0),
            "config_used": cfg,
        },
    }


if __name__ == "__main__":
    mcp.run()