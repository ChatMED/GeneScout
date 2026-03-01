import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode

import requests
import xml.etree.ElementTree as ET
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

from shared_config import get_node_config

mcp = FastMCP("literature_search_engine")
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ALLOWED = re.compile(r'^[\w\s\-\(\)\[\]":\/\.\,\+]+$')


class RateLimiter:
    def __init__(self, max_calls_per_second: int = 3):
        self.max_calls = max_calls_per_second
        self.period = 1.0
        self.calls: List[float] = []
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.calls = self.calls[1:]
            self.calls.append(time.time())


RATE_LIMITER = RateLimiter(max_calls_per_second=3)


def _sleep_backoff(attempt: int, base: float = 1.2, cap: float = 8.0) -> None:
    t = min(cap, base**attempt) + random.uniform(0, 0.1)
    time.sleep(t)


def _get_with_retries(url: str, *, max_retries: int = 4, timeout: int = 20) -> requests.Response:
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            RATE_LIMITER.wait_if_needed()
            r = requests.get(url, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}")
                _sleep_backoff(attempt + 1)
                continue
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last = e
            _sleep_backoff(attempt + 1)
        except Exception as e:
            last = e
            _sleep_backoff(attempt + 1)
    raise RuntimeError(f"Request failed after retries: {url}\n{last}")


def _count_bridges(abstracts: List[str], gene_symbol: str, biochem_terms: List[str]) -> Tuple[int, int]:
    g = gene_symbol.lower()
    terms = [t.lower() for t in biochem_terms if t and t.strip()]
    if not terms:
        return 0, 0

    bridge_abs = 0
    bridge_sentence = 0

    for abs_text in abstracts:
        text = (abs_text or "").lower()
        if g not in text:
            continue
        if any(t in text for t in terms):
            bridge_abs += 1
        for sent in _SENT_SPLIT.split(text):
            if g in sent and any(t in sent for t in terms):
                bridge_sentence += 1

    return bridge_abs, bridge_sentence


def pubmed_esearch(query: str, retmax: int = 5, api_key: Optional[str] = None) -> Tuple[int, List[str]]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
        "tool": "literature_search_engine",
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{NCBI_EUTILS}/esearch.fcgi?{urlencode(params)}"
    try:
        r = _get_with_retries(url, timeout=20)
        data = (r.json() or {}).get("esearchresult", {}) or {}
        return int(data.get("count", 0) or 0), data.get("idlist", []) or []
    except Exception as e:
        print(f"esearch failed: {e}")
        return 0, []


def fetch_esummary_chunk(chunk: List[str], api_key: Optional[str], tool: str) -> Dict[str, Dict[str, Any]]:
    params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "json", "tool": tool}
    if api_key:
        params["api_key"] = api_key
    url = f"{NCBI_EUTILS}/esummary.fcgi?{urlencode(params)}"
    try:
        r = _get_with_retries(url, timeout=20)
        data = r.json() or {}
        result = data.get("result", {}) or {}

        meta: Dict[str, Dict[str, Any]] = {}
        for pmid in chunk:
            rec = result.get(str(pmid), {}) or {}
            if not rec:
                continue

            doi = None
            pmcid = None
            for aid in rec.get("articleids", []) or []:
                idtype = (aid.get("idtype") or "").lower()
                val = (aid.get("value") or "").strip()
                if not val:
                    continue
                if idtype == "doi":
                    doi = val
                elif idtype == "pmcid":
                    pmcid = val

            pubdate = (rec.get("pubdate") or "").strip()
            year = None
            if pubdate:
                m = re.search(r"\b(19|20)\d{2}\b", pubdate)
                if m:
                    year = int(m.group(0))

            authors = [(a.get("name") or "").strip() for a in rec.get("authors", []) or []]

            meta[str(pmid)] = {
                "pmid": str(pmid),
                "title": (rec.get("title") or "").strip(),
                "journal": (rec.get("source") or "").strip(),
                "pubdate": pubdate,
                "year": year,
                "doi": doi,
                "pmcid": pmcid,
                "authors": [a for a in authors if a][:6],
            }

        return meta
    except Exception as e:
        print(f"Warning: esummary chunk failed: {e}")
        return {}


def pubmed_esummary_meta(pmids: List[str], api_key: Optional[str] = None, max_workers: int = 3) -> Dict[str, Dict[str, Any]]:
    if not pmids:
        return {}
    meta: Dict[str, Dict[str, Any]] = {}
    chunks = [pmids[i : i + 30] for i in range(0, len(pmids), 30)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_esummary_chunk, chunk, api_key, "literature_search_engine"): chunk for chunk in
                   chunks}
        for future in as_completed(futures):
            meta.update(future.result())
    return meta


def fetch_efetch_chunk(chunk: List[str], api_key: Optional[str], tool: str) -> Dict[str, str]:
    params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml", "tool": tool}
    if api_key:
        params["api_key"] = api_key
    url = f"{NCBI_EUTILS}/efetch.fcgi?{urlencode(params)}"
    try:
        r = _get_with_retries(url, timeout=20)
        root = ET.fromstring(r.text)
        abstracts: Dict[str, str] = {}

        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            if pmid_el is None or not (pmid_el.text or "").strip():
                continue
            pmid = pmid_el.text.strip()

            abs_texts = ["".join(abs_el.itertext()).strip() for abs_el in article.findall(".//Abstract/AbstractText")]
            abs_texts = [t for t in abs_texts if t]
            if abs_texts:
                abstracts[pmid] = " ".join(abs_texts)

        return abstracts
    except Exception as e:
        print(f"Warning: efetch chunk failed: {e}")
        return {}


def pubmed_efetch_abstracts(pmids: List[str], api_key: Optional[str] = None, max_workers: int = 3) -> Dict[str, str]:
    if not pmids:
        return {}
    abstracts: Dict[str, str] = {}
    chunks = [pmids[i : i + 30] for i in range(0, len(pmids), 30)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_efetch_chunk, chunk, api_key, "literature_search_engine"): chunk for chunk in chunks}
        for future in as_completed(futures):
            try:
                abstracts.update(future.result())
            except Exception as e:
                print(f"Error in efetch chunk: {e}")
    return abstracts


def extract_bridge_snippets(abstracts: List[str], gene: str, biochemicals: List[str], max_snips: int = 2) -> List[str]:
    g = gene.lower()
    terms = [b.lower() for b in (biochemicals or []) if b and b.strip()]
    if not terms:
        return []

    out: List[str] = []
    for abs_text in abstracts:
        text = (abs_text or "").strip()
        low = text.lower()
        if g not in low:
            continue
        for sent in _SENT_SPLIT.split(text):
            s_low = sent.lower()
            if g in s_low and any(t in s_low for t in terms):
                out.append(sent.strip())
                if len(out) >= max_snips:
                    return out
    return out


def produce_pubmed_query_llm(
    *,
    clinical_note: str,
    genes: List[str],
    phenotypes: Optional[List[str]] = None,
    biochemicals: Optional[List[str]] = None,
    model: str = "gpt-4o",
    api_key: str = "OPENAI_API_KEY",
    max_genes: int = 1,
    max_terms: int = 10,
) -> str:
    if not api_key:
        raise ValueError("produce_pubmed_query_llm: api_key is required")

    genes = [g.strip() for g in (genes or []) if g and g.strip()][:max_genes]
    if not genes:
        raise ValueError("produce_pubmed_query_llm: genes is empty")

    client = OpenAI(api_key=api_key)

    system = (
        "You write ONE PubMed (Entrez) query string.\n"
        "You will receive a clinical note and a gene. Make a query that checks PubMed for\n"
        "papers reporting phenotypes, diseases, manifestations with the same gene.\n"
        "If a syndrome is present use only the syndrome and gene.\n"
        "Return ONLY the query string (no quotes, no markdown, no explanation).\n"
    )

    user = {
        "clinical_note": (clinical_note or "")[:3500],
        "genes": genes,
        "phenotypes_hint": (phenotypes or [])[:max_terms],
        "biochemicals_hint": (biochemicals or [])[:max_terms],
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0,
        top_p=1,
        seed=12345,
    )

    q = (resp.choices[0].message.content or "").strip()
    if not q:
        raise ValueError("LLM returned empty query")
    if q.count("(") != q.count(")"):
        raise ValueError(f"LLM query has unbalanced parentheses:\n{q}")
    if len(q) > 1200:
        q = q[:1200]
    if not _ALLOWED.match(q):
        raise ValueError(f"LLM query contains unexpected characters:\n{q}")

    required = "(genetics[sh] OR variant[tiab] OR mutation[tiab]) AND humans[mh]"
    if required not in q:
        q = f"({q}) AND {required}"
    return q

def gene_symbol_from_ranked_item(r: Dict[str, Any]) -> Optional[str]:
    for k in ("symbol", "gene_symbol", "gene", "Gene", "GENE", "name"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            for kk in ("symbol", "gene_symbol", "name", "id"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    return None


def literature_rerank_delta(
    *,
    gene_symbol: str,
    n_hits: int,
    abstracts: List[str],
    biochem_terms: List[str],
    w_hits: float,
    w_bridge: float,
    max_delta: float,
) -> float:
    k = max(1, len(abstracts))
    bridge_abs, bridge_sentence = _count_bridges(abstracts, gene_symbol, biochem_terms)
    norm_base = math.log1p(500)
    hit_strength = min(1.0, math.log1p(max(0, n_hits)) / norm_base)
    bridge_rate = bridge_abs / k
    sentence_rate = min(1.0, bridge_sentence / (2 * k))
    lit_signal = (w_hits * hit_strength) + (w_bridge * (0.6 * bridge_rate + 0.4 * sentence_rate))
    return max(0.0, min(max_delta, lit_signal))


def score_semantic_bridge(abstract: str, gene: str, biochemicals: List[str]) -> float:
    score = 0.0
    abs_l = abstract.lower()
    g_l = gene.lower()
    for b in biochemicals:
        b_l = b.lower()
        if g_l in abs_l and b_l in abs_l:
            score += 1.0
            for sentence in abs_l.split("."):
                if g_l in sentence and b_l in sentence:
                    score += 2.0
                    break
    return score


def parse_phenotypes_input(phenotypes: Union[str, List[str], None]) -> List[str]:
    if phenotypes is None:
        return []
    if isinstance(phenotypes, list):
        return [p.strip() for p in phenotypes if p and p.strip()]
    lines = [ln.strip() for ln in str(phenotypes).splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        parts = ln.split("\t")
        out.append(parts[2].strip() if len(parts) >= 4 else ln)
    seen: set = set()
    dedup: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


@dataclass(frozen=True)
class _ValidatorCfg:
    search_top_n: int
    retmax_per_gene: int
    w_hits: float
    w_bridge: float
    max_delta: float
    use_llm_query: bool


def _read_validator_cfg(blackboard: Optional[Dict[str, Any]], use_llm_query: bool) -> Tuple[_ValidatorCfg, Dict[str, Any]]:
    cfg = get_node_config(blackboard or {}, "validator")
    vc = _ValidatorCfg(
        search_top_n=int(cfg["search_top_n"]),
        retmax_per_gene=int(cfg["retmax_per_gene"]),
        w_hits=float(cfg["w_hits"]),
        w_bridge=float(cfg["w_bridge"]),
        max_delta=float(cfg["max_delta"]),
        use_llm_query=bool(cfg.get("use_llm_query", use_llm_query)),
    )
    return vc, cfg


def process_single_gene(
    gene: str,
    phenos: List[str],
    biochemicals: List[str],
    clinical_note: Optional[str],
    use_llm_query: bool,
    llm_model: str,
    llm_api_key: Optional[str],
    retmax_per_gene: int,
    api_key: Optional[str],
) -> Dict[str, Any]:
    try:
        query = ""
        if use_llm_query and clinical_note and llm_api_key:
            query = produce_pubmed_query_llm(
                clinical_note=clinical_note,
                genes=[gene],
                phenotypes=phenos,
                biochemicals=biochemicals,
                model=llm_model,
                api_key=llm_api_key,
                max_genes=1,
                max_terms=10,
            )

        total_hits, pmids = pubmed_esearch(query, retmax=retmax_per_gene, api_key=api_key)
        meta_map = pubmed_esummary_meta(pmids, api_key=api_key)
        abstract_map = pubmed_efetch_abstracts(pmids, api_key=api_key)

        citations: List[Dict[str, Any]] = []
        for pmid in pmids[:3]:
            m = meta_map.get(str(pmid), {"pmid": str(pmid)})
            citations.append(
                {
                    "pmid": m.get("pmid"),
                    "doi": m.get("doi"),
                    "pmcid": m.get("pmcid"),
                    "title": m.get("title"),
                    "journal": m.get("journal"),
                    "year": m.get("year"),
                }
            )

        abs_texts = list(abstract_map.values())
        snips = extract_bridge_snippets(abs_texts, gene, biochemicals, max_snips=2)
        bridge_abs, bridge_sentence = _count_bridges(abs_texts, gene, biochemicals)

        raw_lit_score = 0.0
        for abs_text in abs_texts:
            raw_lit_score += 0.5
            raw_lit_score += score_semantic_bridge(abs_text, gene, biochemicals)

        return {
            "gene": gene,
            "data": {
                "n_hits": int(total_hits),
                "n_retrieved": len(pmids),
                "bridge_abs": int(bridge_abs),
                "bridge_sentence": int(bridge_sentence),
                "raw_lit_score": round(raw_lit_score, 2),
                "top_pmids": pmids[:3],
                "citations": citations,
                "bridge_snippets": snips,
                "pubmed_query": query,
                "abstracts": abs_texts,
            },
        }
    except Exception as e:
        print(f"Error processing gene {gene}: {e}")
        return {
            "gene": gene,
            "data": {
                "error": str(e),
                "n_hits": 0,
                "n_retrieved": 0,
                "bridge_abs": 0,
                "bridge_sentence": 0,
                "raw_lit_score": 0.0,
                "top_pmids": [],
                "citations": [],
                "bridge_snippets": [],
                "abstracts": [],
                "pubmed_query": "",
            },
        }


@mcp.tool()
def search_literature_nuanced(
    genes: List[str],
    clinical_note: Optional[str] = None,
    phenotypes: Union[str, List[str]] = None,
    biochemicals: Optional[List[str]] = None,
    syndromes: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    additional_context: Optional[List[str]] = None,
    query_terms: Optional[Dict[str, Any]] = None,
    noise_terms: Optional[List[str]] = None,
    use_llm_query: bool = True,
    llm_model: str = "gpt-4o",
    llm_api_key: str = "OPENAI_API_KEY",
    ranked_genes: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    sleep_s: float = 0.0,
    max_workers: int = 5,
    blackboard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vc, cfg_raw = _read_validator_cfg(blackboard, use_llm_query=use_llm_query)

    phenos = parse_phenotypes_input(phenotypes)
    biochemicals = biochemicals or []
    syndromes = syndromes or []
    keywords = keywords or []
    additional_context = additional_context or []
    noise_terms = noise_terms or []

    global RATE_LIMITER
    if api_key:
        RATE_LIMITER = RateLimiter(max_calls_per_second=8)

    if sleep_s and sleep_s > 0:
        time.sleep(float(sleep_s))

    results: Dict[str, Any] = {
        "genes": {},
        "query_meta": {
            "has_clinical_note": bool(clinical_note),
            "use_llm_query": bool(vc.use_llm_query and clinical_note and llm_api_key),
            "llm_model": llm_model if (vc.use_llm_query and clinical_note and llm_api_key) else None,
            "n_phenotypes": len(phenos),
            "n_biochemicals": len(biochemicals),
            "retmax_per_gene": vc.retmax_per_gene,
            "config_used": cfg_raw,
        },
    }

    valid_genes = [g.strip() for g in (genes or []) if g and g.strip()]
    if vc.search_top_n > 0:
        valid_genes = valid_genes[: int(vc.search_top_n)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_gene,
                gene,
                phenos,
                biochemicals,
                clinical_note,  # ← correct position
                vc.use_llm_query,
                llm_model,
                llm_api_key,
                vc.retmax_per_gene,
                api_key,
            ): gene
            for gene in valid_genes
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and "gene" in result:
                    results["genes"][result["gene"]] = result["data"]
            except Exception as e:
                gene = futures[future]
                print(f"Failed to process gene {gene}: {e}")

    if ranked_genes:
        reranked: List[Dict[str, Any]] = []
        processed = set(valid_genes)

        for r in ranked_genes:
            symbol = gene_symbol_from_ranked_item(r)
            if not symbol or symbol not in processed:
                continue

            lit_data = results["genes"].get(symbol, {})
            hpo_score = float(r.get("score", 0.0))
            n_hits = int(lit_data.get("n_hits", 0) or 0)
            abstract_texts = lit_data.get("abstracts", []) or []

            delta = literature_rerank_delta(
                gene_symbol=symbol,
                n_hits=n_hits,
                abstracts=abstract_texts,
                biochem_terms=biochemicals,
                w_hits=vc.w_hits,
                w_bridge=vc.w_bridge,
                max_delta=vc.max_delta,
            )

            entry = dict(r)
            entry["score"] = round(hpo_score + delta, 4)
            entry["evidence_summary"] = {
                "lit_total_hits": n_hits,
                "lit_retrieved": int(lit_data.get("n_retrieved", 0) or 0),
                "bridge_abs": int(lit_data.get("bridge_abs", 0) or 0),
                "bridge_sentence": int(lit_data.get("bridge_sentence", 0) or 0),
                "top_pmids": lit_data.get("top_pmids", []),
                "delta_added": round(delta, 4),
                "citations": (lit_data.get("citations", []) or [])[:3],
                "bridge_snippets": (lit_data.get("bridge_snippets", []) or [])[:2],
                "pubmed_query": lit_data.get("pubmed_query", ""),
            }
            reranked.append(entry)

        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return {"reranked_results": reranked, "meta": results}

    return results


if __name__ == "__main__":
    mcp.run()