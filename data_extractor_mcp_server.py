from __future__ import annotations
import os
from openai import OpenAI
import ast
import json
import os
from typing import Any, Dict, List

import nltk
from mcp.server.fastmcp import FastMCP
from api import annotate_text

mcp = FastMCP("phenotype_annotator")

PHENO_CONTEXT_REWRITE_SYSTEM = r"""
Rewrite a clinical note to improve phenotype extraction (e.g., PhenoBERT) and downstream gene prioritization.

Return ONLY valid JSON with:
- rewritten_note: string (coherent narrative, not a list)
- quotes_used: list of objects {phrase, supporting_text}

Rules:
- Do NOT add new facts (symptoms/diagnoses/tests/timing/certainty).
- Preserve key phenotypes, explicit syndrome/disease suspicions, and explicit negations.
- Prefer phenotype-centric phrasing when supported by the text (HPO-like wording allowed if faithful).
- If original is vague, keep it vague.
- For ovarian cancer use "ovarian carcinoma"; for breast cancer use "breast carcinoma".

quotes_used:
- For each major rewrite, include:
  - phrase: short phrase you wrote
  - supporting_text: exact quote (<=25 words) from the original supporting it.

Output must be valid JSON only.
"""

NORMALIZE_SYSTEM = r"""
Extract structured search inputs and prioritization inputs from a clinical note.

Return ONLY valid JSON with:
- biochemical: list[string] (analytes/biomarkers only; remove qualifiers like elevated/low/level)
- negatives: list[string] (ONLY explicit negations)
- syndromes: list[string] (ONLY explicitly mentioned)
- keywords: list[string] (short, useful search terms)
- additional_context: list[string] (short, useful context terms)
- query_terms: object with:
    - phenotypes_high_value: list[string]
    - phenotypes_generic: list[string]
    - mechanisms: list[string] (ONLY explicit)
    - locus_terms: list[string] (ONLY explicit; locus is chromosomal location, not gene)
    - syndromes: list[string]
    - biomarkers: list[string]
    - drugs: list[string]
    - pathogens: list[string]
    - procedures_dropped: list[string]
- noise_terms: list[string] (generic/irrelevant items to ignore)

Output must be valid JSON only.
"""

LOCUS_SIGNAL_SYSTEM = r"""
Extract locus-relevant signals from a clinical note for downstream gene prioritization.

Return ONLY valid JSON with:
- suspected_syndromes: list of objects {term, supporting_text}
- locus_terms: list of objects {term, supporting_text}
- mechanism_terms: list of objects {term, supporting_text}
- trigger_phrases: list of objects {phrase, supporting_text}

Rules:
1) suspected_syndromes MUST be anchored by an exact quote (<=25 words).
2) You may add AT MOST 2 "pattern syndromes" if strongly supported by quotes, e.g.:
   - "familial progressive ataxia"
   These are NOT diagnoses and must be justified by supporting_text.
3) locus_terms ONLY if a chromosomal/cytogenetic locus is explicitly present in the text (never genes).
4) mechanism_terms ONLY if explicitly stated (never inferred).
5) Keep lists short (max 6 each).
"""


def _ensure_nltk_resource(path: str, download_name: str) -> None:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def ensure_nltk_ready() -> None:
    _ensure_nltk_resource("tokenizers/punkt", "punkt")
    _ensure_nltk_resource("corpora/stopwords", "stopwords")
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)


def _dedup_clean_list(xs: Any, max_n: int) -> List[str]:
    if xs is None:
        return []
    if not isinstance(xs, list):
        if isinstance(xs, (tuple, set)):
            xs = list(xs)
        else:
            return []
    out: List[str] = []
    seen: set[str] = set()
    for x in xs:
        if x is None:
            continue
        s = (x if isinstance(x, str) else str(x)).strip()
        s = " ".join(s.split())
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def _coerce_term_list(xs: Any, max_n: int) -> List[str]:
    if xs is None or not isinstance(xs, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for x in xs:
        term = ""
        if isinstance(x, dict):
            term = str(x.get("term", "")).strip()
        elif isinstance(x, str):
            s = x.strip()
            if s.startswith("{") and "term" in s:
                try:
                    obj = ast.literal_eval(s)
                    if isinstance(obj, dict) and "term" in obj:
                        term = str(obj.get("term", "")).strip()
                    else:
                        term = s
                except Exception:
                    term = s
            else:
                term = s
        else:
            continue
        term = " ".join(term.split())
        if not term:
            continue
        k = term.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(term)
        if len(out) >= max_n:
            break
    return out


def _openai_json_call(system_prompt: str, user_text: str) -> Dict[str, Any]:

    api_key = "OPENAI_API_KEY"
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def phenotype_rewrite_with_llm(text: str) -> Dict[str, Any]:
    data = _openai_json_call(PHENO_CONTEXT_REWRITE_SYSTEM, text)
    if "rewritten_note" not in data:
        raise RuntimeError("Missing rewritten_note in rewrite output.")
    if not isinstance(data.get("quotes_used"), list):
        data["quotes_used"] = []
    return data


def normalize_clinical_note_llm(text: str) -> Dict[str, Any]:
    data = _openai_json_call(NORMALIZE_SYSTEM, text)

    data.setdefault("biochemical", [])
    data.setdefault("negatives", [])
    data.setdefault("syndromes", [])
    data.setdefault("keywords", [])
    data.setdefault("additional_context", [])
    data.setdefault("query_terms", {})
    data.setdefault("noise_terms", [])

    data["biochemical"] = _dedup_clean_list(data["biochemical"], 12)
    data["negatives"] = _dedup_clean_list(data["negatives"], 12)
    data["syndromes"] = _coerce_term_list(data["syndromes"], 12)
    data["keywords"] = _dedup_clean_list(data["keywords"], 12)
    data["additional_context"] = _dedup_clean_list(data["additional_context"], 12)
    data["noise_terms"] = _dedup_clean_list(data["noise_terms"], 20)

    qt = data["query_terms"] if isinstance(data.get("query_terms"), dict) else {}

    def qget(k: str) -> List[str]:
        return _dedup_clean_list(qt.get(k, []), 12)

    syns = qt.get("syndromes", [])
    syndromes = _coerce_term_list(syns, 12)

    data["query_terms"] = {
        "phenotypes_high_value": qget("phenotypes_high_value"),
        "phenotypes_generic": qget("phenotypes_generic"),
        "mechanisms": qget("mechanisms"),
        "locus_terms": qget("locus_terms"),
        "syndromes": syndromes,
        "biomarkers": qget("biomarkers"),
        "drugs": qget("drugs"),
        "pathogens": qget("pathogens"),
        "procedures_dropped": qget("procedures_dropped"),
    }
    return data


def extract_locus_signals_llm(text: str) -> Dict[str, Any]:
    data = _openai_json_call(LOCUS_SIGNAL_SYSTEM, text)

    data.setdefault("suspected_syndromes", [])
    data.setdefault("locus_terms", [])
    data.setdefault("mechanism_terms", [])
    data.setdefault("trigger_phrases", [])

    data["suspected_syndromes"] = _coerce_term_list(data["suspected_syndromes"], 10)
    data["locus_terms"] = _dedup_clean_list(data["locus_terms"], 12)
    data["mechanism_terms"] = _dedup_clean_list(data["mechanism_terms"], 12)

    tp = data.get("trigger_phrases")
    if not isinstance(tp, list):
        tp = []
    safe_tp = []
    for it in tp[:25]:
        if not isinstance(it, dict):
            continue
        phrase = str(it.get("phrase", "")).strip()
        support = str(it.get("supporting_text", "")).strip()
        if phrase and support:
            safe_tp.append({"phrase": phrase, "supporting_text": support[:180]})
    data["trigger_phrases"] = safe_tp
    return data


def merge_locus_into_normalization(normalization: Dict[str, Any], locus: Dict[str, Any]) -> Dict[str, Any]:
    normalization = dict(normalization or {})
    qt = normalization.get("query_terms") if isinstance(normalization.get("query_terms"), dict) else {}

    merged_syndromes = _dedup_clean_list(
        (normalization.get("syndromes") or []) + (locus.get("suspected_syndromes") or []),
        12,
    )
    merged_mech = _dedup_clean_list((qt.get("mechanisms", []) or []) + (locus.get("mechanism_terms") or []), 12)
    merged_loci = _dedup_clean_list((qt.get("locus_terms", []) or []) + (locus.get("locus_terms") or []), 12)
    merged_qt_syns = _dedup_clean_list((qt.get("syndromes", []) or []) + merged_syndromes, 12)

    normalization["syndromes"] = merged_syndromes
    normalization["query_terms"] = dict(qt)
    normalization["query_terms"]["mechanisms"] = merged_mech
    normalization["query_terms"]["locus_terms"] = merged_loci
    normalization["query_terms"]["syndromes"] = merged_qt_syns
    return normalization


def parse_annotate_text_output(result: Any) -> Dict[str, List[str]]:
    text = result if isinstance(result, str) else str(result)
    lines = text.strip().splitlines() if text.strip() else []
    hpo_ids: List[str] = []
    hpo_names: List[str] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 5:
            parts = line.split()
        if len(parts) >= 4:
            name = parts[2].strip()
            hpo = parts[3].strip()
            if hpo.startswith("HP:"):
                hpo_ids.append(hpo)
                hpo_names.append(name)
    seen = set()
    out_ids, out_names = [], []
    for hid, nm in zip(hpo_ids, hpo_names):
        if hid in seen:
            continue
        seen.add(hid)
        out_ids.append(hid)
        out_names.append(nm)
    return {"hpo_ids": out_ids, "hpo_names": out_names}


@mcp.tool()
def annotate_case(
    text: str,
    *,
    ensure_nltk: bool = True,
    normalize_with_llm: bool = True,
    rewrite_to_phenotypes: bool = True,
) -> Any:
    if ensure_nltk:
        ensure_nltk_ready()

    original_text = text
    text_for_hpo = original_text

    normalization: Dict[str, Any] = {}
    if normalize_with_llm:
        normalization = normalize_clinical_note_llm(original_text)

    rewrite: Dict[str, Any] = {}
    if rewrite_to_phenotypes:
        rewrite = phenotype_rewrite_with_llm(original_text)
        text_for_hpo = rewrite["rewritten_note"]

    parsed = parse_annotate_text_output(annotate_text(text_for_hpo))

    try:
        locus_signals = extract_locus_signals_llm(original_text)
    except Exception as e:
        locus_signals = {
            "error": str(e),
            "suspected_syndromes": [],
            "locus_terms": [],
            "mechanism_terms": [],
            "trigger_phrases": [],
        }

    if normalize_with_llm:
        normalization = merge_locus_into_normalization(normalization, locus_signals)

    payload: Dict[str, Any] = {
        "hpo_ids": parsed["hpo_ids"],
        "hpo_names": parsed["hpo_names"],
        "text_used_for_hpo": text_for_hpo,
        "rewrite": rewrite,
        "locus_signals": locus_signals,
    }
    if normalize_with_llm:
        payload["normalization"] = normalization
    return payload


if __name__ == "__main__":
    mcp.run()