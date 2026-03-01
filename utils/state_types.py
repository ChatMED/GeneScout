from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from utils.shared_config import DEFAULT_BLACKBOARD


class DiagnosticState(TypedDict):
    clinical_note: str

    annotate_payload: Dict[str, Any]
    hpo_ids: List[str]
    hpo_names: List[str]
    biochemical_findings: List[str]
    negatives: List[str]
    syndromes: List[str]
    keywords: List[str]
    additional_context: List[str]
    query_terms: Dict[str, Any]
    noise_terms: List[str]

    candidate_genes: List[Dict[str, Any]]
    biochem_reranked_genes: List[Dict[str, Any]]
    biochem_reasoning: Dict[str, Any]
    golden_genes: List[Dict[str, Any]]
    golden_reasoning: Dict[str, Any]

    pubmed_evidence: Dict[str, Any]
    literature_reranked_genes: List[Dict[str, Any]]

    locus_reranked_genes: List[Dict[str, Any]]
    locus_raw_genes: List[Dict[str, Any]]
    locus_evidence: Dict[str, Any]

    fhir_bundle_path: str
    fhir_summary_path: str
    fhir_summary: Dict[str, Any]

    steps_completed: Annotated[List[str], operator.add]
    blackboard: Dict[str, Any]
    merged_genes: List[Dict[str, Any]]


def build_initial_state(case_text: str) -> DiagnosticState:
    return {
        "clinical_note": case_text,

        "annotate_payload": {},
        "hpo_ids": [],
        "hpo_names": [],
        "biochemical_findings": [],
        "negatives": [],
        "syndromes": [],
        "keywords": [],
        "additional_context": [],
        "query_terms": {},
        "noise_terms": [],

        "candidate_genes": [],
        "biochem_reranked_genes": [],
        "biochem_reasoning": {},
        "golden_genes": [],
        "golden_reasoning": {},

        "pubmed_evidence": {},
        "literature_reranked_genes": [],

        "locus_reranked_genes": [],
        "locus_raw_genes": [],
        "locus_evidence": {},

        "fhir_bundle_path": "",
        "fhir_summary_path": "",
        "fhir_summary": {},

        "steps_completed": [],
        "blackboard": DEFAULT_BLACKBOARD.copy(),
        "merged_genes": [],
    }