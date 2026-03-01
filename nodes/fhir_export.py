from __future__ import annotations

from typing import Any, Dict

from utils.mcp_parsing import parse_mcp_result
from utils.mcp_registry import ToolRegistry
from utils.state_types import DiagnosticState


async def fhir_export_node(
    state: DiagnosticState,
    registry: ToolRegistry,
    reports_out_dir: str,
    summary_model: str,
) -> Dict[str, Any]:
    tool = await registry.get_tool("export_gene_prioritization_fhir_bundle")

    final_genes = state.get("merged_genes") or []
    bb = state.get("blackboard") or {}
    bb_for_export = {k: v for k, v in bb.items() if k != "next_actions"}

    hypotheses = bb.get("hypotheses") or []
    top_dx = hypotheses[0].get("dx") if hypotheses and isinstance(hypotheses[0], dict) else None

    payload = {
        "clinical_note": state["clinical_note"],
        "hpo_ids": state.get("hpo_ids", []) or [],
        "hpo_names": state.get("hpo_names", []) or [],
        "biochemical_findings": state.get("biochemical_findings", []) or [],
        "negatives": state.get("negatives", []) or [],
        "candidate_genes": final_genes,
        "pubmed_evidence": state.get("pubmed_evidence", {}) or {},
        "golden_reasoning": state.get("golden_reasoning", {}) or {},
        "biochem_reasoning": state.get("biochem_reasoning", {}) or {},
        "blackboard": bb_for_export,
        "suspected_condition_text": top_dx,
        "top_n": 20,
        "include_llm_summary": True,
        "summary_model": summary_model,
        "out_dir": reports_out_dir,
        "file_stem": "case_report",
        "annotate_payload": state.get("annotate_payload", {}) or {},
        "locus_evidence": state.get("locus_evidence", {}) or {},
        "locus_reranked_genes": state.get("locus_reranked_genes", []) or [],
        "literature_evidence_raw": {
            "pubmed_evidence": state.get("pubmed_evidence", {}) or {},
            "literature_reranked_genes": state.get("literature_reranked_genes", []) or [],
        },
        "steps_completed": state.get("steps_completed", []) or [],
    }

    result = await tool.ainvoke(payload)
    data = parse_mcp_result(result)

    return {
        "fhir_bundle_path": data.get("bundle_path", "") or "",
        "fhir_summary_path": data.get("summary_path", "") or "",
        "fhir_summary": data.get("summary", {}) or {},
        "steps_completed": ["fhir_export"],
    }