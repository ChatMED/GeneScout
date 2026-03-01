from __future__ import annotations

from typing import Callable

from langgraph.graph import StateGraph, END

from utils.mcp_registry import ToolRegistry
from utils.settings import ModelConfig, OutputConfig
from utils.state_types import DiagnosticState
from utils.nodes.registrar import registrar_node
from utils.nodes.locus_boost import locus_boost_node
from utils.nodes.geneticist import geneticist_node
from utils.nodes.biochemist import biochemist_node
from utils.nodes.validator import validator_node
from utils.nodes.fhir_export import fhir_export_node
from utils.nodes.controller import controller_node


def _bind(fn, *args, **kwargs) -> Callable:
    async def wrapped(state):
        return await fn(state, *args, **kwargs)
    return wrapped


def should_run_locus(state: DiagnosticState) -> str:
    completed = state.get("steps_completed") or []
    if "locus_boost" in completed:
        return "skip_locus"

    bb = state.get("blackboard") or {}
    next_actions = bb.get("next_actions") or []
    suggested = [a.get("tool", "") for a in next_actions]

    if "locus_boost" in suggested:
        return "run_locus"
    if any(t in suggested for t in ["geneticist", "skip_locus"]):
        return "skip_locus"

    annotate = state.get("annotate_payload") or {}
    has_syndromes = bool(state.get("syndromes") or annotate.get("syndromes"))
    has_diseases = bool(annotate.get("diseases") or annotate.get("suspected_conditions"))
    has_locus_terms = bool((annotate.get("locus_signals") or {}).get("locus_terms"))

    if has_syndromes or has_diseases or has_locus_terms:
        return "run_locus"
    return "skip_locus"


def should_run_biochem(state: DiagnosticState) -> str:
    bb = state.get("blackboard") or {}
    next_actions = bb.get("next_actions") or []
    suggested = [a.get("tool", "") for a in next_actions]

    if "biochemist" in suggested:
        return "biochemist"
    if "validator" in suggested:
        return "validator"

    biochem = state.get("biochemical_findings") or []
    if biochem and any(str(f).strip() for f in biochem):
        return "biochemist"
    return "validator"


def build_workflow_app(
    *,
    registry: ToolRegistry,
    model_cfg: ModelConfig,
    out_cfg: OutputConfig,
):
    workflow = StateGraph(DiagnosticState)

    workflow.add_node("registrar", _bind(registrar_node, registry))
    workflow.add_node("locus_boost", _bind(locus_boost_node, registry, model_cfg.controller_model))
    workflow.add_node("geneticist", _bind(geneticist_node, registry, model_cfg.geneticist_reasoning_model))
    workflow.add_node("biochemist", _bind(biochemist_node, registry))
    workflow.add_node("validator", _bind(validator_node, registry, model_cfg.validator_llm_model))
    workflow.add_node("fhir_export", _bind(fhir_export_node, registry, out_cfg.reports_out_dir, model_cfg.fhir_summary_model))

    workflow.add_node("ctrl_after_registrar", _bind(controller_node, registry, "registrar", model_cfg.controller_model))
    workflow.add_node("ctrl_after_locus", _bind(controller_node, registry, "locus_boost", model_cfg.controller_model))
    workflow.add_node("ctrl_after_geneticist", _bind(controller_node, registry, "geneticist", model_cfg.controller_model))
    workflow.add_node("ctrl_after_biochemist", _bind(controller_node, registry, "biochemist", model_cfg.controller_model))
    workflow.add_node("ctrl_after_validator", _bind(controller_node, registry, "validator", model_cfg.controller_model))

    workflow.set_entry_point("registrar")

    workflow.add_edge("registrar", "ctrl_after_registrar")
    workflow.add_conditional_edges(
        "ctrl_after_registrar",
        should_run_locus,
        {"run_locus": "locus_boost", "skip_locus": "geneticist"},
    )

    workflow.add_edge("locus_boost", "ctrl_after_locus")
    workflow.add_edge("ctrl_after_locus", "geneticist")

    workflow.add_edge("geneticist", "ctrl_after_geneticist")
    workflow.add_conditional_edges(
        "ctrl_after_geneticist",
        should_run_biochem,
        {"biochemist": "biochemist", "validator": "validator"},
    )

    workflow.add_edge("biochemist", "ctrl_after_biochemist")
    workflow.add_edge("ctrl_after_biochemist", "validator")

    workflow.add_edge("validator", "ctrl_after_validator")
    workflow.add_edge("ctrl_after_validator", "fhir_export")

    workflow.add_edge("fhir_export", END)

    return workflow.compile()