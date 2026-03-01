from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ModelConfig:
    controller_model: str = "gpt-5-mini"
    validator_llm_model: str = "gpt-4o-mini"
    geneticist_reasoning_model: str = "gpt-4o-mini"
    fhir_summary_model: str = "gpt-4o-mini"


@dataclass(frozen=True)
class OutputConfig:
    reports_out_dir: str
    file_stem: str = "case_report"


def default_mcp_servers() -> Dict[str, Dict[str, Any]]:
    return {
        "phenotype_annotator": {
            "command": "python",
            "args": ["data_extractor_mcp_server.py"],
            "transport": "stdio",
        },
        "gene_extraction": {
            "command": "python",
            "args": ["gene_extractor_mcp_server.py"],
            "transport": "stdio",
        },
        "biochemical_refiner": {
            "command": "python",
            "args": ["biochemical_mcp_server.py"],
            "transport": "stdio",
        },
        "literature_search_engine": {
            "command": "python",
            "args": ["literature_search_mcp_server.py"],
            "transport": "stdio",
        },
        "locus_boost": {
            "command": "python",
            "args": ["locus_mcp_server.py"],
            "transport": "stdio",
        },
        "fhir_exporter": {
            "command": "python",
            "args": ["fhir_mcp_server.py"],
            "transport": "stdio",
        },
        "reasoning_controller": {
            "command": "python",
            "args": ["reasoning_controler_mcp_server.py"],
            "transport": "stdio",
        },
    }