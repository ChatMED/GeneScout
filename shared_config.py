from typing import Any, Dict, Optional
import copy

DEFAULT_BLACKBOARD: Dict[str, Any] = {
    "case_summary": "",
    "signals": {
        "has_syndrome": False,
        "has_biochem": False,
        "n_hpo": 0,
    },
    "hypotheses": [],
    "weights": {
        "hpo":        0.50,
        "locus":      0.30,
        "biochem":    0.10,
        "literature": 0.10,
    },
    "node_config": {
        "locus_boost": {
            "max_diseases":       5,
            "ot_gene_cap":        400,
            "max_delta":          0.38,
            "inject_new":         True,
            "inject_base_score":  0.12,
            "min_inject_llm_score": 0.62,
            "use_llm":            True,
            "llm_weight":         0.25,
        },
        "geneticist": {
            "top_n":              200,
            "restrict":           False,
            "similarity_method":  "resnik",
            "similarity_combine": "funSimMax",
            "include_reasoning":  False,
            "reasoning_top_n":    25,
        },
        "biochemist": {
            "cap_pathways":           25,
            "eps":                    0.08,
            "discovery_base":         0.10,
            "w_direct_rxn":           1.00,
            "w_uniprot":              0.90,
            "w_gwas_prox":            0.80,
            "w_pathway":              0.35,
            "include_reasoning":      True,
            "reasoning_top_n":        25,
            "inject_new":             True,
        },
        "validator": {
            "search_top_n":   300,
            "retmax_per_gene": 10,
            "w_hits":          0.15,
            "w_bridge":        0.25,
            "max_delta":       0.50,
            "use_llm_query":   True,
        },
    },
    "merge_policy": {
        "normalization": "score_based",
        "merge":         "weighted_sum",
        "top_n":         250,
        "locus_quality_min": 0.4,
        "locus_quality_max": 1.0,
        "caps": {
            "inject_locus":  200,
            "inject_biochem": 30,
            "use_literature_only_if_bridge": True,
        },
    },
    "next_actions": [],
}

def new_blackboard() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_BLACKBOARD)

def get_node_config(blackboard: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    defaults = DEFAULT_BLACKBOARD.get("node_config", {}).get(node_name, {})
    bb_node  = (blackboard.get("node_config") or {}).get(node_name) or {}
    return {**defaults, **bb_node}

def get_merge_policy(blackboard: Dict[str, Any]) -> Dict[str, Any]:
    defaults = DEFAULT_BLACKBOARD["merge_policy"]
    bb_mp    = blackboard.get("merge_policy") or {}
    return {**defaults, **bb_mp}

def get_weights(blackboard: Dict[str, Any]) -> Dict[str, float]:
    defaults = DEFAULT_BLACKBOARD["weights"]
    bb_w     = blackboard.get("weights") or {}
    w = {k: float(bb_w.get(k, defaults[k])) for k in defaults}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}