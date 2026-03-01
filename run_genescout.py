from __future__ import annotations
import os
print(os.getenv("OPENAI_API_KEY"))
import asyncio
import warnings
from pprint import pprint
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient

from utils.mcp_registry import ToolRegistry
from utils.settings import ModelConfig, OutputConfig, default_mcp_servers
from utils.state_types import build_initial_state
from utils.workflow_graph import build_workflow_app
from utils.node_helpers import safe_topk_merged, gene_symbol, gene_score

warnings.filterwarnings("ignore", message="Input sequence provided is already iteration in string format No operation performed")
warnings.filterwarnings("ignore", message="Downcasting object dtype arrays")


async def run_clinical_note(note_text: str) -> Dict[str, Any]:
    model_cfg = ModelConfig()
    out_cfg = OutputConfig(
        reports_out_dir="../final_reports",
        file_stem="case_report",
    )

    mcp_client = MultiServerMCPClient(default_mcp_servers())
    registry = ToolRegistry(mcp_client)

    app = build_workflow_app(registry=registry, model_cfg=model_cfg, out_cfg=out_cfg)

    final_state = await app.ainvoke(build_initial_state(note_text))
    return final_state


async def main():
    # Put your clinical note here (or load from file)
    clinical_note = """A 13-year-old adolescent boy presented with the chief complaint of recurrent headache for 1 month. One month prior to the consult, the patient developed headache without an obvious trigger, mainly concentrated in the left temporo-occipital region, mild to moderate (2–4 points, total score of 10 points) (11), usually characterized by dull pain, accompanied by vascular pulsation, and occasionally tingling, lasting from several seconds to several minutes, and had four to five attacks in the preceding month. There were no prodromal or accompanying symptoms of aura, photophobia, phonophobia, nausea, or change in level of consciousness. No dizziness or any other neurological symptoms were noted. The patient had no history of hypertension, migraines, sinusitis, or the use of drugs that could cause headaches as an adverse effect. The patient was transferred to our hospital due to headache of unknown cause.

Systemic examination revealed a light brown plaque with clear boundaries of different sizes scattered on the chest and abdomen, back, shoulder, and left upper arm, not protruding from the skin surface, and with a diameter of 0.3–8.0 cm (Figure 1). The patient's mother had similar skin manifestations, but did not have headaches. The laboratory examination results were unremarkable. We performed a lumbar puncture and cerebrospinal fluid examination, which showed no significant abnormalities. Both autoimmune encephalopathy antibodies and central nervous system (CNS) demyelinating antibodies were negative. MRI of the head revealed multiple abnormal signal foci in the bilateral basal ganglia, thalamus, and pons (Figure 2)."""

    final_state = await run_clinical_note(clinical_note)

    print("\n=== Steps completed ===")
    print(" -> ".join(final_state.get("steps_completed", []) or []))

    print("\n=== Top 20 merged genes ===")
    top20 = safe_topk_merged(final_state, k=20)
    for i, g in enumerate(top20, 1):
        print(f"{i:02d}. {gene_symbol(g):<12}  score={gene_score(g)}")

    print("\n=== FHIR outputs ===")
    print("Bundle:", final_state.get("fhir_bundle_path"))
    print("Summary:", final_state.get("fhir_summary_path"))

    print("\n=== Blackboard hypotheses (if any) ===")
    bb = final_state.get("blackboard") or {}
    hyps = bb.get("hypotheses") or []
    if hyps:
        pprint(hyps[:3])
    else:
        print("(none)")


if __name__ == "__main__":
    asyncio.run(main())