# 🧬 GeneScout
**GeneScout: A Traceable and Interpretable Multi-Agent System for Phenotype- and Biochemistry-Driven Gene Ranking**

GeneScout is a modular multi-agent pipeline that ranks candidate genes by integrating phenotype evidence (via PhenoBERT) with biochemical, locus, and literature reasoning agents to produce traceable and interpretable gene prioritization.

---

## 📦 Dependencies (Required Repos)

GeneScout depends on **two repositories**:

### 1) 🧠 GeneScout repo  
Clone this repository (your GeneScout code):

```bash
git clone https://github.com/ChatMED/GeneScout.git
```
### 2) 🔬 PhenoBERT repo

PhenoBERT is required for phenotype extraction and downstream phenotype-driven gene ranking:

```bash
git clone https://github.com/EclipseCN/PhenoBERT.git
```
### 📥 Install requirements

Install dependencies from both repositories:

```bash
pip install -U pip
cd GeneScout
pip install -r requirements.txt
cd ../PhenoBERT
pip install -r requirements.txt
```
## ⚠️ IMPORTANT — Required Folder Placement

Due to a **PhenoBERT internal path constraint**, all GeneScout runtime files must be placed inside:

```text
PhenoBERT/phenobert/utils/
```
## 🗂️ Expected Structure

After copying, the structure should look like:

```text
PhenoBERT/
  phenobert/
    utils/
      data/
      final_reports/
      nodes/
      utils/
      biochemical_mcp_server.py
      data_extractor_mcp_server.py
      gene_extractor_mcp_server.py
      literature_search_mcp_server.py
      locus_mcp_server.py
      reasoning_controler_mcp_server.py
      run_genescout.py
      shared_config.py
```
## ▶️ Running GeneScout

Navigate to the required execution directory:

```bash
cd PhenoBERT/phenobert/utils
```
First add the OPENAI_API_KEY where needed. In run_genescout.py you can change the case description and run the pipeline:

```bash
python run_genescout.py
```
## 🧠 System Components

GeneScout includes the following MCP agents:

- 🧬 **Clinical Phenotype Extractor**  
- 🧪 **HPO-Based Gene Ranker**  
- 📍 **Genomic Locus Integrator**  
- 🧫 **Biochemical Evidence Refiner**  
- 📚 **Literature-Based Validation Agent**  
- 🤖 **Reasoning Lead Agent (LLM orchestrator)**  

After final ranking, the results are processed by the 🧾 **HL7 FHIR Exporter and Clinician Report Generator**.
