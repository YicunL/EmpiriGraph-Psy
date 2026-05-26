# EmpiriGraph-Psy

Code and data for the paper:

> **EmpiriGraph-Psy: A Dataset and LLM Pipeline for Extracting Empirical Relation Graphs from Psychology Abstracts**  
> Anonymous ACL submission

---

## Overview

This repository contains:

1. **Dataset** — 210 psychology abstracts annotated with normalized variables, hierarchical construct–variable links, empirical relation types (associational, mechanistic, moderational, hierarchical), and validation states (validated / null / hypothesized).
2. **Pipeline** — A five-stage LLM extraction pipeline (variable extraction → normalization & hierarchy → evidence selection → relation extraction → edge validation).
3. **Evaluation** — A structure-first graph evaluation framework based on maximum common subgraph alignment.

---

## Repository Structure

```
EmpiriGraph-Psy/
├── data/
│   ├── raw/
│   │   ├── EmpiriGraph-Psy_gold_annotation.json   # Gold annotation (210 abstracts)
│   │   └── annotation_csv/
│   │       ├── coder_A_r.csv                              # Coder A annotations (50 abstracts)
│   │       └── coder_B_r.csv                              # Coder B annotations (50 abstracts)
│   └── output/
│       ├── pipeline_gpt54_gpt52_210/      # Best model: GPT-5.4 + GPT-5.2, 210 abstracts
│       ├── pipeline_gpt54_165/            # GPT-5.4 all steps, 165 abstracts
│       ├── pipeline_gpt52_165/            # GPT-5.2 all steps, 165 abstracts
│       ├── pipeline_gpt4o_165/            # GPT-4o all steps, 165 abstracts
│       ├── pipeline_claude_sonnet46_165/  # Claude Sonnet 4.6, 165 abstracts
│       ├── pipeline_claude_opus47_165/    # Claude Opus 4.7, 165 abstracts
│       ├── pipeline_gemini3flash_165/      # Gemini 3 Flash, 165 abstracts
│       ├── pipeline_deepseek_v4pro_210/   # DeepSeek V4 Pro, 210 abstracts
│       ├── graph_eval/                    # Graph evaluation outputs (F1 per model)
│       ├── intercoder/                    # Inter-annotator agreement outputs
│       └── single_step_gpt54_210_batch/   # Direct-prompting baseline batch files
│
└── notebooks/
    ├── graph_eval_functions.py                  # Core graph parsing, preprocessing, MCS eval
    ├── extraction_pipeline_v10.ipynb            # Pipeline prompt definitions (source of truth)
    ├── pipeline_gpt54_gpt52.ipynb               # Run pipeline: GPT-5.4 + GPT-5.2
    ├── pipeline_single_step_gpt54.ipynb         # Run direct-prompting baseline
    ├── graph_eval.ipynb                         # Graph structural evaluation (all conditions)
    ├── intercoder_agreement.ipynb               # Inter-annotator agreement (Cohen κ, Fleiss κ, F1)
    ├── node_validation.ipynb                    # Node label cosine similarity (F1-safe swap)
    ├── graph_structural_eval.ipynb              # Single-spec graph eval (legacy)
    └── label_studio_json_to_csv.ipynb           # Convert Label Studio JSON to coder CSV format
```

Each pipeline output directory contains:
- `results_step5_1.csv` — Final LLM-extracted graphs (one row per abstract)
- `results_step1to4.csv` — Intermediate steps 1–4 outputs
- `true_staged_batch_workdir/step{1..5_1}/` — Raw OpenAI Batch API input/output JSONL files

---

## Data

> **Note:** This repository contains a **10-abstract sample** of the full dataset for anonymous review. The complete 210-abstract dataset will be released upon paper acceptance.

### Gold Annotation (`data/raw/EmpiriGraph-Psy_gold_annotation.json`)

10-abstract sample (stratified across all 6 journals, covering all 4 relation types) drawn from the full 210-abstract dataset. The full dataset covers six journals (BRT, JAP, JCCP, JCP, JEP: Educational, JEP: General), sampled at 35 abstracts per journal and 30 per decade (1960s–2025). Each abstract is annotated with:
- Normalized variable names
- Hierarchical construct–variable links
- Empirical relation edges (associational, mechanistic, moderational) with validation states
- Final resolved gold graph after coder review

To comply with copyright restrictions on abstract text, the raw abstract text is not redistributed. The release includes only ORN identifiers and the derived annotation layer.

### Coder Annotations (`data/raw/annotation_csv/`)

Sample of independent annotations by two coders on the overlap subset, used for inter-annotator agreement (see Table 3 in the paper). Columns: `task_id`, `variables`, `hierarchy`, `directional`, `correlational`, `moderation`.

---

## Notebooks

### 1. Extraction Pipeline

**`pipeline_gpt54_gpt52.ipynb`** — Runs the full five-stage pipeline using GPT-5.4 for Steps 1 and 5, GPT-5.2 for Steps 2–4, via the OpenAI Batch API. Configure `EVAL_SPECS` in the config cell to select the input data and output directory.

**`pipeline_single_step_gpt54.ipynb`** — Runs the direct-prompting baseline (single-step, GPT-5.4) via OpenAI Batch API. Outputs `single_step_gpt54_210_results.csv`.

**`extraction_pipeline_v10.ipynb`** — Source of truth for all step prompts. Not intended to be run directly; prompts are loaded from here by the batch notebooks.

### 2. Graph Evaluation

**`graph_eval.ipynb`** — Evaluates predicted graphs against the gold annotation across four conditions:
- `A_typed`: Full directed typed graph
- `B_higher_typed`: Higher-level graph (hierarchy children collapsed)
- `C_agnostic`: Type-agnostic (all edges treated as a single relation type)
- `D_higher_agnostic`: Both transforms combined

To evaluate a specific model, uncomment the corresponding entry in `EVAL_SPECS` in the config cell. Results are saved to `data/output/graph_eval/`.

**`graph_eval_functions.py`** — All graph parsing, preprocessing, and MCS evaluation functions. Required by `graph_eval.ipynb` and `intercoder_agreement.ipynb`.

### 3. Inter-annotator Agreement

**`intercoder_agreement.ipynb`** — Computes:
- Pairwise graph structural F1 (Coder A vs GT, Coder B vs GT, Coder A vs Coder B)
- Pairwise Cohen's κ (MCS-aligned multiclass, 5 categories)
- Fleiss' κ across all three coders

Results saved to `data/output/intercoder/`.

### 4. Node Validation

**`node_validation.ipynb`** — Validates structurally aligned node pairs by computing embedding-based cosine similarity (text-embedding-3-small). Applies F1-safe swap optimization: reassigns node label pairs to maximize cosine similarity while preserving matched edge count. Results saved to `data/output/node_validation/`.

---

## Reproducing Key Results

### Table 1 (Best model performance)

1. Run `graph_eval.ipynb` with `EVAL_SPECS = [('gpt54_gpt52_210', ...)]`
2. Results in `data/output/graph_eval/graph_eval_gpt54_gpt52_210_*.csv`

### Table 2 (Model comparison)

Run `graph_eval.ipynb` once per model (uncomment one spec at a time).

### Table 3 (Inter-annotator agreement)

Run `intercoder_agreement.ipynb`. Results in `data/output/intercoder/`.

### Figure 3 (Confusion matrix and error breakdown)

Computed within `graph_eval.ipynb` after running Condition A evaluation.

### Figure 6 / Table 6–7 (Node validation)

Run `node_validation.ipynb`. Results in `data/output/node_validation/`.

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Register the kernel for Jupyter

```bash
python -m ipykernel install --user --name empirigraph --display-name "EmpiriGraph-Psy"
```

Then select the `EmpiriGraph-Psy` kernel when opening any notebook.

### 3. Set API keys

Create a `.env` file in the repo root (never committed):

```
OPENAI_API_KEY=sk-...
```

The pipeline notebooks load this automatically via `os.getenv('OPENAI_API_KEY')`. For Anthropic (Claude) or Google (Gemini) runs, add `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` respectively.

---

## License

The annotation layer is released under CC BY 4.0. Code is released under MIT License.
