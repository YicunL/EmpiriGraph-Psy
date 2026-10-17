# EmpiriGraph-Psy

## Data

| Path | Contents |
| --- | --- |
| [`data/raw/EmpiriGraph-Psy_gold_annotation.json`](data/raw/EmpiriGraph-Psy_gold_annotation.json) | Gold graphs |
| [`data/raw/annotation_csv/`](data/raw/annotation_csv/) | Annotator A, B, and C graphs |
| [`data/output/`](data/output/) | Model outputs and graph evaluations |
| [`data/results/`](data/results/) | Evaluation, ablation, model, and retrieval tables |

## Code

| Path | Purpose |
| --- | --- |
| [`notebooks/extraction_pipeline_v10.ipynb`](notebooks/extraction_pipeline_v10.ipynb) | Five-stage extraction |
| [`notebooks/pipeline_gpt54_gpt52.ipynb`](notebooks/pipeline_gpt54_gpt52.ipynb) | GPT-5.4/GPT-5.2 Batch pipeline |
| [`notebooks/graph_eval.ipynb`](notebooks/graph_eval.ipynb) | Structural graph evaluation |
| [`notebooks/intercoder_agreement.ipynb`](notebooks/intercoder_agreement.ipynb) | Annotator agreement |
| [`notebooks/node_validation.ipynb`](notebooks/node_validation.ipynb) | Node alignment |
| [`scripts/annotation_to_csv.py`](scripts/annotation_to_csv.py) | Label Studio JSON to CSV |
| [`scripts/ablation_prompts.py`](scripts/ablation_prompts.py) | Pipeline ablations |
| [`scripts/gliner_relex.py`](scripts/gliner_relex.py) | GLiNER-Relex baseline |

Install [`requirements.txt`](requirements.txt) and set `OPENAI_API_KEY` for the API notebooks.
