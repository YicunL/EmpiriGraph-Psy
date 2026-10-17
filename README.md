# EmpiriGraph-Psy

Code and sample data for the revised ACL submission, **“EmpiriGraph-Psy: A Dataset and LLM Pipeline for Extracting Empirical Relation Graphs from Psychology Abstracts.”** The study uses 210 annotated abstracts; this repository contains ten examples.

## Files

| Path | Contents |
| --- | --- |
| [`data/raw/`](data/raw/) | Gold and coder samples |
| [`data/output/`](data/output/) | Example predictions and evaluations |
| [`data/paper_results/`](data/paper_results/) | Submitted tables and retrieval sample |
| [`notebooks/extraction_pipeline_v10.ipynb`](notebooks/extraction_pipeline_v10.ipynb) | Five-stage prompts |
| [`notebooks/pipeline_gpt54_gpt52.ipynb`](notebooks/pipeline_gpt54_gpt52.ipynb) | Batch runner |
| [`notebooks/graph_eval.ipynb`](notebooks/graph_eval.ipynb) | Structural F1 |
| [`notebooks/intercoder_agreement.ipynb`](notebooks/intercoder_agreement.ipynb) | Intercoder agreement |
| [`notebooks/node_validation.ipynb`](notebooks/node_validation.ipynb) | Node alignment checks |
| [`scripts/ablation_prompts.py`](scripts/ablation_prompts.py) | Stage-omission prompts |
| [`scripts/gliner_relex_sample.py`](scripts/gliner_relex_sample.py) | GLiNER-Relex sample runner |

Steps 1 and 5 use **GPT-5.4**; Steps 2–4 use **GPT-5.2**, with low reasoning effort and low verbosity. The submitted selected run scores **0.719 micro F1** and **0.736 macro F1**. Three mixed-model repeats average **0.728 ± 0.011 macro F1**.

## Run the sample

Install `requirements.txt`, set `OPENAI_API_KEY`, and run `notebooks/pipeline_gpt54_gpt52.ipynb`. Its Batch API calls incur cost. To run the encoder baseline, install `gliner==0.2.29` with PyTorch, then execute `python scripts/gliner_relex_sample.py`.

The 210-abstract gold set, full prediction runs, and 404-abstract retrieval corpus are not included. The CSVs in [`data/paper_results/`](data/paper_results/) transcribe submitted values; the ten-example files do not reproduce those tables. The 0.736 run's exact execution snapshot is unavailable. The retrieval table uses ten post hoc selected questions (four graph-assisted F1 wins, six ties). Estimated API costs reflect the study, not current prices.

See the [Evidence Atlas](https://empirigraph.org/atlas/) for the downstream application. Check source-journal terms before redistributing abstract text in the example files.
