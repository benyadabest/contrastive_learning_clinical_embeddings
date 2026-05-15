majority of our work over the semester was in Google Colab!

# Contrastive Learning for Clinical Embeddings

Fine-tuning [EmbeddingGemma-300m](https://huggingface.co/google/embeddinggemma-300m) on MIMIC-III clinical notes using temporal and hierarchical contrastive learning to produce embeddings that capture clinical semantics over stylistic differences.

Based on [Radical Health AI's approach](https://radicalhealth.ai/blog/training-a-model-that-understands-your-notes-7x-better-than-openai), which achieved 0.934 AUROC on diagnosis prediction vs 0.809 for OpenAI.

## Live demo

A staff-facing FastAPI copilot exercising all four downstream applications (semantic note search, evidence-grounded Q&A, cohort discovery by patient or free-text phenotype, and patient-trajectory analysis with LLM-generated NL interpretation) is deployed on Hugging Face Spaces:

**→ [huggingface.co/spaces/judoben/constrastive-embeddings](https://huggingface.co/spaces/judoben/constrastive-embeddings)**

The public Space runs against a synthetic corpus (the MIMIC-III data use agreement prohibits redistribution of raw notes); the app architecture and runtime behaviour are identical to a local run against the real corpus — see [Frontend: downstream apps demo](#frontend-downstream-apps-demo) for both deployment paths.

## Project artifacts

- **Final report:** [docs/4701_project_report.pdf](docs/4701_project_report.pdf)
- **End-to-end training pipeline:** [notebooks/medical_embeddings.ipynb](notebooks/medical_embeddings.ipynb) — the Colab notebook covering MIMIC-III download, preprocessing, baseline + fine-tuned embedding generation, contrastive training (InfoNCE + hierarchical), and headline evaluation
- **Downstream applications demo:** [notebooks/downstream_apps.ipynb](notebooks/downstream_apps.ipynb) — notebook-level demonstration of the four downstream tasks (precursor to the FastAPI app)

## Released models

Both fine-tuned checkpoints are hosted on the Hugging Face Hub:

| Loss | HF Hub | Top-5 Recall (diagnostic) | Top-10 Recall (diagnostic) | Top-5 Recall (held-out) | Top-10 Recall (held-out) | Macro AUROC |
|---|---|---:|---:|---:|---:|---:|
| Temporal InfoNCE | [`gaspard-loeillot/embeddinggemma-mimic-infonce`](https://huggingface.co/gaspard-loeillot/embeddinggemma-mimic-infonce) | 47.14% | 66.69% | 28.30% | 40.52% | 0.9445 |
| Hierarchical (HiMulCon-style) | [`gaspard-loeillot/embeddinggemma-mimic-hierarchical`](https://huggingface.co/gaspard-loeillot/embeddinggemma-mimic-hierarchical) | **67.13%** | **84.43%** | 27.86% | **40.80%** | **0.9474** |

Load via `SentenceTransformer("gaspard-loeillot/embeddinggemma-mimic-hierarchical")`.

95% bootstrap CIs (n=1000, seed=42):

| Loss | Top-5 Recall (diagnostic) CI | Top-10 Recall (diagnostic) CI | Top-5 Recall (held-out) CI | Top-10 Recall (held-out) CI | Macro AUROC CI* |
|---|---:|---:|---:|---:|---:|
| Temporal InfoNCE | [46.51%, 47.74%] | [66.05%, 67.30%] | [26.31%, 30.46%] | [38.42%, 42.73%] | [0.9325, 0.9562] |
| Hierarchical (HiMulCon-style) | [66.54%, 67.69%] | [84.00%, 84.89%] | [25.92%, 29.96%] | [38.64%, 43.06%] | [0.9361, 0.9586] |

\* Macro-AUROC CI is class-bootstrap over `per_class_auroc` (not per-note test-row bootstrap), since per-note probability arrays are not persisted in current artifacts.

## Approach

1. **Temporal contrastive learning (InfoNCE):** Anchor = patient note at time *t*, positive = same patient's note at *t+1*, in-batch negatives from other patients. Forces embeddings to capture patient trajectory rather than writing style.

2. **Hierarchical contrastive learning (HiMulCon-style):** Extends temporal loss with soft targets from ICD-9 code hierarchy — notes sharing diagnosis chapters get partial positive weight, producing embeddings that reflect clinical similarity at multiple granularities.

## Headline result: embeddings shifted from style to clinical content

Silhouette score on 5000 anchor embeddings, comparing how cleanly each model's embedding
geometry separates by **note category** (style) versus **ICD-9 chapter** (clinical content):

| Model | sil(chapter) | sil(category) | delta = cat − chap |
|---|---:|---:|---:|
| OpenAI text-embedding-3-large | -0.045 | +0.043 | **+0.089** |
| OpenAI text-embedding-3-small | -0.054 | +0.016 | **+0.070** |
| google/embeddinggemma-300m (vanilla) | -0.053 | -0.017 | **+0.036** |
| InfoNCE fine-tuned | -0.057 | -0.089 | **−0.032** |
| Hierarchical fine-tuned | -0.066 | -0.098 | **−0.032** |

Every baseline organizes its embedding space *more strongly by note style* than by clinical content. After contrastive fine-tuning the sign of this delta flips — both fine-tuned variants align with clinical content rather than style.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*Formerly* required MIMIC-III data in `mimic-iii-clinical-database-demo-1.4/`. The demo dataset has structured data but no clinical notes — the full dataset is located at [PhysioNet](https://physionet.org/content/mimiciii/1.4/).

Now requires MIMIC-III data in `MIMIC -III (10000 patients)/`, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k). This dataset is a subset of the full dataset with all tables included.

## Usage

```bash
# 1. Preprocess MIMIC-III data
python src/preprocess.py

# 2. Generate baseline embeddings
python src/embed.py --mode pairs --model library-model-embeddinggemma
python src/embed.py --mode pairs --model text-embedding-3-small

# 3. Fine-tune with contrastive loss
python src/train_contrastive.py --loss infonce --epochs 5
python src/train_contrastive.py --loss hierarchical --epochs 10

# 4. Evaluate
python src/evaluate.py --task compare
python src/evaluate.py --task umap --embeddings embeddings/<file>.npy
python src/evaluate.py --task heldout-recall --heldout-patients 50 --heldout-seed 42 --fail-on-model-error --no-json-fallback
python src/report_metrics.py --bootstrap-n 1000 --seed 42
```

`compare` recall values are training-distribution diagnostics. Use `heldout-recall` as
the primary generalization metric.

## Evaluation

| Metric | Task |
|--------|------|
| Top-5 recall accuracy | Retrieving next patient note from embeddings |
| Patient-held-out recall | Next-note retrieval on patients unseen during contrastive fine-tuning |
| Macro AUROC | Multi-label ICD-9 diagnosis prediction (logistic regression on frozen embeddings) |
| UMAP visualization | Embedding clusters colored by ICD chapter |

## Model Comparison

| Model | Type |
|-------|------|
| OpenAI text-embedding-3-small | General-purpose baseline |
| OpenAI text-embedding-3-large | General-purpose baseline |
| EmbeddingGemma (temporal contrastive) | Fine-tuned baseline |
| EmbeddingGemma (hierarchical contrastive) | Our extension |

## Project Structure

```
src/
├── preprocess.py          # MIMIC-III data preprocessing + temporal pair construction
├── embed.py               # Embedding generation (EmbeddingGemma + OpenAI)
├── train_contrastive.py   # Contrastive fine-tuning (InfoNCE + hierarchical)
├── evaluate.py            # Note recall, diagnosis prediction, UMAP (incl. UMAP-by-category)
├── push_to_hub.py         # One-shot script to push checkpoints to HF Hub
├── rag.py                 # FAISS index + patient search + RAG + cohort + trajectory
└── api/                   # FastAPI frontend wrapping the four downstream apps
    ├── main.py            # App + lifespan (builds CorpusIndex once at startup)
    ├── routes.py          # /api/{health,patients,search,ask,similar-patients,trajectory}
    ├── deps.py            # CorpusState singleton + patient-pool cache
    ├── schemas.py         # Pydantic request bodies
    └── static/            # Single-page UI (index.html + app.js + style.css)
notebooks/
├── medical_embeddings.ipynb   # End-to-end Colab pipeline (data → embed → fine-tune → eval)
└── downstream_apps.ipynb      # Notebook demo of the four downstream applications
scripts/
├── make_synthetic_corpus.py   # Build a tiny stand-in corpus for the frontend smoke test
└── run_api_synthetic.sh       # Boot the API against the synthetic corpus
docs/
├── 4701_project_report.pdf    # Final project report
├── assignment.md
└── radical_health_blog.md
Dockerfile                     # Container build (HF Spaces / generic host)
```

## Frontend: downstream apps demo

FastAPI + vanilla HTML/JS app that wraps `src/rag.py` and serves the four downstream
applications (search / RAG / cohort / trajectory) at `http://localhost:8000`.

### Run against the real corpus

Requires the cached anchor embeddings and the row-aligned `temporal_pairs_small.json`
(both gitignored — produced by `embed.py` + `dataset_reduce.py`):

```bash
uvicorn src.api.main:app --reload
```

Defaults to the hierarchical fine-tuned model from HF Hub. Override via env vars
(see [src/api/main.py](src/api/main.py)) to swap model, pairs file, or ICD map.

If `OPENAI_API_KEY` is set in `.env`, the **Ask** tab returns an LLM-generated answer
grounded in the retrieved notes. Otherwise it returns retrieval only.

### Run against a synthetic corpus (no MIMIC required)

For demoing the UI without the full embedding pipeline:

```bash
python scripts/make_synthetic_corpus.py   # ~80 notes, ~30 s
scripts/run_api_synthetic.sh              # boots on port 8765
```

Synthetic data uses `sentence-transformers/all-MiniLM-L6-v2` (small, fast load) so
similarity scores remain meaningful: the same model encodes the corpus and the
query.

### Deploy

The provided `Dockerfile` works as-is for [Hugging Face Spaces (Docker SDK)](https://huggingface.co/docs/hub/spaces-sdks-docker)
and generic container hosts. The image expects `data/` and `embeddings/` to be
populated at build time; for a public deploy, bundle a non-PHI subset of MIMIC
into the build context.

## Reproducing the headline UMAP-by-category finding

```bash
python src/evaluate.py --task umap-anchors --all-models --n-samples 5000
```

Generates 10 PNGs (5 models × 2 colorings) plus `results/umap_anchors_silhouette.json`.

## Report-ready statistical artifacts

```bash
python src/report_metrics.py --bootstrap-n 1000 --seed 42
```

Generates:
- `results/bootstrap_recall_ci_table.csv`
- `results/bootstrap_auroc_ci_table.csv`
- `results/bootstrap_ci_summary.json`
- `results/per_class_auroc_grouped.png`

## Authors

Benjamin Shvartsman, Timothy Lin, Gaspard Loeillot
