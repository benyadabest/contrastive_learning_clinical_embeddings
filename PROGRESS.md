# Progress

## 2026-03-13: Project scaffolding + pipeline implementation

### Completed

**MCP Server Config**
- Added Baseten docs MCP server to `.mcp.json` (`https://docs.baseten.co/mcp`)
- `npx add-mcp` requires an interactive terminal — run manually if you want the guided setup: `npx add-mcp https://docs.baseten.co`

**Python Environment**
- Created `.venv` with Python 3.12
- Installed all dependencies: `sentence-transformers`, `torch`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `umap-learn`, `matplotlib`, `openai`, `python-dotenv`, `tqdm`
- Pinned in `requirements.txt`

**Data Preprocessing (`src/preprocess.py`)**
- Loads MIMIC-III tables: NOTEEVENTS, ADMISSIONS, PATIENTS, DIAGNOSES_ICD
- Builds temporal note pairs (anchor at t, positive at t+1) per patient
- Builds ICD-9 hierarchy map (admission → list of codes, with chapter-level grouping)
- Exports: `icd_hierarchy.json`, `admissions_summary.csv`, `diagnosis_labels.csv`
- Verified on demo dataset: 129 admissions, 1761 diagnosis labels, 100 patients
- **Note**: NOTEEVENTS is empty in the MIMIC-III demo. Full dataset from PhysioNet needed for training.

**Embedding Generation (`src/embed.py`)**
- Supports EmbeddingGemma-300m via SentenceTransformer
- Supports OpenAI `text-embedding-3-small` and `text-embedding-3-large` baselines
- Two modes: per-note embeddings, and temporal pair embeddings (anchor/positive)
- Batch processing with progress bars

**Contrastive Fine-Tuning (`src/train_contrastive.py`)**
- InfoNCE loss (temporal contrastive) — Radical Health baseline
- Hierarchical contrastive loss (HiMulCon-style) — uses ICD chapter structure for soft targets
- Supports MPS (Apple Silicon), CUDA, and CPU
- Cosine annealing LR schedule, gradient clipping
- Saves best + final model checkpoints, training logs

**Evaluation (`src/evaluate.py`)**
- Note Recall: top-k accuracy of retrieving next note from embeddings
- Diagnosis Prediction: multi-label ICD-9 classification with OneVsRest logistic regression, reports macro AUROC
- UMAP visualization: embedding space colored by ICD chapter
- Full model comparison mode: runs all metrics across OpenAI + EmbeddingGemma variants

**Other**
- `.gitignore` configured (excludes `.env`, `.venv`, data, models, embeddings)
- OpenAI and Baseten API keys in `.env`

### Blockers / Next Steps

1. **Get full MIMIC-III NOTEEVENTS**: The demo dataset has no clinical notes. Apply on PhysioNet or use the Kaggle download.
2. **Run baseline embeddings**: Once notes are available:
   ```bash
   source .venv/bin/activate
   python src/preprocess.py
   python src/embed.py --mode pairs --model google/embeddinggemma-300m
   python src/embed.py --mode pairs --model text-embedding-3-small
   python src/embed.py --mode pairs --model text-embedding-3-large
   ```
3. **Run contrastive fine-tuning**:
   ```bash
   python src/train_contrastive.py --loss infonce --epochs 5
   python src/train_contrastive.py --loss hierarchical --epochs 10
   ```
4. **Evaluate**:
   ```bash
   python src/evaluate.py --task compare
   python src/evaluate.py --task umap --embeddings embeddings/<file>.npy
   ```
5. **Baseten deployment**: Deploy fine-tuned model as embedding service
6. **Fasten Health integration**: Connect embeddings to patient record layer

### Project Structure
```
medical-notes-embeddings/
├── .claude/CLAUDE.md          # Project instructions
├── .mcp.json                  # MCP server config (Baseten docs)
├── .env                       # API keys (not committed)
├── requirements.txt           # Python dependencies
├── PROGRESS.md                # This file
├── docs/
│   ├── assignment.md          # Project spec
│   └── radical_health_blog.md # Reference blog post
├── src/
│   ├── preprocess.py          # MIMIC-III data preprocessing
│   ├── embed.py               # Embedding generation
│   ├── train_contrastive.py   # Contrastive fine-tuning
│   └── evaluate.py            # Evaluation pipeline
├── mimic-iii-clinical-database-demo-1.4/  # Demo data
├── data/                      # Preprocessed outputs (generated)
├── embeddings/                # Embedding files (generated)
├── models/                    # Fine-tuned models (generated)
└── results/                   # Evaluation results (generated)
```
