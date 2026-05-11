# Next Steps

After `src/train_contrastive.py` produces fine-tuned checkpoints in `models/`, the
remaining pipeline generates embeddings from those checkpoints, runs the headline
comparison, and persists results.

## 1. Generate embeddings with the fine-tuned models

Training saves checkpoints to `models/embeddinggemma_<loss>_best/` and
`models/embeddinggemma_<loss>_final/`. To compare them against the baselines,
run `embed.py` against each `_best` checkpoint:

```bash
python src/embed.py --mode pairs --input data/temporal_pairs.json \
    --model models/embeddinggemma_infonce_best

python src/embed.py --mode pairs --input data/temporal_pairs.json \
    --model models/embeddinggemma_hierarchical_best
```

`SentenceTransformer` loads from a local path identically to a HuggingFace hub ID.
Outputs land in `embeddings/`.

## 2. Run the full comparison

```bash
python src/evaluate.py --task compare --notes data/notes_with_icd.csv
```

This produces note recall (top-k) and diagnosis AUROC across every embedding set
in `embeddings/` — typically 3 baselines (Gemma + OpenAI 3-small + OpenAI 3-large)
plus 2 fine-tuned (InfoNCE + hierarchical).

## 3. UMAP visualization

```bash
python src/evaluate.py --task umap --notes data/notes_with_icd.csv \
    --embeddings embeddings/anchor_embeddings_models_embeddinggemma_hierarchical_best.npy
```

Swap `--embeddings` for whichever model you want to visualize. Outputs go to `results/`.

## Colab: persist outputs to Drive

`embeddings/`, `models/`, and `results/` are gitignored. To save them across
Colab session disconnects:

```python
import shutil, os
DRIVE_OUT = '/content/drive/MyDrive/medical_notes_embeddings_outputs'
os.makedirs(DRIVE_OUT, exist_ok=True)

for d in ['embeddings', 'models', 'results']:
    src = f'/content/contrastive_learning_clinical_embeddings/{d}'
    dst = f'{DRIVE_OUT}/{d}'
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Saved {d} -> Drive")
```

## Timing estimates (Colab T4, full 10k-patient dataset)

| Step | Estimate |
|---|---|
| Embed with 2 fine-tuned models | ~60–120 min |
| Evaluate compare | ~few min |
| UMAP | ~1–2 min |

For faster iteration during debugging, run `python src/dataset_reduce.py` first
to produce 500-patient subsets, then point the commands above at
`data/temporal_pairs_small.json` and `data/notes_with_icd_small.csv`. Full
pipeline drops from ~4–10 hr to ~20–60 min on T4.
