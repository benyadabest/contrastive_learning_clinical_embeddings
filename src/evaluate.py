"""
Evaluation pipeline for clinical embeddings.

Tasks:
1. Note Recall: top-k accuracy of retrieving the next note for a patient
2. Diagnosis Prediction: multi-label ICD classification from frozen embeddings
3. UMAP Visualization: embedding space colored by diagnosis chapter
4. UMAP-by-category vs. UMAP-by-chapter (pair-anchor variant): tests whether
   the embedding geometry is driven by clinical content (ICD chapter) or by
   stylistic note-template structure (NOTEEVENTS.CATEGORY).

Usage:
    python src/evaluate.py --task recall --model google/embeddinggemma-300m
    python src/evaluate.py --task diagnosis --embeddings embeddings/embeddings_google_embeddinggemma_300m.npy
    python src/evaluate.py --task umap --embeddings embeddings/embeddings_google_embeddinggemma_300m.npy
    python src/evaluate.py --task umap-anchors --model models_embeddinggemma_hierarchical_best
    python src/evaluate.py --task umap-anchors --all-models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

PAIR_ANCHOR_MODELS = [
    "google_embeddinggemma_300m",
    "models_embeddinggemma_infonce_best",
    "models_embeddinggemma_hierarchical_best",
    "text_embedding_3_small",
    "text_embedding_3_large",
]


def evaluate_note_recall(
    anchor_embs: np.ndarray,
    positive_embs: np.ndarray,
    top_k: list[int] | None = None,
) -> dict[str, float]:
    """
    Evaluate note recall: for each anchor, check if the correct positive
    is in the top-k most similar embeddings.

    Returns top-k accuracy for each k.
    """
    if top_k is None:
        top_k = [1, 5, 10]

    # Normalize
    anchor_norm = anchor_embs / np.linalg.norm(anchor_embs, axis=1, keepdims=True)
    positive_norm = positive_embs / np.linalg.norm(positive_embs, axis=1, keepdims=True)

    # Cosine similarity matrix: (N, N)
    sim_matrix = anchor_norm @ positive_norm.T

    results = {}
    n = sim_matrix.shape[0]

    for k in top_k:
        # For each anchor, get top-k indices
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
        # Check if the correct positive (diagonal) is in top-k
        correct = sum(1 for i in range(n) if i in top_k_indices[i])
        accuracy = correct / n
        results[f"top_{k}_accuracy"] = accuracy
        print(f"  Top-{k} recall accuracy: {accuracy:.4f} ({correct}/{n})")

    return results


def evaluate_diagnosis_prediction(
    embeddings: np.ndarray,
    notes_df: pd.DataFrame,
    top_n_codes: int = 25,
) -> dict[str, float]:
    """
    Multi-label ICD-9 diagnosis prediction using frozen embeddings.

    Trains OneVsRest logistic regression on the most frequent ICD codes.
    Reports AUROC and top-k accuracy.
    """
    # Parse ICD codes
    if "icd_codes" in notes_df.columns:
        notes_df = notes_df.copy()
        notes_df["icd_codes"] = notes_df["icd_codes"].apply(
            lambda x: eval(x) if isinstance(x, str) and (x.strip('[] ') != 'nan') else (x if isinstance(x, list) else [])
        )
    else:
        raise ValueError("notes_df must have 'icd_codes' column")

    # Filter to notes with ICD codes
    mask = notes_df["icd_codes"].apply(len) > 0
    embeddings_filtered = embeddings[mask.values]
    labels = notes_df.loc[mask, "icd_codes"].tolist()

    if len(embeddings_filtered) < 20:
        print("  Too few samples with ICD codes for diagnosis prediction")
        return {"auroc": 0.0, "note": "insufficient_data"}

    # Get top-N most frequent codes
    all_codes = [code for codes in labels for code in codes]
    code_counts = pd.Series(all_codes).value_counts()
    top_codes = code_counts.head(top_n_codes).index.tolist()
    print(f"  Using top {len(top_codes)} ICD codes (most frequent)")

    # Filter labels to top codes only
    labels_filtered = [[c for c in codes if c in top_codes] for codes in labels]
    mask2 = [len(codes) > 0 for codes in labels_filtered]
    embeddings_filtered = embeddings_filtered[mask2]
    labels_filtered = [codes for codes, m in zip(labels_filtered, mask2) if m]

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=top_codes)
    y = mlb.fit_transform(labels_filtered)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_filtered, y, test_size=0.2, random_state=42,
    )

    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train OneVsRest logistic regression
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = clf.predict_proba(X_test)

    # Compute AUROC (macro, only for classes present in test set)
    present_classes = y_test.sum(axis=0) > 0
    if present_classes.sum() < 2:
        print("  Too few classes present in test set")
        return {"auroc": 0.0, "note": "insufficient_classes"}

    auroc = roc_auc_score(
        y_test[:, present_classes],
        y_pred_proba[:, present_classes],
        average="macro",
    )
    print(f"  Macro AUROC: {auroc:.4f}")

    # Per-class AUROC for top codes
    per_class = {}
    for i, code in enumerate(top_codes):
        if present_classes[i] and y_test[:, i].sum() > 0:
            try:
                auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
                per_class[code] = float(auc)
            except ValueError:
                pass

    return {"auroc_macro": float(auroc), "per_class_auroc": per_class}


def create_umap_visualization(
    embeddings: np.ndarray,
    notes_df: pd.DataFrame,
    output_path: Path,
    n_samples: int = 5000,
) -> None:
    """Create UMAP visualization of embeddings colored by primary ICD chapter."""
    import matplotlib.pyplot as plt
    import umap

    from preprocess import get_icd_chapter

    # Sample if too many
    if len(embeddings) > n_samples:
        idx = np.random.RandomState(42).choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
        notes_df = notes_df.iloc[idx]

    # Get primary ICD chapter for each note
    if "icd_codes" in notes_df.columns:
        chapters = notes_df["icd_codes"].apply(
            lambda x: get_icd_chapter((eval(x) if isinstance(x, str) else x)[0])
            if (isinstance(x, str) and eval(x)) or (isinstance(x, list) and x)
            else "none"
        )
    else:
        chapters = pd.Series(["none"] * len(notes_df))

    # UMAP reduction
    print("  Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    unique_chapters = sorted(chapters.unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chapters)))

    for chapter, color in zip(unique_chapters, colors):
        mask = chapters == chapter
        label = chapter.split("_", 1)[-1] if "_" in chapter else chapter
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], label=label, s=5, alpha=0.6)

    ax.set_title("UMAP of Clinical Note Embeddings (colored by ICD chapter)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3, fontsize=8)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  UMAP saved to {output_path}")
    plt.close()


def _build_anchor_metadata(
    pairs: list[dict],
    icd_map: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """
    For every anchor in `pairs`, derive (category, primary_chapter).

    - category comes from `anchor_category` (NOTEEVENTS.CATEGORY of the anchor note).
    - primary_chapter is the ICD-9 chapter of the *first* code on the anchor admission;
      "none" if the admission has no codes; "unknown" if the code falls outside known
      chapters.
    """
    from preprocess import get_icd_chapter

    categories: list[str] = []
    chapters: list[str] = []
    for p in pairs:
        cat = p.get("anchor_category")
        if not isinstance(cat, str) or not cat:
            cat = "unknown"
        categories.append(cat)

        hadm = p.get("anchor_hadm_id")
        chap = "none"
        if hadm is not None:
            try:
                codes = icd_map.get(str(int(hadm)), [])
            except (TypeError, ValueError):
                codes = []
            if codes:
                chap = get_icd_chapter(codes[0])
        chapters.append(chap)
    return categories, chapters


def _scatter_by_label(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    legend_max: int = 25,
) -> None:
    """Scatter `coords` colored by `labels`, with a legend (truncated if too long)."""
    import matplotlib.pyplot as plt

    labels_arr = np.asarray(labels)
    counts = pd.Series(labels_arr).value_counts()
    unique = counts.index.tolist()
    cmap = plt.cm.tab20
    if len(unique) > 20:
        cmap = plt.cm.gist_ncar
    colors = cmap(np.linspace(0, 1, max(len(unique), 2)))

    fig, ax = plt.subplots(figsize=(14, 10))
    for label, color in zip(unique, colors):
        mask = labels_arr == label
        display = label.split("_", 1)[-1] if "_" in label else label
        display = f"{display}  (n={int(mask.sum())})"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], label=display, s=5, alpha=0.6,
        )

    ax.set_title(title)
    handles, labs = ax.get_legend_handles_labels()
    if len(handles) > legend_max:
        handles, labs = handles[:legend_max], labs[:legend_max]
        labs[-1] = labs[-1] + " ..."
    ax.legend(
        handles, labs,
        bbox_to_anchor=(1.05, 1), loc="upper left",
        markerscale=3, fontsize=8,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def umap_pair_anchors(
    model_safe_name: str,
    pairs_path: Path = DATA_DIR / "temporal_pairs_small.json",
    icd_map_path: Path = DATA_DIR / "icd_hierarchy.json",
    embeddings_dir: Path = EMBEDDINGS_DIR,
    output_dir: Path = RESULTS_DIR,
    n_samples: int = 5000,
    seed: int = 42,
    silhouette_n: int = 5000,
) -> dict:
    """
    Generate parallel UMAPs of the anchor embeddings for one model, colored by
    (a) primary ICD chapter and (b) anchor note category. Computes silhouette
    scores against both labelings on the *raw* high-dimensional embeddings, so
    the visual finding has a quantitative companion.

    Returns a dict of metadata + silhouette scores; also writes:
      - {output_dir}/umap_embeddings_{model_safe_name}_by_chapter.png
      - {output_dir}/umap_embeddings_{model_safe_name}_by_category.png

    Both PNGs share identical UMAP coordinates (UMAP is fit once); only the
    coloring differs. This is the only valid way to compare the two clusterings.
    """
    import umap

    output_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = embeddings_dir / f"anchor_embeddings_{model_safe_name}.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"Missing anchor embeddings: {anchor_path}")

    # Load embeddings (mmap to avoid RAM blowups on the OpenAI-3-large 581 MB file)
    anchors = np.load(anchor_path, mmap_mode="r")
    n_total = anchors.shape[0]
    print(f"[{model_safe_name}] anchor embeddings: shape={anchors.shape} dtype={anchors.dtype}")

    # Load metadata
    with open(pairs_path) as f:
        pairs = json.load(f)
    if len(pairs) != n_total:
        raise ValueError(
            f"Mismatch: {len(pairs)} pairs in {pairs_path.name} vs "
            f"{n_total} embeddings in {anchor_path.name}. The cached embeddings "
            "must have been generated from a different temporal_pairs file."
        )
    with open(icd_map_path) as f:
        icd_map = json.load(f)

    categories_full, chapters_full = _build_anchor_metadata(pairs, icd_map)

    # Sample with fixed seed; materialize the sampled rows into RAM and cast to
    # float32 (sklearn's silhouette_score is faster on float32 than float64).
    rng = np.random.RandomState(seed)
    if n_samples >= n_total:
        idx = np.arange(n_total)
    else:
        idx = rng.choice(n_total, size=n_samples, replace=False)
    idx_sorted = np.sort(idx)  # mmap slicing is faster on sorted indices
    X = np.asarray(anchors[idx_sorted], dtype=np.float32)
    cats = [categories_full[i] for i in idx_sorted]
    chaps = [chapters_full[i] for i in idx_sorted]

    # Fit UMAP once
    print(f"[{model_safe_name}] fitting UMAP on {X.shape}...")
    reducer = umap.UMAP(
        n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1, metric="cosine",
    )
    coords = reducer.fit_transform(X)
    print(f"[{model_safe_name}] UMAP done.")

    # Plot both colorings on the same coords
    title_prefix = f"UMAP of anchor embeddings ({model_safe_name})"
    _scatter_by_label(
        coords, chaps,
        title=f"{title_prefix}\ncolored by primary ICD-9 chapter",
        output_path=output_dir / f"umap_embeddings_{model_safe_name}_by_chapter.png",
    )
    _scatter_by_label(
        coords, cats,
        title=f"{title_prefix}\ncolored by note category",
        output_path=output_dir / f"umap_embeddings_{model_safe_name}_by_category.png",
    )

    # Silhouette scores on the raw high-dim embeddings (sub-sample for speed if
    # needed). Higher silhouette => labels separate the embedding geometry better.
    sil_idx = idx_sorted
    if silhouette_n < len(sil_idx):
        sil_pick = rng.choice(len(sil_idx), size=silhouette_n, replace=False)
        sil_idx = idx_sorted[sil_pick]
        X_sil = np.asarray(anchors[np.sort(sil_idx)], dtype=np.float32)
        cats_sil = [categories_full[i] for i in np.sort(sil_idx)]
        chaps_sil = [chapters_full[i] for i in np.sort(sil_idx)]
    else:
        X_sil, cats_sil, chaps_sil = X, cats, chaps

    # Silhouette is undefined if a single cluster only -> filter out singletons
    def _safe_silhouette(X_, labs):
        labs_arr = np.asarray(labs)
        counts = pd.Series(labs_arr).value_counts()
        keep_labels = set(counts[counts >= 2].index)
        keep = np.array([i for i, l in enumerate(labs_arr) if l in keep_labels])
        if len(keep) < 50 or pd.Series(labs_arr[keep]).nunique() < 2:
            return float("nan")
        return float(silhouette_score(X_[keep], labs_arr[keep], metric="cosine"))

    sil_chapter = _safe_silhouette(X_sil, chaps_sil)
    sil_category = _safe_silhouette(X_sil, cats_sil)

    print(
        f"[{model_safe_name}] silhouette  by_chapter={sil_chapter:.4f}  "
        f"by_category={sil_category:.4f}  delta(cat-chap)={sil_category - sil_chapter:+.4f}"
    )

    return {
        "model": model_safe_name,
        "n_anchors_total": int(n_total),
        "n_umap_samples": int(len(idx_sorted)),
        "n_silhouette_samples": int(len(sil_idx)),
        "n_unique_chapters": int(pd.Series(chaps).nunique()),
        "n_unique_categories": int(pd.Series(cats).nunique()),
        "silhouette_by_chapter": sil_chapter,
        "silhouette_by_category": sil_category,
        "silhouette_delta": sil_category - sil_chapter,
    }


def run_umap_anchors_all(
    models: list[str] | None = None,
    output_dir: Path = RESULTS_DIR,
    n_samples: int = 5000,
) -> None:
    """Run umap_pair_anchors for every model and persist a comparison JSON."""
    if models is None:
        models = PAIR_ANCHOR_MODELS
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for m in models:
        try:
            res = umap_pair_anchors(m, output_dir=output_dir, n_samples=n_samples)
            results.append(res)
        except FileNotFoundError as e:
            print(f"[skip] {m}: {e}")

    out = output_dir / "umap_anchors_silhouette.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSilhouette table -> {out}")

    if results:
        print(f"\n{'Model':<48} {'sil(chap)':<12} {'sil(cat)':<12} {'delta':<10}")
        print("-" * 82)
        for r in results:
            print(
                f"{r['model']:<48} "
                f"{r['silhouette_by_chapter']:<12.4f} "
                f"{r['silhouette_by_category']:<12.4f} "
                f"{r['silhouette_delta']:<+10.4f}"
            )


def run_full_comparison(
    models: list[dict[str, str]],
    notes_path: Path = DATA_DIR / "notes_with_icd.csv",
    output_dir: Path = RESULTS_DIR,
) -> None:
    """Run recall + diagnosis prediction across all model variants and compare."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_info in models:
        name = model_info["name"]
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 60}")

        safe_name = name.replace("/", "_").replace("-", "_")

        # Note recall
        anchor_path = EMBEDDINGS_DIR / f"anchor_embeddings_{safe_name}.npy"
        pos_path = EMBEDDINGS_DIR / f"positive_embeddings_{safe_name}.npy"

        recall_results = {}
        if anchor_path.exists() and pos_path.exists():
            anchors = np.load(anchor_path)
            positives = np.load(pos_path)
            print("\nNote Recall:")
            recall_results = evaluate_note_recall(anchors, positives)
        else:
            print(f"  Skipping recall (no pair embeddings at {anchor_path})")

        # Diagnosis prediction
        emb_path = EMBEDDINGS_DIR / f"embeddings_{safe_name}.npy"
        diag_results = {}
        if emb_path.exists() and notes_path.exists():
            embeddings = np.load(emb_path)
            notes_df = pd.read_csv(notes_path)
            print("\nDiagnosis Prediction:")
            diag_results = evaluate_diagnosis_prediction(embeddings, notes_df)
        else:
            print(f"  Skipping diagnosis prediction (no embeddings at {emb_path})")

        results[name] = {
            "recall": recall_results,
            "diagnosis": diag_results,
        }

    # Save comparison
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison results saved to {output_dir / 'comparison_results.json'}")

    # Print summary table
    print(f"\n{'Model':<40} {'Top-5 Recall':<15} {'AUROC':<10}")
    print("-" * 65)
    for name, res in results.items():
        recall = res["recall"].get("top_5_accuracy", "N/A")
        auroc = res["diagnosis"].get("auroc_macro", "N/A")
        recall_str = f"{recall:.4f}" if isinstance(recall, float) else recall
        auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) else auroc
        print(f"{name:<40} {recall_str:<15} {auroc_str:<10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clinical embeddings")
    parser.add_argument(
        "--task",
        choices=["recall", "diagnosis", "umap", "compare", "umap-anchors"],
        default="compare",
    )
    parser.add_argument("--embeddings", type=Path, help="Path to embeddings .npy file")
    parser.add_argument("--anchor-embeddings", type=Path)
    parser.add_argument("--positive-embeddings", type=Path)
    parser.add_argument("--notes", type=Path, default=DATA_DIR / "notes_with_icd.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-name", type=str, default="umap_embeddings.png")
    parser.add_argument("--top-n-codes", type=int, default=25)
    # umap-anchors task
    parser.add_argument(
        "--model", default=None,
        help="Safe-name of the model for umap-anchors (e.g., models_embeddinggemma_hierarchical_best)",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run umap-anchors over PAIR_ANCHOR_MODELS",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Sample size for UMAP fit (default 5000 for parity with existing UMAPs)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "recall":
        if not args.anchor_embeddings or not args.positive_embeddings:
            print("Error: --anchor-embeddings and --positive-embeddings required for recall")
            return
        anchors = np.load(args.anchor_embeddings)
        positives = np.load(args.positive_embeddings)
        results = evaluate_note_recall(anchors, positives)
        with open(args.output_dir / "recall_results.json", "w") as f:
            json.dump(results, f, indent=2)

    elif args.task == "diagnosis":
        if not args.embeddings:
            print("Error: --embeddings required for diagnosis prediction")
            return
        embeddings = np.load(args.embeddings)
        notes_df = pd.read_csv(args.notes)
        results = evaluate_diagnosis_prediction(embeddings, notes_df, top_n_codes=args.top_n_codes)
        with open(args.output_dir / "diagnosis_results.json", "w") as f:
            json.dump(results, f, indent=2)

    elif args.task == "umap":
        if not args.embeddings:
            print("Error: --embeddings required for UMAP")
            return
        embeddings = np.load(args.embeddings)
        notes_df = pd.read_csv(args.notes)
        create_umap_visualization(
            embeddings, notes_df,
            output_path=args.output_dir / args.output_name,
        )

    elif args.task == "compare":
        models = [
            {"name": "text-embedding-3-small"},
            {"name": "text-embedding-3-large"},
            {"name": "google/embeddinggemma-300m"},
            {"name": "models_embeddinggemma_infonce_best"},
            {"name": "models_embeddinggemma_hierarchical_best"}
        ]
        run_full_comparison(models, notes_path=args.notes, output_dir=args.output_dir)

    elif args.task == "umap-anchors":
        if args.all_models:
            run_umap_anchors_all(
                models=PAIR_ANCHOR_MODELS,
                output_dir=args.output_dir,
                n_samples=args.n_samples,
            )
        elif args.model:
            res = umap_pair_anchors(
                args.model,
                output_dir=args.output_dir,
                n_samples=args.n_samples,
            )
            print(json.dumps(res, indent=2))
        else:
            print("Error: provide either --model <safe_name> or --all-models for umap-anchors")


if __name__ == "__main__":
    main()
