"""
Evaluation pipeline for clinical embeddings.

Tasks:
1. Note Recall: top-k accuracy of retrieving the next note for a patient
2. Diagnosis Prediction: multi-label ICD classification from frozen embeddings
3. UMAP Visualization: embedding space colored by diagnosis

Usage:
    python src/evaluate.py --task recall --model google/embeddinggemma-300m
    python src/evaluate.py --task diagnosis --embeddings embeddings/embeddings_google_embeddinggemma_300m.npy
    python src/evaluate.py --task umap --embeddings embeddings/embeddings_google_embeddinggemma_300m.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


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
            lambda x: eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
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


def run_full_comparison(
    models: list[dict[str, str]],
    pairs_path: Path = DATA_DIR / "temporal_pairs.json",
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
    parser.add_argument("--task", choices=["recall", "diagnosis", "umap", "compare"],
                        default="compare")
    parser.add_argument("--embeddings", type=Path, help="Path to embeddings .npy file")
    parser.add_argument("--anchor-embeddings", type=Path)
    parser.add_argument("--positive-embeddings", type=Path)
    parser.add_argument("--notes", type=Path, default=DATA_DIR / "notes_with_icd.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--top-n-codes", type=int, default=25)
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
            output_path=args.output_dir / "umap_embeddings.png",
        )

    elif args.task == "compare":
        models = [
            {"name": "text-embedding-3-small"},
            {"name": "text-embedding-3-large"},
            {"name": "google/embeddinggemma-300m"},
        ]
        run_full_comparison(models, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
