"""
Generate report-ready statistical artifacts:

1) Bootstrap confidence intervals (CIs) for recall metrics.
2) Bootstrap CIs for macro-AUROC using available per-class AUROCs.
3) Grouped per-class AUROC plot across models.

Outputs (under results/ by default):
  - bootstrap_recall_ci_table.csv
  - bootstrap_auroc_ci_table.csv
  - bootstrap_ci_summary.json
  - per_class_auroc_grouped.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
EMBEDDINGS_DIR = ROOT / "embeddings"

SAFE_NAME_BY_MODEL_NAME = {
    "text-embedding-3-small": "text_embedding_3_small",
    "text-embedding-3-large": "text_embedding_3_large",
    "google/embeddinggemma-300m": "google_embeddinggemma_300m",
    "models_embeddinggemma_infonce_best": "models_embeddinggemma_infonce_best",
    "models_embeddinggemma_hierarchical_best": "models_embeddinggemma_hierarchical_best",
}

DISPLAY_BY_SAFE_NAME = {
    "text_embedding_3_small": "OpenAI text-embedding-3-small",
    "text_embedding_3_large": "OpenAI text-embedding-3-large",
    "google_embeddinggemma_300m": "EmbeddingGemma (vanilla)",
    "models_embeddinggemma_infonce_best": "InfoNCE fine-tuned",
    "models_embeddinggemma_hierarchical_best": "Hierarchical fine-tuned",
}


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI for the mean of a 1D numeric array."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        v = float(vals[0])
        return v, v

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(vals)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(vals[idx].mean())

    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


def _topk_hits(
    anchor_embs: np.ndarray,
    positive_embs: np.ndarray,
    max_k: int = 10,
) -> dict[int, np.ndarray]:
    """
    Return hit indicators for each k in {1..max_k} using cosine similarity.
    hit[k][i] = 1 if true positive for anchor i is within top-k.
    """
    import faiss

    if anchor_embs.shape[0] != positive_embs.shape[0]:
        raise ValueError(
            f"Anchor/positive row mismatch: {anchor_embs.shape[0]} vs {positive_embs.shape[0]}"
        )
    if anchor_embs.shape[1] != positive_embs.shape[1]:
        raise ValueError(
            f"Anchor/positive dim mismatch: {anchor_embs.shape[1]} vs {positive_embs.shape[1]}"
        )

    a = np.asarray(anchor_embs, dtype=np.float32).copy()
    p = np.asarray(positive_embs, dtype=np.float32).copy()
    a /= np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    p /= np.linalg.norm(p, axis=1, keepdims=True).clip(min=1e-8)

    index = faiss.IndexFlatIP(p.shape[1])
    index.add(p)
    _, top_idx = index.search(a, max_k)

    rows = np.arange(a.shape[0])[:, None]
    out: dict[int, np.ndarray] = {}
    for k in (1, 5, 10):
        if k > max_k:
            continue
        hit = np.any(top_idx[:, :k] == rows, axis=1).astype(np.float32)
        out[k] = hit
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _safe_from_model_name(model_name: str) -> str:
    return SAFE_NAME_BY_MODEL_NAME.get(
        model_name, model_name.replace("/", "_").replace("-", "_")
    )


def build_recall_ci_table(
    comparison: dict[str, Any],
    heldout: dict[str, Any] | None,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Compute bootstrap CIs for diagnostic recall and held-out recall."""
    rows: list[dict[str, Any]] = []

    # Training-distribution diagnostic recall (from full cached pair embeddings)
    for model_name in comparison:
        safe = _safe_from_model_name(model_name)
        disp = DISPLAY_BY_SAFE_NAME.get(safe, model_name)
        anchor_path = EMBEDDINGS_DIR / f"anchor_embeddings_{safe}.npy"
        positive_path = EMBEDDINGS_DIR / f"positive_embeddings_{safe}.npy"

        if not (anchor_path.exists() and positive_path.exists()):
            continue

        hits = _topk_hits(np.load(anchor_path), np.load(positive_path), max_k=10)
        for k in (1, 5, 10):
            point = float(hits[k].mean())
            lo, hi = _bootstrap_mean_ci(hits[k], n_boot=n_boot, seed=seed + k)
            rows.append(
                {
                    "context": "training_distribution_diagnostic",
                    "model_name": model_name,
                    "model_safe_name": safe,
                    "model_display": disp,
                    "metric": f"top_{k}_recall",
                    "point_estimate": point,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_samples": int(len(hits[k])),
                    "bootstrap_n": int(n_boot),
                    "bootstrap_seed": int(seed),
                }
            )

    # Patient-held-out recall (if available)
    if heldout is not None:
        split_id = heldout.get("split", {}).get("split_id")
        if isinstance(split_id, str) and split_id:
            for safe in heldout.get("patient_heldout_generalization_recall", {}):
                disp = DISPLAY_BY_SAFE_NAME.get(safe, safe)
                anchor_path = (
                    EMBEDDINGS_DIR
                    / "heldout_recall"
                    / split_id
                    / f"anchor_embeddings_{safe}.npy"
                )
                positive_path = (
                    EMBEDDINGS_DIR
                    / "heldout_recall"
                    / split_id
                    / f"positive_embeddings_{safe}.npy"
                )
                if not (anchor_path.exists() and positive_path.exists()):
                    continue
                hits = _topk_hits(np.load(anchor_path), np.load(positive_path), max_k=10)
                for k in (1, 5, 10):
                    point = float(hits[k].mean())
                    lo, hi = _bootstrap_mean_ci(hits[k], n_boot=n_boot, seed=seed + 100 + k)
                    rows.append(
                        {
                            "context": "patient_heldout_generalization",
                            "model_name": safe,
                            "model_safe_name": safe,
                            "model_display": disp,
                            "metric": f"top_{k}_recall",
                            "point_estimate": point,
                            "ci_lower": lo,
                            "ci_upper": hi,
                            "n_samples": int(len(hits[k])),
                            "bootstrap_n": int(n_boot),
                            "bootstrap_seed": int(seed),
                        }
                    )
    return pd.DataFrame(rows)


def build_auroc_ci_table(
    comparison: dict[str, Any],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """
    Build AUROC CI table.

    CI method used here is class-bootstrap over `per_class_auroc` values because
    per-note prediction arrays are not persisted in current artifacts.
    """
    rows: list[dict[str, Any]] = []
    for model_name, model_data in comparison.items():
        diag = model_data.get("diagnosis", {})
        macro = diag.get("auroc_macro")
        per_class = diag.get("per_class_auroc", {})
        values = np.asarray(list(per_class.values()), dtype=np.float64)
        lo, hi = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed + 777)
        rows.append(
            {
                "model_name": model_name,
                "model_safe_name": _safe_from_model_name(model_name),
                "model_display": DISPLAY_BY_SAFE_NAME.get(
                    _safe_from_model_name(model_name), model_name
                ),
                "metric": "macro_auroc",
                "point_estimate": float(macro) if macro is not None else float("nan"),
                "ci_lower": lo,
                "ci_upper": hi,
                "n_classes": int(len(values)),
                "bootstrap_n": int(n_boot),
                "bootstrap_seed": int(seed),
                "ci_method": "class_bootstrap_over_per_class_auroc",
            }
        )
    return pd.DataFrame(rows)


def plot_per_class_auroc(
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    """Grouped bar chart of per-class AUROC for all models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_dir = output_path.parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(comparison.keys())
    per_class_maps = [
        comparison[m].get("diagnosis", {}).get("per_class_auroc", {}) for m in model_names
    ]
    codes = sorted(set().union(*(m.keys() for m in per_class_maps)))
    if not codes:
        raise ValueError("No per-class AUROC values found in comparison_results.json")

    # Order classes by the mean AUROC across models (descending) for readability.
    def _mean_auc(code: str) -> float:
        vals = [m.get(code, np.nan) for m in per_class_maps]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    codes = sorted(codes, key=_mean_auc, reverse=True)

    x = np.arange(len(codes))
    width = 0.8 / max(1, len(model_names))
    fig, ax = plt.subplots(figsize=(20, 7))

    cmap = plt.get_cmap("tab10", len(model_names))
    for i, model in enumerate(model_names):
        safe = _safe_from_model_name(model)
        disp = DISPLAY_BY_SAFE_NAME.get(safe, model)
        model_map = comparison.get(model, {}).get("diagnosis", {}).get("per_class_auroc", {})
        vals = [model_map.get(c, np.nan) for c in codes]
        offset = (i - (len(model_names) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=disp, color=cmap(i))

    ax.set_title("Per-class AUROC by model (top ICD codes)")
    ax.set_xlabel("ICD code")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=70, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bootstrap CI + per-class AUROC artifacts")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=RESULTS_DIR / "comparison_results.json",
        help="Path to comparison_results.json",
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=RESULTS_DIR / "heldout_recall_results.json",
        help="Path to heldout_recall_results.json (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for generated artifacts",
    )
    parser.add_argument("--bootstrap-n", type=int, default=1000, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = _load_json(args.comparison)
    heldout = _load_json(args.heldout) if args.heldout.exists() else None

    recall_df = build_recall_ci_table(
        comparison=comparison,
        heldout=heldout,
        n_boot=args.bootstrap_n,
        seed=args.seed,
    )
    auroc_df = build_auroc_ci_table(
        comparison=comparison,
        n_boot=args.bootstrap_n,
        seed=args.seed,
    )

    recall_csv = args.output_dir / "bootstrap_recall_ci_table.csv"
    auroc_csv = args.output_dir / "bootstrap_auroc_ci_table.csv"
    fig_path = args.output_dir / "per_class_auroc_grouped.png"
    summary_json = args.output_dir / "bootstrap_ci_summary.json"

    recall_df.to_csv(recall_csv, index=False)
    auroc_df.to_csv(auroc_csv, index=False)
    plot_per_class_auroc(comparison=comparison, output_path=fig_path)

    summary = {
        "inputs": {
            "comparison_results": str(args.comparison),
            "heldout_results": str(args.heldout) if args.heldout.exists() else None,
        },
        "settings": {"bootstrap_n": int(args.bootstrap_n), "seed": int(args.seed)},
        "artifacts": {
            "bootstrap_recall_ci_table_csv": str(recall_csv),
            "bootstrap_auroc_ci_table_csv": str(auroc_csv),
            "per_class_auroc_grouped_png": str(fig_path),
        },
        "notes": [
            "Recall CIs are sample-bootstrap CIs over pair-level hit indicators.",
            "AUROC CIs are class-bootstrap CIs over per_class_auroc values "
            "(not patient-level test-row bootstrap) due missing persisted per-note prediction arrays.",
        ],
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {recall_csv}")
    print(f"Saved: {auroc_csv}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()

