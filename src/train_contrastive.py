"""
Contrastive fine-tuning of EmbeddingGemma-300m on MIMIC-III temporal note pairs.

Implements:
1. Temporal contrastive learning (InfoNCE) - Radical Health baseline
2. Hierarchical contrastive learning (HiMulCon-style) with ICD structure

Usage:
    python src/train_contrastive.py --epochs 5 --batch-size 32
    python src/train_contrastive.py --loss hierarchical --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from preprocess import get_icd_chapter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class TemporalPairsDataset(Dataset):
    """Dataset of (anchor_text, positive_text) temporal note pairs."""

    def __init__(self, pairs_path: Path, icd_map_path: Path | None = None):
        with open(pairs_path) as f:
            self.pairs = json.load(f)

        self.icd_map: dict[str, list[str]] = {}
        if icd_map_path and icd_map_path.exists():
            with open(icd_map_path) as f:
                self.icd_map = json.load(f)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        item = {
            "anchor_text": pair["anchor_text"],
            "positive_text": pair["positive_text"],
            "subject_id": pair["subject_id"],
        }

        anchor_hadm = str(pair.get("anchor_hadm_id", ""))
        if anchor_hadm in self.icd_map:
            codes = self.icd_map[anchor_hadm]
            item["icd_codes"] = codes
            item["icd_chapters"] = list({get_icd_chapter(c) for c in codes})
        else:
            item["icd_codes"] = []
            item["icd_chapters"] = []

        return item


def info_nce_loss(anchor_embs: torch.Tensor, positive_embs: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss with in-batch negatives.

    anchor_embs: (B, D) normalized embeddings of anchors
    positive_embs: (B, D) normalized embeddings of positives
    """
    anchor_embs = F.normalize(anchor_embs, dim=1)
    positive_embs = F.normalize(positive_embs, dim=1)

    # Similarity matrix: (B, B)
    logits = torch.mm(anchor_embs, positive_embs.t()) / temperature

    # Labels: each anchor's positive is at the diagonal
    labels = torch.arange(logits.size(0), device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def hierarchical_contrastive_loss(
    anchor_embs: torch.Tensor,
    positive_embs: torch.Tensor,
    icd_chapters: list[list[str]],
    temperature: float = 0.07,
    chapter_weight: float = 0.3,
) -> torch.Tensor:
    """
    HiMulCon-style hierarchical contrastive loss.

    Beyond the temporal positive, samples sharing ICD chapters
    get a softer contrastive target (partial positive).
    """
    anchor_embs = F.normalize(anchor_embs, dim=1)
    positive_embs = F.normalize(positive_embs, dim=1)
    batch_size = anchor_embs.size(0)

    logits = torch.mm(anchor_embs, positive_embs.t()) / temperature

    # Build soft targets: diagonal = 1.0, shared chapter = chapter_weight
    targets = torch.zeros(batch_size, batch_size, device=logits.device)
    for i in range(batch_size):
        targets[i, i] = 1.0
        if icd_chapters[i]:
            chapters_i = set(icd_chapters[i])
            for j in range(batch_size):
                if i != j and icd_chapters[j]:
                    overlap = chapters_i & set(icd_chapters[j])
                    if overlap:
                        targets[i, j] = chapter_weight * len(overlap) / len(chapters_i)

    # Normalize targets to form a distribution
    targets = targets / targets.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Cross-entropy with soft targets
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length ICD code lists."""
    return {
        "anchor_text": [item["anchor_text"] for item in batch],
        "positive_text": [item["positive_text"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "icd_codes": [item["icd_codes"] for item in batch],
        "icd_chapters": [item["icd_chapters"] for item in batch],
    }


def train(
    model_name: str = "google/embeddinggemma-300m",
    pairs_path: Path = DATA_DIR / "temporal_pairs.json",
    icd_map_path: Path = DATA_DIR / "icd_hierarchy.json",
    output_dir: Path = MODELS_DIR,
    loss_type: str = "infonce",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    temperature: float = 0.07,
    chapter_weight: float = 0.3,
    max_length: int = 512,
) -> None:
    """Fine-tune embedding model with contrastive loss."""
    from sentence_transformers import SentenceTransformer

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Move the underlying transformer to device
    model = model.to(device)

    dataset = TemporalPairsDataset(pairs_path, icd_map_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    print(f"Dataset: {len(dataset)} pairs, {len(dataloader)} batches")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    best_loss = float("inf")
    training_log: list[dict] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            anchor_features = tokenizer(
                batch["anchor_text"], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            positive_features = tokenizer(
                batch["positive_text"], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)

            anchor_embs = model(anchor_features)["sentence_embedding"]
            positive_embs = model(positive_features)["sentence_embedding"]

            if loss_type == "hierarchical":
                loss = hierarchical_contrastive_loss(
                    anchor_embs, positive_embs,
                    batch["icd_chapters"],
                    temperature=temperature,
                    chapter_weight=chapter_weight,
                )
            else:
                loss = info_nce_loss(anchor_embs, positive_embs, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = np.mean(epoch_losses)
        log_entry = {"epoch": epoch + 1, "avg_loss": float(avg_loss), "loss_type": loss_type}
        training_log.append(log_entry)
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = output_dir / f"embeddinggemma_{loss_type}_best"
            model.save(str(save_path))
            print(f"  Saved best model to {save_path}")

    # Save final model
    final_path = output_dir / f"embeddinggemma_{loss_type}_final"
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    # Save training log
    log_path = output_dir / f"training_log_{loss_type}.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning for clinical embeddings")
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--pairs", type=Path, default=DATA_DIR / "temporal_pairs.json")
    parser.add_argument("--icd-map", type=Path, default=DATA_DIR / "icd_hierarchy.json")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--loss", choices=["infonce", "hierarchical"], default="infonce",
                        help="Loss function: infonce (temporal baseline) or hierarchical")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--chapter-weight", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train(
        model_name=args.model,
        pairs_path=args.pairs,
        icd_map_path=args.icd_map,
        output_dir=args.output_dir,
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        chapter_weight=args.chapter_weight,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
