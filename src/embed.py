"""
Generate embeddings using SentenceTransformer models (EmbeddingGemma-300m)
and OpenAI baselines. Supports batch processing for large note collections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"


def load_model(model_name: str = "google/embeddinggemma-300m"):
    """Load a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(
    model,
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a list of texts into embeddings."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def embed_with_openai(
    texts: list[str],
    model_name: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> np.ndarray:
    """Generate embeddings using OpenAI API."""
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("OPENAI-API-KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI-API-KEY or OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"OpenAI {model_name}"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model_name)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def embed_notes_from_file(
    input_path: Path,
    output_dir: Path,
    model_name: str = "google/embeddinggemma-300m",
    text_column: str = "text",
    batch_size: int = 32,
) -> Path:
    """Load notes from CSV, generate embeddings, save as .npy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    texts = df[text_column].tolist()
    print(f"Loaded {len(texts)} texts from {input_path}")

    if model_name.startswith("text-embedding-3"):
        embeddings = embed_with_openai(texts, model_name=model_name, batch_size=batch_size)
    else:
        model = load_model(model_name)
        embeddings = embed_texts(model, texts, batch_size=batch_size)

    safe_name = model_name.replace("/", "_").replace("-", "_")
    output_path = output_dir / f"embeddings_{safe_name}.npy"
    np.save(output_path, embeddings)
    print(f"Saved embeddings: {embeddings.shape} -> {output_path}")

    # Also save metadata (subject_id, hadm_id) alongside
    meta_cols = [c for c in ["subject_id", "hadm_id", "chartdate", "category"] if c in df.columns]
    if meta_cols:
        df[meta_cols].to_csv(output_dir / f"metadata_{safe_name}.csv", index=False)

    return output_path


def embed_temporal_pairs(
    pairs_path: Path,
    output_dir: Path,
    model_name: str = "google/embeddinggemma-300m",
    batch_size: int = 32,
) -> tuple[Path, Path]:
    """Embed anchor and positive texts from temporal pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pairs_path) as f:
        pairs = json.load(f)

    anchors = [p["anchor_text"] for p in pairs]
    positives = [p["positive_text"] for p in pairs]

    if model_name.startswith("text-embedding-3"):
        anchor_embs = embed_with_openai(anchors, model_name=model_name, batch_size=batch_size)
        pos_embs = embed_with_openai(positives, model_name=model_name, batch_size=batch_size)
    else:
        model = load_model(model_name)
        print("Encoding anchors...")
        anchor_embs = embed_texts(model, anchors, batch_size=batch_size)
        print("Encoding positives...")
        pos_embs = embed_texts(model, positives, batch_size=batch_size)

    safe_name = model_name.replace("/", "_").replace("-", "_")
    anchor_path = output_dir / f"anchor_embeddings_{safe_name}.npy"
    pos_path = output_dir / f"positive_embeddings_{safe_name}.npy"
    np.save(anchor_path, anchor_embs)
    np.save(pos_path, pos_embs)
    print(f"Anchor embeddings: {anchor_embs.shape} -> {anchor_path}")
    print(f"Positive embeddings: {pos_embs.shape} -> {pos_path}")
    return anchor_path, pos_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings for clinical notes")
    parser.add_argument("--input", type=Path, help="Path to notes CSV or temporal_pairs.json")
    parser.add_argument("--output-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--model", default="google/embeddinggemma-300m",
                        help="Model name (SentenceTransformer or OpenAI)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--mode", choices=["notes", "pairs"], default="notes",
                        help="'notes' for per-note embeddings, 'pairs' for temporal pair embeddings")
    args = parser.parse_args()

    if args.input is None:
        if args.mode == "pairs":
            args.input = DATA_DIR / "temporal_pairs.json"
        else:
            args.input = DATA_DIR / "notes_with_icd.csv"

    if args.mode == "pairs":
        embed_temporal_pairs(args.input, args.output_dir, args.model, args.batch_size)
    else:
        embed_notes_from_file(args.input, args.output_dir, args.model, args.text_column, args.batch_size)


if __name__ == "__main__":
    main()
