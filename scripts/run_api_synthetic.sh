#!/usr/bin/env bash
# Boot the FastAPI frontend against the synthetic test corpus.
# Used for smoke-testing without needing the real MIMIC embeddings.
set -euo pipefail

cd "$(dirname "$0")/.."

export EMBEDDINGS_MODEL_SAFE_NAME=synthetic
export PAIRS_FILENAME=temporal_pairs_synthetic.json
export ICD_MAP_FILENAME=icd_hierarchy_synthetic.json
export HF_REPO=sentence-transformers/all-MiniLM-L6-v2

PORT="${PORT:-8765}"
exec ../../../.venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port "$PORT"
