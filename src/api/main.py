"""FastAPI app exposing the four downstream applications from src/rag.py.

Run locally with:
    uvicorn src.api.main:app --reload

Environment variables (override defaults for non-canonical corpora):
    EMBEDDINGS_MODEL_SAFE_NAME  default: models_embeddinggemma_hierarchical_best
    PAIRS_FILENAME              default: temporal_pairs_small.json   (relative to data/)
    ICD_MAP_FILENAME            default: icd_hierarchy.json           (relative to data/)
    LOCAL_MODEL_PATH            default: models/embeddinggemma_hierarchical_best (auto)
    HF_REPO                     default: gaspard-loeillot/embeddinggemma-mimic-hierarchical (auto)
"""

from __future__ import annotations

# OMP env vars must be set BEFORE numpy/faiss/torch get imported -- otherwise
# on Apple Silicon, faiss-cpu and torch's bundled libomp.dylib collide and
# segfault. Keep this block at the top, ahead of any other import.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make sure src/ is importable so `from rag import ...` (and rag's own
# `from preprocess import ...`) resolve regardless of how uvicorn was launched.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .deps import get_state
from .routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag import build_index  # type: ignore

    model_safe_name = os.getenv(
        "EMBEDDINGS_MODEL_SAFE_NAME", "models_embeddinggemma_hierarchical_best"
    )
    pairs_filename = os.getenv("PAIRS_FILENAME", "temporal_pairs_small.json")
    icd_map_filename = os.getenv("ICD_MAP_FILENAME", "icd_hierarchy.json")

    local_model_path_env = os.getenv("LOCAL_MODEL_PATH")
    local_model_path = Path(local_model_path_env) if local_model_path_env else None
    hf_repo = os.getenv("HF_REPO")

    pairs_path = ROOT / "data" / pairs_filename
    icd_map_path = ROOT / "data" / icd_map_filename
    embeddings_dir = ROOT / "embeddings"

    print(
        f"[api] building corpus: model_safe_name={model_safe_name} "
        f"pairs={pairs_path.name} icd={icd_map_path.name}"
    )

    s = get_state()
    s.corpus = build_index(
        model_safe_name=model_safe_name,
        pairs_path=pairs_path,
        icd_map_path=icd_map_path,
        embeddings_dir=embeddings_dir,
        local_model_path=local_model_path,
        hf_repo=hf_repo,
    )
    s.model_safe_name = model_safe_name
    print(
        f"[api] corpus ready: {s.corpus.index.ntotal} notes, "
        f"{s.corpus.meta['subject_id'].nunique()} patients"
    )
    yield


app = FastAPI(
    title="MIMIC-III Embedding Downstream Apps",
    description=(
        "Search / RAG / cohort discovery / patient trajectory on top of the "
        "fine-tuned EmbeddingGemma MIMIC-III embeddings."
    ),
    lifespan=lifespan,
)
app.include_router(api_router)

STATIC = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
