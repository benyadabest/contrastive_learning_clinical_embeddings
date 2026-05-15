"""HTTP routes for the four downstream apps + health/patients metadata.

Each route is a thin wrapper around the corresponding function in
src/rag.py; the only translation work is DataFrame -> JSON-safe records.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from .deps import get_state
from .schemas import (
    AskRequest,
    SearchRequest,
    SimilarPatientsByTextRequest,
    SimilarPatientsRequest,
    TrajectoryInterpretRequest,
    TrajectoryRequest,
)

router = APIRouter(prefix="/api")


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    return value


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {col: _json_safe(row[col]) for col in df.columns}
        for _, row in df.iterrows()
    ]


def _require_corpus():
    s = get_state()
    if s.corpus is None:
        raise HTTPException(503, "corpus not loaded")
    return s


@router.get("/health")
def health() -> dict:
    s = get_state()
    if s.corpus is None:
        return {"ok": False, "reason": "corpus not loaded"}
    return {
        "ok": True,
        "n_notes": int(s.corpus.index.ntotal),
        "n_patients": int(s.corpus.meta["subject_id"].nunique()),
        "n_categories": int(s.corpus.meta["anchor_category"].nunique()),
        "model_safe_name": s.model_safe_name,
        "has_openai_key": bool(
            os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI-API-KEY")
        ),
    }


@router.get("/patients")
def patients(limit: int = 200) -> list[dict]:
    s = _require_corpus()
    counts = s.corpus.meta["subject_id"].value_counts().head(limit)
    return [
        {"subject_id": int(sid), "n_notes": int(n)}
        for sid, n in counts.items()
    ]


@router.post("/search")
def search_endpoint(req: SearchRequest) -> list[dict]:
    s = _require_corpus()
    from rag import search  # type: ignore

    df = search(s.corpus, req.query, k=req.k, snippet_chars=req.snippet_chars)
    return _df_to_records(df)


@router.post("/ask")
def ask_endpoint(req: AskRequest) -> dict:
    s = _require_corpus()
    from rag import ask  # type: ignore

    res = ask(
        s.corpus,
        req.question,
        k=req.k,
        openai_model=req.model,
        mode=req.mode,
    )
    res["retrieved"] = _df_to_records(res["retrieved"])
    return res


@router.post("/similar-patients")
def similar_patients_endpoint(req: SimilarPatientsRequest) -> list[dict]:
    s = _require_corpus()
    from rag import find_similar_patients  # type: ignore

    try:
        df = find_similar_patients(
            s.corpus, req.subject_id, k=req.k, pooled=s.patient_pooled()
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    return _df_to_records(df)


@router.post("/similar-patients-by-text")
def similar_patients_by_text_endpoint(req: SimilarPatientsByTextRequest) -> list[dict]:
    s = _require_corpus()
    from rag import find_similar_patients_by_text  # type: ignore

    df = find_similar_patients_by_text(
        s.corpus, req.query, k=req.k, pooled=s.patient_pooled()
    )
    return _df_to_records(df)


@router.post("/trajectory")
def trajectory_endpoint(req: TrajectoryRequest) -> dict:
    s = _require_corpus()
    from rag import patient_trajectory  # type: ignore

    try:
        traj = patient_trajectory(s.corpus, req.subject_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    df = traj["notes"][[
        "anchor_date",
        "anchor_category",
        "anchor_hadm_id",
        "cos_sim_prev",
        "l2_prev",
        "pca1",
        "pca2",
    ]].copy()

    notes_records = []
    for _, r in df.iterrows():
        hadm = r["anchor_hadm_id"]
        if pd.isna(hadm):
            hadm_out = None
        else:
            try:
                hadm_out = int(hadm)
            except (TypeError, ValueError):
                hadm_out = str(hadm)
        notes_records.append({
            "anchor_date": None if pd.isna(r["anchor_date"]) else str(r["anchor_date"]),
            "anchor_category": r["anchor_category"],
            "anchor_hadm_id": hadm_out,
            "cos_sim_prev": _json_safe(r["cos_sim_prev"]),
            "l2_prev": _json_safe(r["l2_prev"]),
            "pca1": _json_safe(r["pca1"]) if "pca1" in df.columns else None,
            "pca2": _json_safe(r["pca2"]) if "pca2" in df.columns else None,
        })

    l2_valid = df["l2_prev"].iloc[1:] if len(df) > 1 else df["l2_prev"]
    spike_threshold = float(l2_valid.quantile(0.95)) if len(l2_valid) else 0.0

    return {
        "subject_id": int(req.subject_id),
        "notes": notes_records,
        "stats": {
            "n_notes": int(len(df)),
            "median_l2": float(l2_valid.median()) if len(l2_valid) else 0.0,
            "max_l2": float(df["l2_prev"].max()),
            "spike_threshold_p95": spike_threshold,
        },
    }


@router.post("/trajectory/interpret")
def trajectory_interpret_endpoint(req: TrajectoryInterpretRequest) -> dict:
    s = _require_corpus()
    from rag import interpret_trajectory  # type: ignore

    try:
        return interpret_trajectory(s.corpus, req.subject_id, openai_model=req.model)
    except ValueError as e:
        raise HTTPException(404, str(e))
