"""Pydantic request schemas for the API.

Response shapes are kept as plain dicts (built per-route from pandas
DataFrames) rather than declared models, so the same JSON shape that the
notebook prints stays untouched.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=50)
    snippet_chars: int = Field(300, ge=50, le=2000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)
    model: str = "gpt-4o-mini"
    mode: Literal["evidence", "strict"] = "evidence"


class SimilarPatientsRequest(BaseModel):
    subject_id: int
    k: int = Field(10, ge=1, le=50)


class SimilarPatientsByTextRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=50)


class TrajectoryRequest(BaseModel):
    subject_id: int


class TrajectoryInterpretRequest(BaseModel):
    subject_id: int
    model: str = "gpt-4o-mini"
