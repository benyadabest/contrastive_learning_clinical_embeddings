"""Shared API state.

The CorpusIndex is heavy to construct (~10s for model handle + FAISS build),
so it is built once at app startup by the lifespan hook in main.py and
stashed on this module-level singleton. Patient-level mean-pooled embeddings
are cached on first /similar-patients request and reused thereafter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class CorpusState:
    def __init__(self) -> None:
        self.corpus: object | None = None
        self.model_safe_name: str = ""
        self._patient_cache: Optional[tuple[np.ndarray, np.ndarray]] = None

    def patient_pooled(self) -> tuple[np.ndarray, np.ndarray]:
        if self._patient_cache is None:
            from rag import patient_level_embeddings  # type: ignore
            self._patient_cache = patient_level_embeddings(self.corpus)
        return self._patient_cache


state = CorpusState()


def get_state() -> CorpusState:
    return state
