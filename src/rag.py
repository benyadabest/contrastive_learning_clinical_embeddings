"""
Downstream applications on top of the fine-tuned MIMIC-III embeddings.

Four capabilities, all built on the same FAISS index over the 23,657 cached
anchor-note embeddings of the fine-tuned EmbeddingGemma model:

1. `search(query, k)` -> patient-record search by free-text query
2. `ask(question, k, openai_model)` -> retrieval-augmented generation with cited notes
3. `find_similar_patients(subject_id, k)` -> cohort discovery via mean-pooled
   patient-level embeddings, with Jaccard ICD-chapter overlap as a quantitative check
4. `patient_trajectory(subject_id)` -> per-patient note sequence with embedding-velocity
   and PCA projection for visualization

The default embedding source is the *hierarchical* fine-tuned variant (best on note
recall and AUROC); pass `model_safe_name="models_embeddinggemma_infonce_best"` to
swap to the InfoNCE variant or `"google_embeddinggemma_300m"` to use the vanilla
baseline as a control.

This module is designed to be importable from a notebook so the final demo can be
just a sequence of cell-level calls.
"""

from __future__ import annotations

# NOTE: OMP env vars must be set before faiss / torch / numpy are imported,
# otherwise faiss-cpu and PyTorch's bundled libomp.dylib can collide on Apple
# Silicon and silently segfault the process. See:
#   https://github.com/facebookresearch/faiss/issues/3079
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = ROOT / "embeddings"
RESULTS_DIR = ROOT / "results"

load_dotenv(ROOT / ".env")

DEFAULT_MODEL_SAFE_NAME = "models_embeddinggemma_hierarchical_best"
DEFAULT_HF_REPO = "gaspard-loeillot/embeddinggemma-mimic-hierarchical"
DEFAULT_LOCAL_MODEL_PATH = ROOT / "models" / "embeddinggemma_hierarchical_best"


@dataclass
class CorpusIndex:
    """Bundle of (FAISS index, metadata DataFrame, model handle).

    The DataFrame `meta` is row-aligned with the FAISS index: row i of `meta`
    corresponds to vector i in the index. `model_loader` is a zero-arg callable
    that lazily loads the SentenceTransformer (deferred until a query needs to
    be encoded; avoids paying the ~10s load cost when the user only wants
    similarity search over precomputed vectors).
    """
    index: object  # faiss.Index
    meta: pd.DataFrame
    embeddings: np.ndarray  # (N, D), L2-normalized float32
    model_loader: callable
    icd_map: dict[str, list[str]]
    model_safe_name: str
    _model_cache: list = None

    def encode(self, text: str | list[str]) -> np.ndarray:
        """Encode query text with the corpus's embedding model. L2-normalized."""
        if self._model_cache is None or self._model_cache[0] is None:
            print(f"[CorpusIndex] loading model for query encoding...")
            self._model_cache = [self.model_loader()]
        m = self._model_cache[0]
        embs = m.encode(
            [text] if isinstance(text, str) else text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        return embs


def _make_loader(local_path: Path | None, hf_repo: str | None):
    """Returns a callable that loads a SentenceTransformer when invoked.

    Prefers a local path if given; otherwise falls back to the HF Hub repo id.
    The double-indirection lets `build_index` finish quickly and only pay the
    load cost the first time someone runs a query.
    """
    def _load():
        from sentence_transformers import SentenceTransformer
        if local_path is not None and Path(local_path).exists():
            print(f"  loading from local: {local_path}")
            return SentenceTransformer(str(local_path))
        if hf_repo is not None:
            print(f"  loading from HF Hub: {hf_repo}")
            return SentenceTransformer(hf_repo)
        raise RuntimeError("No local model path or HF repo specified")
    return _load


def build_index(
    model_safe_name: str = DEFAULT_MODEL_SAFE_NAME,
    pairs_path: Path = DATA_DIR / "temporal_pairs_small.json",
    icd_map_path: Path = DATA_DIR / "icd_hierarchy.json",
    embeddings_dir: Path = EMBEDDINGS_DIR,
    local_model_path: Path | None = None,
    hf_repo: str | None = None,
) -> CorpusIndex:
    """
    Build a FAISS IP index over anchor embeddings, plus metadata DataFrame.

    Cached anchor embeddings are L2-normalized once at load time, so the
    subsequent `IndexFlatIP` searches return cosine similarities directly.
    """
    import faiss

    anchor_path = embeddings_dir / f"anchor_embeddings_{model_safe_name}.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"Missing anchor embeddings: {anchor_path}")

    print(f"[build_index] loading {anchor_path.name}...")
    embs = np.load(anchor_path).astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    embs = embs / norms
    print(f"  shape={embs.shape}  dtype={embs.dtype}  L2-normalized")

    print(f"[build_index] loading metadata...")
    with open(pairs_path) as f:
        pairs = json.load(f)
    if len(pairs) != embs.shape[0]:
        raise ValueError(
            f"Mismatch: {len(pairs)} pairs vs {embs.shape[0]} embeddings. "
            f"Check that {pairs_path.name} is the file that was embedded."
        )

    meta = pd.DataFrame([{
        "subject_id": p["subject_id"],
        "anchor_hadm_id": p.get("anchor_hadm_id"),
        "anchor_category": p.get("anchor_category") or "unknown",
        "anchor_date": p.get("anchor_date"),
        "anchor_text": p.get("anchor_text", ""),
    } for p in pairs])

    with open(icd_map_path) as f:
        icd_map = json.load(f)

    print(f"[build_index] building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    print(f"  index size={index.ntotal}")

    if local_model_path is None:
        local_model_path = DEFAULT_LOCAL_MODEL_PATH if model_safe_name == DEFAULT_MODEL_SAFE_NAME else None
    if hf_repo is None and model_safe_name == DEFAULT_MODEL_SAFE_NAME:
        hf_repo = DEFAULT_HF_REPO

    return CorpusIndex(
        index=index,
        meta=meta,
        embeddings=embs,
        model_loader=_make_loader(local_model_path, hf_repo),
        icd_map=icd_map,
        model_safe_name=model_safe_name,
    )


def search(
    corpus: CorpusIndex,
    query: str,
    k: int = 10,
    snippet_chars: int = 200,
) -> pd.DataFrame:
    """
    Return the top-k notes most similar to `query` as a DataFrame.

    Each row contains: similarity, subject_id, anchor_hadm_id, anchor_category,
    anchor_date, snippet (first `snippet_chars` chars of the note text), and
    icd_codes for the corresponding admission.
    """
    q = corpus.encode(query)
    sims, idx = corpus.index.search(q, k)
    sims = sims[0]
    idx = idx[0]

    rows = []
    for sim, i in zip(sims, idx):
        m = corpus.meta.iloc[i]
        hadm = str(int(m["anchor_hadm_id"])) if pd.notna(m["anchor_hadm_id"]) else None
        codes = corpus.icd_map.get(hadm, []) if hadm else []
        snippet = (m["anchor_text"] or "").strip().replace("\n", " ")[:snippet_chars]
        rows.append({
            "rank": len(rows) + 1,
            "similarity": float(sim),
            "subject_id": int(m["subject_id"]),
            "anchor_hadm_id": hadm,
            "category": m["anchor_category"],
            "date": m["anchor_date"],
            "icd_codes": codes,
            "snippet": snippet,
        })
    return pd.DataFrame(rows)


def ask(
    corpus: CorpusIndex,
    question: str,
    k: int = 5,
    openai_model: str = "gpt-4o-mini",
    max_note_chars: int = 1500,
) -> dict:
    """
    Retrieval-augmented answer to a clinical question.

    Retrieves top-k notes, formats them with note IDs, and asks an OpenAI chat
    model to answer using ONLY those notes (with explicit "I don't know" if
    the answer isn't grounded in retrieved text).

    If `OPENAI_API_KEY` (or `OPENAI-API-KEY`) is not set, returns the retrieved
    notes only with `answer=None`. This keeps the notebook runnable without an
    OpenAI key; the retrieval is the actual evaluation, the LLM is presentation.
    """
    retrieved = search(corpus, question, k=k, snippet_chars=max_note_chars)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI-API-KEY")
    if not api_key:
        return {
            "question": question,
            "retrieved": retrieved,
            "answer": None,
            "note": "OPENAI_API_KEY not set; returning retrieval only.",
        }

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    context_blocks = []
    for _, r in retrieved.iterrows():
        context_blocks.append(
            f"[Note {r['rank']}  patient={r['subject_id']}  hadm={r['anchor_hadm_id']}  "
            f"category={r['category']}  date={r['date']}  icd={r['icd_codes'][:5]}]\n"
            f"{r['snippet']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a clinical research assistant. Answer the user's question using ONLY "
        "the retrieved clinical notes provided. Cite specific Note IDs (e.g., 'Note 3') "
        "when referencing facts. If the retrieved notes do not contain the answer, "
        "say 'The retrieved notes do not answer this question' and do not speculate. "
        "Do not provide clinical advice; this is a research task."
    )
    user_msg = f"Question: {question}\n\nRetrieved notes:\n\n{context}"

    resp = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    return {
        "question": question,
        "retrieved": retrieved,
        "answer": resp.choices[0].message.content,
        "model": openai_model,
    }


def patient_level_embeddings(corpus: CorpusIndex) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-pooled per-patient embeddings.

    Returns (subject_ids, patient_embs) where row i of patient_embs is the L2-
    normalized mean of all anchor embeddings for subject_ids[i].
    """
    df = corpus.meta.copy()
    df["row"] = np.arange(len(df))
    pooled = []
    sids = []
    for sid, grp in df.groupby("subject_id"):
        vecs = corpus.embeddings[grp["row"].values]
        v = vecs.mean(axis=0)
        v = v / max(np.linalg.norm(v), 1e-8)
        pooled.append(v)
        sids.append(int(sid))
    return np.asarray(sids), np.asarray(pooled, dtype="float32")


def find_similar_patients(
    corpus: CorpusIndex,
    subject_id: int,
    k: int = 10,
    pooled: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Patient cohort discovery via patient-level embedding cosine similarity.

    Reports per-pair Jaccard overlap of ICD-9 chapters as a quantitative sanity
    check — for two patients to be "similar" in the embedding space we'd want
    their diagnosis chapters to overlap above the random baseline.
    """
    from preprocess import get_icd_chapter

    if pooled is None:
        sids, patient_embs = patient_level_embeddings(corpus)
    else:
        sids, patient_embs = pooled

    if subject_id not in sids:
        raise ValueError(f"subject_id {subject_id} not in corpus")
    seed_idx = int(np.where(sids == subject_id)[0][0])
    seed_vec = patient_embs[seed_idx]

    sims = patient_embs @ seed_vec
    order = np.argsort(-sims)

    def _patient_chapters(sid: int) -> set[str]:
        rows = corpus.meta[corpus.meta["subject_id"] == sid]
        chapters: set[str] = set()
        for hadm in rows["anchor_hadm_id"].dropna().unique():
            try:
                key = str(int(hadm))
            except (TypeError, ValueError):
                continue
            for c in corpus.icd_map.get(key, []):
                chapters.add(get_icd_chapter(c))
        return chapters

    seed_chap = _patient_chapters(subject_id)
    rows = []
    for j in order[:k + 1]:
        if int(sids[j]) == subject_id:
            continue
        other_chap = _patient_chapters(int(sids[j]))
        union = seed_chap | other_chap
        inter = seed_chap & other_chap
        jaccard = len(inter) / len(union) if union else 0.0
        rows.append({
            "rank": len(rows) + 1,
            "similarity": float(sims[j]),
            "subject_id": int(sids[j]),
            "shared_chapters": sorted(inter),
            "n_shared": len(inter),
            "jaccard_chapter": jaccard,
        })
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


def patient_trajectory(
    corpus: CorpusIndex,
    subject_id: int,
) -> dict:
    """
    For one patient, return an ordered DataFrame of their notes along with:
      - per-note embedding (L2-normalized)
      - cosine similarity to the previous note (1.0 for the first)
      - L2 distance to the previous note (0.0 for the first; spikes are
        candidate inflection points / acuity changes)
      - 2-D PCA projection of the patient's note embeddings for plotting
    """
    rows = corpus.meta[corpus.meta["subject_id"] == subject_id].copy()
    if len(rows) < 2:
        raise ValueError(f"subject_id {subject_id} has <2 anchor notes; trajectory undefined.")
    rows = rows.sort_values("anchor_date").reset_index()
    indices = rows["index"].values
    vecs = corpus.embeddings[indices]

    cos_sim_prev = np.empty(len(vecs))
    l2_prev = np.empty(len(vecs))
    cos_sim_prev[0] = 1.0
    l2_prev[0] = 0.0
    for i in range(1, len(vecs)):
        cos_sim_prev[i] = float(vecs[i] @ vecs[i - 1])
        l2_prev[i] = float(np.linalg.norm(vecs[i] - vecs[i - 1]))

    rows["cos_sim_prev"] = cos_sim_prev
    rows["l2_prev"] = l2_prev

    if len(vecs) >= 2:
        from sklearn.decomposition import PCA
        n_comp = min(2, len(vecs), vecs.shape[1])
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(vecs)
        if coords.shape[1] == 1:
            coords = np.hstack([coords, np.zeros_like(coords)])
        rows["pca1"] = coords[:, 0]
        rows["pca2"] = coords[:, 1]

    return {
        "subject_id": int(subject_id),
        "notes": rows[["anchor_date", "anchor_category", "anchor_hadm_id",
                       "cos_sim_prev", "l2_prev", "pca1", "pca2"]],
        "embeddings": vecs,
    }


def evaluate_cohort_discovery(
    corpus: CorpusIndex,
    n_seeds: int = 50,
    k: int = 5,
    n_random_pairs: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Quantitative sanity check on cohort discovery:

    For `n_seeds` random patients, compute the mean Jaccard chapter overlap
    between seed and top-k nearest neighbors (by patient embedding cosine).
    Compare against the mean Jaccard overlap for `n_random_pairs` random
    patient pairs. A meaningful signal => neighbors-mean significantly above
    random-mean.

    Reports both means plus a one-sided Mann-Whitney U test.
    """
    from preprocess import get_icd_chapter
    from scipy.stats import mannwhitneyu

    rng = np.random.RandomState(seed)
    sids, patient_embs = patient_level_embeddings(corpus)
    sid_to_idx = {int(s): i for i, s in enumerate(sids)}

    def _chap(sid):
        chapters = set()
        for hadm in corpus.meta[corpus.meta["subject_id"] == sid]["anchor_hadm_id"].dropna().unique():
            try:
                key = str(int(hadm))
            except (TypeError, ValueError):
                continue
            for c in corpus.icd_map.get(key, []):
                chapters.add(get_icd_chapter(c))
        return chapters

    chap_cache = {int(s): _chap(int(s)) for s in sids}

    seeds = rng.choice(sids, size=min(n_seeds, len(sids)), replace=False)
    neighbor_jaccards = []
    for s in seeds:
        seed_chap = chap_cache[int(s)]
        if not seed_chap:
            continue
        seed_vec = patient_embs[sid_to_idx[int(s)]]
        sims = patient_embs @ seed_vec
        order = np.argsort(-sims)
        # skip self (index 0)
        for j in order[1:k + 1]:
            other_chap = chap_cache[int(sids[j])]
            union = seed_chap | other_chap
            if not union:
                continue
            neighbor_jaccards.append(len(seed_chap & other_chap) / len(union))

    random_jaccards = []
    for _ in range(n_random_pairs):
        i, j = rng.choice(len(sids), size=2, replace=False)
        ci, cj = chap_cache[int(sids[i])], chap_cache[int(sids[j])]
        union = ci | cj
        if not union:
            continue
        random_jaccards.append(len(ci & cj) / len(union))

    if not neighbor_jaccards or not random_jaccards:
        return {"error": "insufficient data for cohort eval"}

    u, p = mannwhitneyu(
        neighbor_jaccards, random_jaccards, alternative="greater",
    )
    return {
        "n_seeds": int(len(seeds)),
        "k": int(k),
        "n_random_pairs": int(len(random_jaccards)),
        "neighbor_mean_jaccard": float(np.mean(neighbor_jaccards)),
        "neighbor_median_jaccard": float(np.median(neighbor_jaccards)),
        "random_mean_jaccard": float(np.mean(random_jaccards)),
        "random_median_jaccard": float(np.median(random_jaccards)),
        "lift_mean": float(np.mean(neighbor_jaccards) / max(np.mean(random_jaccards), 1e-8)),
        "mannwhitneyu_one_sided_p": float(p),
    }
