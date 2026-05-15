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


SYSTEM_PROMPT_EVIDENCE = (
    "You are a clinical research assistant. Use the retrieved patient notes as the "
    "primary evidence to answer the user's question.\n\n"
    "Rules:\n"
    "- Ground every specific factual claim about the cases in the notes; cite as [Note N].\n"
    "- You may synthesize patterns across multiple notes (e.g., common interventions, "
    "recurring findings, typical disease progression seen in the cases).\n"
    "- You may add ONE brief sentence of general clinical context if it helps the reader "
    "interpret the cases, but prefix it with 'Context (general):' so it is clearly not "
    "drawn from the notes.\n"
    "- If the retrieved notes are clearly off-topic and provide no useful evidence, say "
    "'The retrieved notes are not relevant to this question' and stop. Do NOT speculate "
    "about cases the notes do not describe.\n"
    "- Do NOT give individual clinical advice. This is for research/education.\n\n"
    "Output structure:\n"
    "1. One short sentence directly answering the question.\n"
    "2. 2-4 bullet points of specific findings from the notes, each ending with [Note N] citations.\n"
    "3. Optional final line beginning 'Context (general):' if useful."
)

SYSTEM_PROMPT_STRICT = (
    "You are a clinical research assistant. Answer the user's question using ONLY "
    "the retrieved clinical notes provided. Cite specific Note IDs (e.g., 'Note 3') "
    "when referencing facts. If the retrieved notes do not contain the answer, "
    "say 'The retrieved notes do not answer this question' and do not speculate. "
    "Do not provide clinical advice; this is a research task."
)


def ask(
    corpus: CorpusIndex,
    question: str,
    k: int = 5,
    openai_model: str = "gpt-4o-mini",
    max_note_chars: int = 1500,
    mode: str = "evidence",
) -> dict:
    """
    Retrieval-augmented answer to a clinical question.

    Retrieves top-k notes, formats them with note IDs, and asks an OpenAI chat
    model to answer based on those notes.

    `mode` controls how strictly the LLM is constrained:
      - "evidence" (default): notes are the primary evidence, but the LLM may
        synthesize patterns across cases and add one flagged sentence of
        general context. More useful for exploratory questions.
      - "strict": LLM may only quote/cite facts directly present in the notes;
        refuses if the literal answer isn't in the retrieved text. Use for
        grounded-retrieval benchmarks.

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
            "mode": mode,
            "note": "OPENAI_API_KEY not set; returning retrieval only.",
        }

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    context_blocks = []
    for _, r in retrieved.iterrows():
        context_blocks.append(
            f"[Note {r['rank']}  similarity={r['similarity']:.3f}  patient={r['subject_id']}  "
            f"hadm={r['anchor_hadm_id']}  category={r['category']}  date={r['date']}  "
            f"icd={r['icd_codes'][:5]}]\n{r['snippet']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    if mode == "strict":
        system = SYSTEM_PROMPT_STRICT
    else:
        system = SYSTEM_PROMPT_EVIDENCE
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
        "mode": mode,
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


def _patient_chapters(corpus: CorpusIndex, subject_id: int) -> set[str]:
    """Distinct ICD-9 chapters a patient's admissions cover.

    Used by cohort-discovery utilities. Lifted out of find_similar_patients
    so the by-text and by-patient flows share it.
    """
    from preprocess import get_icd_chapter

    rows = corpus.meta[corpus.meta["subject_id"] == subject_id]
    chapters: set[str] = set()
    for hadm in rows["anchor_hadm_id"].dropna().unique():
        try:
            key = str(int(hadm))
        except (TypeError, ValueError):
            continue
        for c in corpus.icd_map.get(key, []):
            chapters.add(get_icd_chapter(c))
    return chapters


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

    seed_chap = _patient_chapters(corpus, subject_id)
    rows = []
    for j in order[:k + 1]:
        if int(sids[j]) == subject_id:
            continue
        other_chap = _patient_chapters(corpus, int(sids[j]))
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


def find_similar_patients_by_text(
    corpus: CorpusIndex,
    query: str,
    k: int = 10,
    pooled: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Free-text description -> cohort, via patient-level mean-pooled embeddings.

    Encodes the description with the same fine-tuned model used to build the
    corpus, then ranks patients by cosine similarity of their mean-pooled
    embedding to the query. Returns the top-k patients with their note count,
    ICD-9 chapters, and the seed-vs-patient cosine similarity.
    """
    if pooled is None:
        sids, patient_embs = patient_level_embeddings(corpus)
    else:
        sids, patient_embs = pooled

    q = corpus.encode(query)  # (1, D), L2-normalized
    sims = (patient_embs @ q[0]).astype(float)
    order = np.argsort(-sims)

    rows = []
    for j in order[:k]:
        sid = int(sids[j])
        n_notes = int((corpus.meta["subject_id"] == sid).sum())
        chapters = sorted(_patient_chapters(corpus, sid))
        rows.append({
            "rank": len(rows) + 1,
            "similarity": float(sims[j]),
            "subject_id": sid,
            "n_notes": n_notes,
            "chapters": chapters,
            "n_chapters": len(chapters),
        })
    return pd.DataFrame(rows)


def interpret_trajectory(
    corpus: CorpusIndex,
    subject_id: int,
    openai_model: str = "gpt-4o-mini",
    spike_quantile: float = 0.95,
    max_note_chars: int = 280,
) -> dict:
    """LLM-generated natural-language summary of a patient's trajectory.

    Builds a chronological timeline of the patient's anchor notes (date,
    category, L2 to previous embedding, truncated text), flags spike notes
    above the configured quantile, and asks an OpenAI chat model to produce
    a course summary, 1-3 inflection points, and a pattern label.

    Falls back to `answer=None` + a `note` when no API key is configured,
    matching the behaviour of `ask()`.
    """
    traj = patient_trajectory(corpus, subject_id)
    notes_df = traj["notes"].copy()

    l2 = notes_df["l2_prev"].to_numpy(dtype=float)
    valid_l2 = l2[1:] if len(l2) > 1 else l2
    spike_threshold = float(np.quantile(valid_l2, spike_quantile)) if len(valid_l2) else 0.0

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI-API-KEY")
    if not api_key:
        return {
            "subject_id": int(subject_id),
            "answer": None,
            "spike_threshold": spike_threshold,
            "note": "OPENAI_API_KEY not set; interpretation unavailable.",
        }

    timeline_lines = []
    for i, (_, r) in enumerate(notes_df.iterrows()):
        row_id = int(r["row_id"])
        text = corpus.meta.loc[row_id, "anchor_text"] or ""
        text = " ".join(text.split())[:max_note_chars]
        is_spike = (i > 0) and (float(r["l2_prev"]) > spike_threshold)
        tag = " [SPIKE]" if is_spike else ""
        timeline_lines.append(
            f"#{i} | {r['anchor_date']} | {r['anchor_category']} | "
            f"L2_prev={float(r['l2_prev']):.3f}{tag}\n  {text}"
        )
    timeline = "\n".join(timeline_lines)

    system = (
        "You are a clinical research assistant analyzing a patient's clinical "
        "embedding trajectory. Each note has an L2 distance to the previous "
        "note's embedding — large values (marked [SPIKE]) indicate semantic "
        "shifts (acuity change, transfer, new diagnosis, course reversal). "
        "Based ONLY on the supplied note timeline, produce:\n\n"
        "1. A 1-2 sentence summary of the patient's clinical course.\n"
        "2. 1-3 inflection points, each formatted as: '- Note #N (L2=X.XX) — short interpretation grounded in the note text.'\n"
        "3. An overall pattern label, exactly one of {stable, improving, declining, volatile, mixed}, with one short justification sentence.\n\n"
        "Output as markdown with three bold section headers exactly: **Course summary**, **Inflection points**, **Pattern**. "
        "Cite note indices as [#N]. Do not speculate beyond the notes. Do not provide individual clinical advice."
    )
    user_msg = (
        f"Patient {subject_id} | {len(notes_df)} notes | p95 spike threshold L2={spike_threshold:.3f}.\n\n"
        f"Timeline:\n{timeline}"
    )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    return {
        "subject_id": int(subject_id),
        "answer": resp.choices[0].message.content,
        "model": openai_model,
        "spike_threshold": spike_threshold,
        "n_notes": int(len(notes_df)),
    }


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

    # Deterministic ordering for reproducible trajectory statistics:
    # anchor_date is often day-level only, so we break ties with stable secondary keys.
    rows["row_id"] = rows.index
    rows["anchor_date_dt"] = pd.to_datetime(rows["anchor_date"], errors="coerce")
    rows = rows.sort_values(
        ["anchor_date_dt", "anchor_hadm_id", "anchor_category", "row_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    indices = rows["row_id"].values
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
        "notes": rows[[
            "anchor_date", "anchor_date_dt", "anchor_category", "anchor_hadm_id",
            "cos_sim_prev", "l2_prev", "pca1", "pca2", "row_id",
        ]],
        "embeddings": vecs,
    }


def _resolve_mimic_event_paths(
    mimic_root: Path | None = None,
) -> tuple[Path, Path]:
    """
    Resolve ADMISSIONS and ICUSTAYS CSV paths.

    Prefers explicit `mimic_root`; otherwise uses the project default
    `MIMIC -III (10000 patients)/`.
    """
    if mimic_root is None:
        mimic_root = ROOT / "MIMIC -III (10000 patients)"
    admissions_path = mimic_root / "ADMISSIONS" / "ADMISSIONS_sorted.csv"
    icu_path = mimic_root / "ICUSTAYS" / "ICUSTAYS_sorted.csv"
    if not admissions_path.exists():
        raise FileNotFoundError(f"Missing ADMISSIONS file: {admissions_path}")
    if not icu_path.exists():
        raise FileNotFoundError(f"Missing ICUSTAYS file: {icu_path}")
    return admissions_path, icu_path


def load_patient_events(
    subject_id: int,
    mimic_root: Path | None = None,
) -> pd.DataFrame:
    """
    Load admission and ICU boundary events for one patient.

    Returns a DataFrame with columns:
      - event_time (datetime64)
      - event_type (admit/discharge/icu_in/icu_out)
      - hadm_id (nullable int)
    """
    admissions_path, icu_path = _resolve_mimic_event_paths(mimic_root=mimic_root)

    adm = pd.read_csv(
        admissions_path,
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"],
        parse_dates=["ADMITTIME", "DISCHTIME"],
    )
    adm.columns = [c.lower() for c in adm.columns]
    adm = adm[adm["subject_id"] == int(subject_id)].copy()

    icu = pd.read_csv(
        icu_path,
        usecols=["SUBJECT_ID", "HADM_ID", "INTIME", "OUTTIME"],
        parse_dates=["INTIME", "OUTTIME"],
    )
    icu.columns = [c.lower() for c in icu.columns]
    icu = icu[icu["subject_id"] == int(subject_id)].copy()

    events = []
    for _, r in adm.iterrows():
        hadm = int(r["hadm_id"]) if pd.notna(r["hadm_id"]) else None
        if pd.notna(r["admittime"]):
            events.append({"event_time": r["admittime"], "event_type": "admit", "hadm_id": hadm})
        if pd.notna(r["dischtime"]):
            events.append({"event_time": r["dischtime"], "event_type": "discharge", "hadm_id": hadm})

    for _, r in icu.iterrows():
        hadm = int(r["hadm_id"]) if pd.notna(r["hadm_id"]) else None
        if pd.notna(r["intime"]):
            events.append({"event_time": r["intime"], "event_type": "icu_in", "hadm_id": hadm})
        if pd.notna(r["outtime"]):
            events.append({"event_time": r["outtime"], "event_type": "icu_out", "hadm_id": hadm})

    if not events:
        return pd.DataFrame(columns=["event_time", "event_type", "hadm_id"])
    out = pd.DataFrame(events).sort_values("event_time").reset_index(drop=True)
    return out


def evaluate_trajectory_event_alignment(
    corpus: CorpusIndex,
    subject_id: int,
    spike_quantile: float = 0.95,
    window_notes: int = 2,
    n_permutations: int = 1000,
    seed: int = 42,
    mimic_root: Path | None = None,
) -> dict:
    """
    Quantify whether embedding-velocity spikes align with clinical boundary events.

    Method:
      1) Compute note-level trajectory and define spikes as notes with l2_prev above
         the specified quantile.
      2) Load ADMISSIONS + ICUSTAYS events and map each event to its nearest note.
      3) A spike is "aligned" if within ±`window_notes` of any mapped event note.
      4) Compare observed alignment rate to a permutation baseline.
    """
    traj = patient_trajectory(corpus, subject_id)
    notes = traj["notes"].copy()
    notes["anchor_date_dt"] = pd.to_datetime(notes["anchor_date_dt"], errors="coerce")
    valid_notes = notes[notes["anchor_date_dt"].notna()].copy()
    if len(valid_notes) < 5:
        return {"error": "insufficient_notes_with_valid_datetimes"}

    events = load_patient_events(subject_id=subject_id, mimic_root=mimic_root)
    if len(events) == 0:
        return {"error": "no_admission_or_icu_events_found"}

    # Spike definition ignores first point (l2_prev=0 by construction).
    l2 = notes["l2_prev"].to_numpy(dtype=float)
    valid_l2 = l2[1:] if len(l2) > 1 else l2
    threshold = float(np.quantile(valid_l2, spike_quantile))
    spike_idx = np.where(l2 > threshold)[0]
    if len(spike_idx) == 0:
        return {"error": "no_spikes_found_for_quantile", "threshold": threshold}

    note_times = notes["anchor_date_dt"].tolist()
    event_note_idx = []
    for _, ev in events.iterrows():
        t = ev["event_time"]
        # nearest note in absolute time
        distances = [abs((nt - t).total_seconds()) if pd.notna(nt) else np.inf for nt in note_times]
        nearest = int(np.argmin(distances))
        event_note_idx.append(nearest)
    event_note_idx = np.array(sorted(set(event_note_idx)), dtype=int)

    def _match_rate(spikes: np.ndarray) -> float:
        if len(spikes) == 0:
            return 0.0
        matched = 0
        for s in spikes:
            if np.any(np.abs(event_note_idx - s) <= window_notes):
                matched += 1
        return float(matched / len(spikes))

    observed_rate = _match_rate(spike_idx)

    rng = np.random.RandomState(seed)
    eligible = np.arange(1, len(notes), dtype=int)  # exclude index 0
    perm_rates = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        sampled = rng.choice(eligible, size=len(spike_idx), replace=False)
        perm_rates[i] = _match_rate(np.sort(sampled))
    baseline_mean = float(np.mean(perm_rates))
    p_value = float((1.0 + np.sum(perm_rates >= observed_rate)) / (1.0 + n_permutations))
    lift = float(observed_rate / max(baseline_mean, 1e-8))

    return {
        "subject_id": int(subject_id),
        "n_notes": int(len(notes)),
        "n_events": int(len(events)),
        "event_types": events["event_type"].value_counts().to_dict(),
        "spike_quantile": float(spike_quantile),
        "window_notes": int(window_notes),
        "threshold_l2": threshold,
        "n_spikes": int(len(spike_idx)),
        "spike_indices": spike_idx.tolist(),
        "event_nearest_note_indices": event_note_idx.tolist(),
        "observed_spike_event_match_rate": observed_rate,
        "permutation_baseline_mean_rate": baseline_mean,
        "match_rate_lift": lift,
        "permutation_p_value_one_sided": p_value,
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
