# Dockerfile for the FastAPI frontend on Hugging Face Spaces (Docker SDK).
#
# Build-time strategy:
#   1) Install Python deps from requirements.txt
#   2) Copy the application source (src/, scripts/, data/)
#   3) Regenerate the synthetic demo corpus at build time so we never have to
#      ship a binary .npy file through git/LFS. The seed is deterministic, so
#      the embeddings + metadata are reproducible across builds.
#
# The Space defaults to the synthetic corpus + sentence-transformers/all-MiniLM-L6-v2.
# Override EMBEDDINGS_MODEL_SAFE_NAME / PAIRS_FILENAME / ICD_MAP_FILENAME / HF_REPO
# in the Space's Variables tab to point at a different corpus (e.g. the real
# MIMIC pipeline) once those artefacts are mounted into the image.
#
# OPENAI_API_KEY is read at runtime by rag.py — set it as a Space *Secret*,
# not a Variable, and never bake it into the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as non-root uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user scripts ./scripts
COPY --chown=user data ./data

# Generate the synthetic demo fixture (writes the .npy + regenerates the
# committed .json files deterministically). ~80 MB MiniLM download + ~3 s encode.
RUN mkdir -p embeddings && python scripts/make_synthetic_corpus.py

# Default the runtime to the synthetic corpus + small MiniLM query encoder.
# Override in the Space's Variables tab for other corpora.
ENV EMBEDDINGS_MODEL_SAFE_NAME=synthetic \
    PAIRS_FILENAME=temporal_pairs_synthetic.json \
    ICD_MAP_FILENAME=icd_hierarchy_synthetic.json \
    HF_REPO=sentence-transformers/all-MiniLM-L6-v2

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
