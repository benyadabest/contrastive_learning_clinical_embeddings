"""
Push fine-tuned SentenceTransformer checkpoints to Hugging Face Hub.

Reads HUGGINGFACE_TOKEN and HF_USERNAME from .env (gitignored). Creates the
target repos (private=False) if they don't exist, then uploads the local
checkpoint folders. The local README.md (model card) is uploaded as the
repo's README, so the hand-written content shows up on the model page.

Usage:
    python src/push_to_hub.py --model infonce
    python src/push_to_hub.py --model hierarchical
    python src/push_to_hub.py --model both
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

load_dotenv(ROOT / ".env")

LOCAL_TO_REPO = {
    "infonce": ("embeddinggemma_infonce_best", "embeddinggemma-mimic-infonce"),
    "hierarchical": ("embeddinggemma_hierarchical_best", "embeddinggemma-mimic-hierarchical"),
}


def push(local_name: str, repo_name: str, username: str, token: str) -> str:
    """Push a single checkpoint folder to the user's HF account."""
    repo_id = f"{username}/{repo_name}"
    local_path = MODELS_DIR / local_name
    if not local_path.exists():
        raise FileNotFoundError(f"Missing checkpoint folder: {local_path}")

    print(f"\n=== {repo_id} ===")
    print(f"Local: {local_path}")

    create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False, repo_type="model")
    print(f"Repo ready: https://huggingface.co/{repo_id}")

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Initial upload of {local_name}",
        ignore_patterns=[".DS_Store", "__pycache__/*", "*.pyc"],
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded -> {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Push fine-tuned checkpoints to HF Hub")
    parser.add_argument("--model", choices=["infonce", "hierarchical", "both"], default="both")
    args = parser.parse_args()

    token = os.getenv("HUGGINGFACE_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token:
        raise SystemExit("HUGGINGFACE_TOKEN missing from .env")
    if not username:
        raise SystemExit("HF_USERNAME missing from .env")

    targets = ["infonce", "hierarchical"] if args.model == "both" else [args.model]
    urls: list[str] = []
    for key in targets:
        local_name, repo_name = LOCAL_TO_REPO[key]
        urls.append(push(local_name, repo_name, username, token))

    print("\nAll uploads complete:")
    for u in urls:
        print(f"  {u}")


if __name__ == "__main__":
    main()
