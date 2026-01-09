from __future__ import annotations

from pathlib import Path

# ---- EDIT THESE ----
HUB_NAMESPACE = "atomwalk12"
PROJECT = "linalgzero-grpo"
EXPERIMENT = "runpod-grpo-006"
STEP = 1000
PRIVATE = False

REPO_ID = f"{HUB_NAMESPACE}/LinAlgZero-GRPO"
CHECKPOINT_DIR = Path(".art") / PROJECT / "models" / EXPERIMENT / "best_models" / "val" / f"{STEP:04d}"
# --------------------

IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/.DS_Store",
]


def main() -> None:
    if not CHECKPOINT_DIR.is_dir():
        raise SystemExit(f"Missing checkpoint dir: {CHECKPOINT_DIR}")

    try:
        from huggingface_hub import HfApi
    except Exception as e:
        raise SystemExit(f"Missing dependency: huggingface_hub ({e})")

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(CHECKPOINT_DIR),
        path_in_repo="",
        commit_message=f"Upload checkpoint step {STEP}",
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"Pushed {CHECKPOINT_DIR} -> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
