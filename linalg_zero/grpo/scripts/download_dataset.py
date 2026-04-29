from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rfvasile/LinalgZero-SFT-110",
    repo_type="model",
    local_dir="./downloaded_best_models",
    allow_patterns=["*"],
)
