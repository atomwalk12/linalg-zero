export HF_HOME=/workspace/linalg-zero/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/linalg-zero/hf-cache
export VLLM_WORKDIR=/workspace/linalg-zero/vllm-cache
export UV_CACHE_DIR=/workspace/linalg-zero/uv-cache
export XDG_CACHE_HOME=/workspace/linalg-zero/.cache
export PIP_CACHE_DIR=/workspace/linalg-zero/pip-cache

export OPENPIPE_API_KEY=opk_9709401450b9f9a4b026db3790d7b2c052c69c2e17
export WANDB_API_KEY=dc1474f287b50e89b54d4b2d6426b2c31353a763
export VLLM_API_KEY="my-secret-key"
export ACCELERATE_MIXED_PRECISION=bf16
export VLLM_NO_USAGE_STATS=1

# VLLM_NO_USAGE_STATS=1 ACCELERATE_MIXED_PRECISION=bf16 uv run linalg_zero/grpo/tau-bench/run_linalg.py
export HF_HUB_NAMESPACE="atomwalk12"
export HF_TOKEN="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"
export HF_REPO_PRIVATE=1
