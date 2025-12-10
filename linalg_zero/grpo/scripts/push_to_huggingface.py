import os

from huggingface_hub import HfApi

# Initialize the API
api = HfApi()

# Get the model's checkpoint directory
# ART stores models in ~/.art/{project}/models/{name}/
model_dir = ".art/linalgzero-grpo/models/run022-004-linalgzero-sft-110-checkpoint-300-forked-step-700/checkpoints/1100"
model_dir = os.path.expanduser(model_dir)

# Create a new repository on HuggingFace
repo_id = "atomwalk12/LinalgZero-GRPO"
api.create_repo(repo_id=repo_id, private=False, repo_type="model")

# Upload the LoRA adapter
api.upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    repo_type="model",
)
