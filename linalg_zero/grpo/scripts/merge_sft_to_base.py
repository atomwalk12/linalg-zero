import argparse

import torch
from huggingface_hub import hf_hub_download
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_sft_lora(
    sft_model_path: str,
    hub_repo_id: str,
    push_to_hub: bool = False,
    dtype: str = "bfloat16",
) -> None:
    """
    Merge a PEFT / LoRA SFT checkpoint into its base model and push directly to Hub.

    This is intended for taking an SFT checkpoint like `atomwalk12/LinalgZero-SFT`
    (trained with resized embeddings and special tokens) and producing a merged
    base model that can be used as the `run.base_model` for GRPO without relying
    on `modules_to_save` (which vLLM does not support for static LoRAs).
    """
    print(f"Loading PEFT config from: {sft_model_path}")
    peft_config = PeftConfig.from_pretrained(sft_model_path)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model from PEFT config: {base_model_name}")

    print("Loading tokenizer from SFT checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    tok_vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {tok_vocab_size}")

    # Inspect adapter weights to infer the embedding size expected by the LoRA
    # checkpoint. This is more reliable than using the tokenizer length because
    # Unsloth pads the vocab to a multiple of 128 when resizing embeddings.
    adapter_file = hf_hub_download(sft_model_path, "adapter_model.safetensors")

    adapter_state = load_file(adapter_file)
    embed_weight_key = None
    for key in adapter_state:
        if key.endswith("embed_tokens.original_module.weight"):
            embed_weight_key = key
            break
    if embed_weight_key is None:
        raise ValueError(
            "Could not find embed_tokens.original_module.weight in adapter state; cannot infer target embedding size."
        )
    target_embed_size = adapter_state[embed_weight_key].shape[0]
    print(f"Adapter expects embedding vocab size: {target_embed_size}")

    torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"

    print(f"Loading base model {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
    )

    # Ensure the base model's embedding matrix matches the adapter checkpoint.
    # The adapter was saved with embeddings padded to a multiple of 128, which
    # may not equal the raw tokenizer vocab length, so we match the adapter.
    current_vocab_size = base_model.get_input_embeddings().weight.size(0)
    if current_vocab_size != target_embed_size:
        print(f"Resizing base model embeddings from {current_vocab_size} to {target_embed_size} to match SFT adapter.")
        base_model.resize_token_embeddings(target_embed_size)
    else:
        print("Base model embeddings already match adapter embedding size.")

    print(f"Loading LoRA adapter from {sft_model_path} and attaching to base model...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        sft_model_path,
        is_trainable=False,
    )

    print("Merging LoRA weights into base model (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    if push_to_hub:
        print(f"Pushing merged model to Hugging Face Hub repo: https://huggingface.co/{hub_repo_id}")
        merged_model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)

        print(f"Done. You can now point GRPO `run.base_model` to: {hub_repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SFT LoRA checkpoint with base model and optionally push to Hugging Face Hub"
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="atomwalk12/LinAlgZero-SFT",
        help="Path or HF repo id of the SFT LoRA checkpoint",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default="atomwalk12/LinAlgZero-SFT-merged",
        help="Hugging Face Hub repo id to push the merged model to (e.g. 'atomwalk12/LinalgZero-SFT-merged')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype to load the base model with (e.g. 'bfloat16', 'float16', 'float32')",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=True,
        help="If set, push the merged model and tokenizer to the Hugging Face Hub",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_sft_lora(
        sft_model_path=args.sft_model_path,
        hub_repo_id=args.hub_repo_id,
        push_to_hub=args.push_to_hub,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
