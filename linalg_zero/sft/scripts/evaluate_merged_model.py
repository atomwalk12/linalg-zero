import unsloth  # noqa: I001, F401
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from linalg_zero.sft.tool_calling_accuracy import ToolCallingAccuracyCallback


def load_unmerged():
    tokenizer = AutoTokenizer.from_pretrained("atomwalk12/LinalgZero-SFT")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    model, _ = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B",
        max_seq_length=8192,
        load_in_4bit=False,
        fast_inference=False,
    )
    print(f"Base model vocab size: {model.get_input_embeddings().weight.size(0)}")
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

    model = PeftModel.from_pretrained(
        model,
        "atomwalk12/LinalgZero-SFT",
        is_trainable=False,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def load_merged():
    checkpoint_path = "results/LinalgZero-SFT-merged-finetuned/checkpoint-400"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    model, tok2 = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=8192,
        load_in_4bit=False,
        fast_inference=False,
    )
    assert len(tok2) == len(tokenizer)

    FastLanguageModel.for_inference(model)

    return model, tokenizer


model, tokenizer = load_merged()

eval_ds = load_dataset("atomwalk12/linalgzero-sft", split="test")  # or whatever split you used

cb = ToolCallingAccuracyCallback(
    model_name="atomwalk12/LinAlgZero-SFT-merged",
    dataset_name="atomwalk12/linalgzero",
    eval_dataset=eval_ds,
)
gen_config = cb.get_generation_config(max_new_tokens=800, tokenizer=tokenizer)


def generate_like_sft_eval(sample_idx: int = 0):
    sample = eval_ds[sample_idx]
    context = list(sample["messages"])
    tools = sample["tools"]

    prompt = tokenizer.apply_chat_template(
        context,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = prompt

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tokenizer=tokenizer,
            **gen_config,
        )

    # Decode only the generated continuation (optional: mimic callback's decoding)
    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = outputs[:, prompt_len:]
    text = tokenizer.decode(gen_tokens[0], skip_special_tokens=False)
    print(text)
    return text


result = generate_like_sft_eval(1)
