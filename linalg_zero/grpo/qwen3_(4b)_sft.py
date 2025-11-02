import os

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # [NEW] Extra 30% context lengths!

from trl import GRPOTrainer
from unsloth import FastLanguageModel

# # Replace GRPOTrainer with patched version
# GRPOTrainer = PatchedGRPOTrainer
max_seq_length = 2048  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    enforce_eager=True,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Disable vLLM for SFT training (only needed for GRPO)
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,  # Reduce if out of memory
)


model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,  # *2 speeds up training
    use_gradient_checkpointing="unsloth",  # Reduces memory usage
    random_state=3407,
)

reasoning_start = "<start_working_out>"  # Acts as <think>
reasoning_end = "<end_working_out>"  # Acts as </think>
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
system_prompt


chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '{system_prompt}' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
    "{% endif %}"
)

# Replace with out specific template:
chat_template = chat_template.replace("'{system_prompt}'", f"'{system_prompt}'").replace(
    "'{reasoning_start}'", f"'{reasoning_start}'"
)
tokenizer.chat_template = chat_template

"""Let's see how our chat template behaves on an example:"""

tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}",
        },
        {"role": "user", "content": "What is 2+2?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

import numpy as np
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]

# Try converting to number - if not, replace with NaN
is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
# Select only numbers
dataset = dataset.iloc[np.where(is_number)[0]]

print(dataset.head())

"""We have to format the dataset to follow our GRPO style formatting:"""


def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    thoughts = thoughts.strip()
    # Add our custom formatting
    final_prompt = reasoning_start + thoughts + reasoning_end + solution_start + expected_answer + solution_end
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]


dataset["Messages"] = dataset.apply(format_dataset, axis=1)

"""Check to see if it worked:"""

tokenizer.apply_chat_template(dataset["Messages"][0], tokenize=False)


dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))

dataset = dataset.loc[dataset["N"] <= max_seq_length / 2].copy()
dataset.shape

from datasets import Dataset

dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize=False)
dataset = Dataset.from_pandas(dataset)


# Save the model immediately after training

print("Training completed successfully!")

# Force clean exit to avoid vLLM CUDA allocator cleanup issues
# This is necessary when using fast_inference=True with enforce_eager=True

import torch

del dataset
torch.cuda.empty_cache()
import gc

gc.collect()


from datasets import load_dataset

dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")


def extract_hash_answer(text):
    # if "####" not in text: return None
    # return text.split("####")[1].strip()
    return text


extract_hash_answer(dataset[0]["solution"])

"""Let's map the dataset! and see the first row:"""

dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    },
    remove_columns=dataset.column_names,
)

"""We create a regex format to match the reasoning sections and answers:"""

import re

# Add optional EOS token matching
solution_end_regex = r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_format

"""We verify it works:"""

match_format.findall(
    "Let me think!<end_working_out><SOLUTION>\n2\n</SOLUTION>",
)

match_format.findall(
    "<start_working_out>Let me think!<end_working_out><SOLUTION>  2  </SOLUTION>\n\n",
)

"""We now want to create a reward function to match the format exactly - we reward it with 3 points if it succeeds:"""


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


"""If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:"""


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!

        # No need to reward <start_working_out> since we always prepend it!
        # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


"""Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:"""


def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer, strict=False):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        # Correct answer gets 5 points!
        if guess == true_answer:
            score += 5.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5  # Penalize wrong answers
            except:
                score -= 4.5  # Penalize
        scores.append(score)
    return scores


"""Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.

We also remove possible commas for example as in 123,456
"""

match_numbers = re.compile(solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL)
print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>  -0.234  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>17</SOLUTION>"))

"""We now prepare our main function which will print out the generated responses and the true answer, along with another reward function which converts text to float via `float` and sees if it's the same."""

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5


def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None for r in responses
    ]

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            "*" * 20 + f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer, strict=False):
        if guess is None:
            scores.append(-2.5)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess == true_answer else -1.5)
        except:
            scores.append(0)
            continue
    return scores


"""Get the top 90% prompt length so we don't accidentally truncate them!

Ie we'll remove the top 10% long prompts.
"""

tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)
print(tokenizer.decode(tokenized[0]["tokens"]))
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

import numpy as np

maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

# Filter only samples smaller than 90% max length
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

"""<a name="Train"></a>
### Train the model

Now set up GRPO Trainer and all configurations!
"""

max_prompt_length = maximum_length + 1  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

from trl import GRPOConfig

# GRPOTrainer already imported and patched at the top of the file
training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=100,
    save_steps=100,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

"""And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!

| Step | Training Loss | reward    | reward_std | completion_length | kl       |
|------|---------------|-----------|------------|-------------------|----------|
| 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
| 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
| 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |

"""

# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)


trainer.train()

os._exit(0)
