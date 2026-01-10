import argparse
import asyncio
import concurrent.futures
import copy
import json
import logging
import math
import os
import re
import shutil
import statistics
import traceback
from collections import Counter
from collections.abc import Generator
from pathlib import Path
from typing import Any

import art
from art.local import LocalBackend
from art.utils import iterate_dataset, limit_concurrency
from art.utils.iterate_dataset import DatasetBatch
from art.utils.output_dirs import get_default_art_path, get_model_dir, get_step_checkpoint_dir
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import wandb
from linalg_zero.grpo.agents.tool_calling_agent import ToolCallingRLAgent
from linalg_zero.grpo.envs import get_env
from linalg_zero.grpo.general_rm import calculate_reward, create_general_rm_trajectory_groups
from linalg_zero.grpo.rl_utils import (
    log_trajectory_to_openpipe,
)
from linalg_zero.grpo.run import agent_factory
from linalg_zero.grpo.task_selection import (
    ShuffleBagSampler,
    ToolCallsMixtureSampler,
    get_task_indices,
)
from linalg_zero.grpo.types import RunConfig, SolveResult, TauBenchPolicyConfig

# Load environment variables
load_dotenv(override=True)

# Suppress LiteLLM logging spam
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

BEST_CHECKPOINT_METRIC = "val/optimal_trajectory"


_COVERAGE_LOG_MAX_TOOL_CALLS_BUCKET = 3
_COVERAGE_PRINT_EVERY_STEPS = 50

_HF_REPO_NAME_ALLOWED = re.compile(r"[^A-Za-z0-9_.-]+")
_HF_UPLOAD_IGNORE_PATTERNS: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/.DS_Store",
)


async def _delete_checkpoints_keep_best(model: art.TrainableModel[TauBenchPolicyConfig]) -> None:
    try:
        await model.delete_checkpoints(best_checkpoint_metric=BEST_CHECKPOINT_METRIC)
    except Exception as e:
        print(f"Warning: delete_checkpoints failed for metric '{BEST_CHECKPOINT_METRIC}': {e}")
        await model.delete_checkpoints()


def _archive_checkpoint(*, model: art.Model[TauBenchPolicyConfig], step: int, split: str) -> None:
    """
    Copy the checkpoint directory for `step` into a persistent archive directory under the model output dir.

    This preserves all checkpoints that were evaluated, even if we later prune the main `checkpoints/` directory.
    """
    try:
        art_path = get_default_art_path()
        model_dir = get_model_dir(model=model, art_path=art_path)
        src = get_step_checkpoint_dir(model_dir, step)
        if not os.path.isdir(src):
            print(f"Warning: checkpoint dir not found for archiving: {src}")
            return

        dst_base = os.path.join(model_dir, "best_models", split)
        os.makedirs(dst_base, exist_ok=True)
        dst = os.path.join(dst_base, f"{step:04d}")
        if os.path.exists(dst):
            return
        shutil.copytree(src, dst)
    except Exception as e:
        print(f"Warning: failed to archive checkpoint at step {step} ({split}): {e}")


def _sanitize_hf_repo_name(name: str) -> str:
    name = _HF_REPO_NAME_ALLOWED.sub("-", name).strip("-.")
    name = re.sub(r"-{2,}", "-", name)
    if not name:
        raise ValueError("Sanitized repo name is empty.")
    return name


def _should_push_experiment_to_hub() -> bool:
    # Explicit opt-out even if namespace is set.
    if os.environ.get("HF_PUSH_EXPERIMENT", "").strip().lower() in {"0", "false", "no"}:
        return False
    return bool(os.environ.get("HF_HUB_NAMESPACE"))


def _get_rank() -> int:
    for key in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            return 0
    return 0


def _push_experiment_dir_to_hf_sync(*, model: art.Model[TauBenchPolicyConfig]) -> None:
    """
    Upload `.art/<project>/models/<experiment>/` to the HF Hub.

    Enabled when `HF_HUB_NAMESPACE` is set and not explicitly disabled via `HF_PUSH_EXPERIMENT=0`.
    """
    if not _should_push_experiment_to_hub():
        print("[hf] Skipping upload: set `HF_HUB_NAMESPACE` to enable post-training push.")
        return

    if _get_rank() != 0:
        print("[hf] Skipping upload on non-zero rank.")
        return

    namespace = os.environ.get("HF_HUB_NAMESPACE")
    assert namespace is not None

    art_path = get_default_art_path()
    experiment_dir = Path(get_model_dir(model=model, art_path=art_path))
    if not experiment_dir.is_dir():
        print(f"[hf] Skipping upload: experiment dir not found: {experiment_dir}")
        return

    project = model.config.run_config.project
    experiment = model.name
    repo_name = _sanitize_hf_repo_name(f"{project}--{experiment}--experiment")
    repo_id = f"{namespace}/{repo_name}"

    private = os.environ.get("HF_REPO_PRIVATE", "").strip().lower() in {"1", "true", "yes"}

    try:
        from huggingface_hub import HfApi
    except Exception as e:  # pragma: no cover
        print("[hf] Skipping upload: missing dependency `huggingface_hub`.")
        print(f"[hf] Import error: {e}")
        return

    print(f"[hf] Uploading experiment dir: {experiment_dir} -> https://huggingface.co/{repo_id}")
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(experiment_dir),
        path_in_repo="",
        commit_message=f"Upload {project}/{experiment} experiment directory",
        ignore_patterns=list(_HF_UPLOAD_IGNORE_PATTERNS),
    )
    print("[hf] Upload complete.")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * q))
    return float(values_sorted[idx])


class _HistoryTail:
    def __init__(self, path: str) -> None:
        self._path = path
        self._pos = 0

    def read_new(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                f.seek(self._pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                self._pos = f.tell()
        except FileNotFoundError:
            return rows
        return rows


def _messages_and_choices_to_messages(messages_and_choices: art.MessagesAndChoices) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in messages_and_choices:
        if hasattr(item, "message"):  # openai Choice-like
            messages.append(item.message.model_dump())  # type: ignore[attr-defined]
        else:
            messages.append(item)  # type: ignore[arg-type]
    return messages


def _extract_tool_name_sequence(traj: art.Trajectory) -> tuple[str, ...]:
    if not traj.messages_and_choices:
        return ()
    messages = _messages_and_choices_to_messages(traj.messages_and_choices)
    tool_names: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tool_call in tool_calls:
            fn = tool_call.get("function") or {}
            name = fn.get("name")
            if isinstance(name, str) and name:
                tool_names.append(name)
    return tuple(tool_names)


class _CurriculumCoverageTracker:
    """
    Per-step task coverage logging.

    Tracks which task indices have been sampled so far, broken down by a simple difficulty proxy:
    the number of teacher tool calls in the task (`len(task.actions)`).

    Intended for debugging curriculum exposure (e.g., ensuring harder tasks actually show up),
    not as a training signal.
    """

    def __init__(self, *, tool_calls_by_index: dict[int, int], max_bucket_to_log: int) -> None:
        self._tool_calls_by_index = dict(tool_calls_by_index)
        self._max_bucket_to_log = int(max_bucket_to_log)
        self._seen: set[int] = set()
        self._seen_unique_by_bucket: Counter[int] = Counter()
        self._total_by_bucket: Counter[int] = Counter(self._tool_calls_by_index.values())
        self._total = len(self._tool_calls_by_index)

    def _bucket_key(self, tool_calls: int) -> int:
        if tool_calls <= self._max_bucket_to_log:
            return tool_calls
        return self._max_bucket_to_log + 1

    def _record_sampled_indices(self, *, sampled_indices: list[int]) -> Counter[int]:
        sampled_by_bucket: Counter[int] = Counter()
        for idx in sampled_indices:
            tool_calls = self._tool_calls_by_index.get(idx)
            if tool_calls is None:
                continue
            bucket = self._bucket_key(tool_calls)
            sampled_by_bucket[bucket] += 1
            if idx in self._seen:
                continue
            self._seen.add(idx)
            self._seen_unique_by_bucket[bucket] += 1
        return sampled_by_bucket

    def advance(self, *, sampled_indices: list[int]) -> None:
        """Update internal coverage state without emitting metrics (used for resuming)."""
        self._record_sampled_indices(sampled_indices=sampled_indices)

    def update(self, *, step: int, sampled_indices: list[int]) -> dict[str, float]:
        sampled_by_bucket = self._record_sampled_indices(sampled_indices=sampled_indices)

        metrics: dict[str, float] = {
            "train/curriculum_seen_unique_total": float(len(self._seen)),
            "train/curriculum_seen_frac_total": float(len(self._seen) / max(1, self._total)),
        }

        for bucket in range(0, self._max_bucket_to_log + 2):
            if bucket == self._max_bucket_to_log + 1:
                suffix = f"{self._max_bucket_to_log + 1}_plus"
                denom = sum(v for k, v in self._total_by_bucket.items() if k > self._max_bucket_to_log)
            else:
                suffix = str(bucket)
                denom = self._total_by_bucket.get(bucket, 0)

            metrics[f"train/curriculum_batch_tool_calls_{suffix}"] = float(sampled_by_bucket.get(bucket, 0))
            metrics[f"train/curriculum_seen_unique_tool_calls_{suffix}"] = float(
                self._seen_unique_by_bucket.get(bucket, 0)
            )
            metrics[f"train/curriculum_seen_frac_tool_calls_{suffix}"] = float(
                self._seen_unique_by_bucket.get(bucket, 0) / max(1, denom)
            )

        if step % _COVERAGE_PRINT_EVERY_STEPS == 0:
            hard_bucket = self._max_bucket_to_log + 1
            hard_total = sum(v for k, v in self._total_by_bucket.items() if k > self._max_bucket_to_log)
            print(
                "[coverage] "
                f"step={step} seen={len(self._seen)}/{self._total} "
                f"batch(tool_calls>={self._max_bucket_to_log + 1})={sampled_by_bucket.get(hard_bucket, 0)} "
                f"seen(tool_calls>={self._max_bucket_to_log + 1})={self._seen_unique_by_bucket.get(hard_bucket, 0)}/{hard_total}"
            )

        return metrics


def _prefill_coverage_tracker(
    *,
    coverage: _CurriculumCoverageTracker,
    initial_step: int,
    train_task_indices: list[int],
    tasks: list[Any],
    config: RunConfig,
    training_config: Any,
) -> None:
    """
    When resuming training mid-run, W&B logging uses the global step index, but the
    in-memory coverage tracker resets on process restart.

    To keep `train/curriculum_seen_*` continuous across restarts, replay the deterministic
    sampler for steps < `initial_step` and update coverage state without logging.
    """
    if initial_step <= 0:
        return

    if config.curriculum is not None and config.curriculum.enabled:
        prefill_iter = _iterate_curriculum(
            base_epoch_size=len(train_task_indices),
            groups_per_step=training_config.groups_per_step,
            num_epochs=training_config.num_epochs,
            initial_step=0,
            tasks=tasks,
            config=config,
            seed=config.seed,
            use_tqdm=False,
        )
    else:
        prefill_iter = iterate_dataset(
            train_task_indices,
            groups_per_step=training_config.groups_per_step,
            num_epochs=training_config.num_epochs,
            initial_step=0,
            use_tqdm=False,
        )

    for batch in prefill_iter:
        if batch.step >= initial_step:
            break
        coverage.advance(sampled_indices=list(batch.items))


def _log_group_diversity(
    *,
    step: int,
    groups: list[art.TrajectoryGroup],
    split: str,
) -> None:
    if not groups:
        return

    group_reward_stds: list[float] = []
    group_first_tool_diversity: list[float] = []
    first_tool_names: list[str] = []

    for group in groups:
        rewards = [float(traj.reward) for traj in group.trajectories]
        group_reward_stds.append(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0)

        first_tools: list[str] = []
        for traj in group.trajectories:
            seq = _extract_tool_name_sequence(traj)
            if seq:
                first_tools.append(seq[0])
        if first_tools:
            first_tool_names.extend(first_tools)
            group_first_tool_diversity.append(len(set(first_tools)) / len(first_tools))
        else:
            group_first_tool_diversity.append(0.0)

    top_first_tools = Counter(first_tool_names).most_common(5)
    reward_std_p95 = (
        float(statistics.quantiles(group_reward_stds, n=20)[-1])
        if len(group_reward_stds) >= 20
        else float(max(group_reward_stds))
    )
    metrics: dict[str, float | str] = {
        f"{split}/group_reward_std_mean": float(statistics.mean(group_reward_stds)),
        f"{split}/group_reward_std_p95": reward_std_p95,
        f"{split}/group_first_tool_diversity_mean": float(statistics.mean(group_first_tool_diversity)),
        f"{split}/group_first_tool_diversity_min": float(min(group_first_tool_diversity)),
    }
    for idx, (name, count) in enumerate(top_first_tools, start=1):
        metrics[f"{split}/first_tool_top{idx}_name"] = name
        metrics[f"{split}/first_tool_top{idx}_count"] = float(count)

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    if step % 10 == 0:
        print(
            f"[{split}] step={step} reward_std_mean={metrics[f'{split}/group_reward_std_mean']:.4f} "
            f"first_tool_div_mean={metrics[f'{split}/group_first_tool_diversity_mean']:.3f} "
            f"top_first_tools={top_first_tools}"
        )


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_messages = []
    for msg in messages:
        cleaned_msg = {k: v for k, v in msg.items() if v is not None}
        cleaned_messages.append(cleaned_msg)
    return cleaned_messages


def _log_retry_evaluation(aggregate: dict[str, float], batch: Any) -> None:
    for metric in ["mean", "min", "max"]:
        print(
            f"[val_retry] step={batch.step} n={int(aggregate.get('n', 0))} "
            f"reward_{metric}={aggregate.get(f'reward_{metric}', float('nan')):.4f} "
            f"optimal_trajectory_{metric}={aggregate.get(f'optimal_trajectory_{metric}', float('nan')):.4f} "
            f"correctness_{metric}={aggregate.get(f'correctness_score_{metric}', float('nan')):.4f} "
            f"format_{metric}={aggregate.get(f'format_score_{metric}', float('nan')):.4f}"
        )


def _log_eval_aggregate(*, split: str, step: int, aggregate: dict[str, float]) -> None:
    """
    Log a single aggregated evaluation point (mean across `eval_retries`) to W&B.

    This avoids writing multiple `val/*` points at the same training step.
    """
    if wandb.run is None:
        return

    payload: dict[str, float] = {}

    # For each metric `k`, log the mean under the canonical `split/k` key and
    for k, v in aggregate.items():
        if not isinstance(v, (int, float)):
            continue
        if k == "n":
            continue
        if k.endswith("_mean"):
            base = k[: -len("_mean")]
            payload[f"{split}/{base}"] = float(v)

    wandb.log(payload, step=step)


def _summarize_trajectories(trajectories: list[art.Trajectory]) -> dict[str, float]:
    """Compute simple mean metrics from a list of trajectories (used for eval retries)."""
    if not trajectories:
        return {}

    rewards = [float(t.reward) for t in trajectories]
    summary: dict[str, float] = {
        "reward": float(statistics.mean(rewards)),
        "reward_std_dev": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
    }

    keys: set[str] = set()
    for t in trajectories:
        if t.metrics:
            keys.update(t.metrics.keys())

    for key in sorted(keys):
        vals: list[float] = []
        for t in trajectories:
            if not t.metrics:
                continue
            v = t.metrics.get(key)
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            summary[key] = float(statistics.mean(vals))

    # Exception rate from rollout errors.
    errors = 0
    for t in trajectories:
        if isinstance((t.metadata or {}).get("error"), str):
            errors += 1
    summary["exception_rate"] = errors / len(trajectories)
    return summary


def _aggregate_retry_summaries(*, summaries: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate multiple eval summaries (each summary is already averaged over tasks).

    Produces mean/min/max/std across retries for each key plus `n` retries.
    """
    if not summaries:
        return {"n": 0.0}

    keys: set[str] = set()
    for s in summaries:
        keys.update(s.keys())

    out: dict[str, float] = {"n": float(len(summaries))}
    for key in sorted(keys):
        vals = [s[key] for s in summaries if isinstance(s.get(key), (int, float))]
        if not vals:
            continue
        out[f"{key}_mean"] = float(statistics.mean(vals))
        out[f"{key}_min"] = float(min(vals))
        out[f"{key}_max"] = float(max(vals))
        out[f"{key}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0

    return out


def _difficulty_for_step(*, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return float(max(0.0, min(1.0, step / (total_steps - 1))))


def _iterate_curriculum(
    *,
    base_epoch_size: int,
    groups_per_step: int,
    num_epochs: int,
    initial_step: int,
    tasks: list[Any],
    config: RunConfig,
    seed: int,
    use_tqdm: bool = True,
) -> Generator[DatasetBatch[int], None, None]:
    """
    Build deterministic curriculum batches.

    Keeps steps-per-epoch constant (based on `base_epoch_size`) by cycling through the eligible
    pool with coverage guarantees (no repeats until the pool is exhausted), and only repeating
    when the curriculum pool is smaller than the required number of draws.

    If `run.curriculum.sampling == "mixture"`, each step draws a fixed-size mixture across
    tool-call buckets (via `ToolCallsMixtureSampler`) instead of sampling uniformly from a
    single eligible pool.
    """
    if base_epoch_size <= 0:
        return

    steps_per_epoch = math.ceil(base_epoch_size / groups_per_step)
    total_steps = steps_per_epoch * num_epochs

    curriculum = config.curriculum
    use_mixture = (
        curriculum is not None
        and curriculum.enabled
        and not config.task_ids
        and getattr(curriculum, "sampling", "unlock") == "mixture"
    )

    sampler = ShuffleBagSampler(seed=seed)
    mixture_sampler: ToolCallsMixtureSampler | None = None
    if use_mixture:
        base_indices = get_task_indices(
            task_ids=config.task_ids,
            start_index=config.start_index,
            end_index=config.end_index,
            tasks=tasks,
            curriculum=None,
            difficulty=None,
            seed=seed,
        )
        mixture_sampler = ToolCallsMixtureSampler(
            tasks=tasks,
            indices=base_indices,
            curriculum=curriculum,
            seed=seed,
        )

    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            initial=initial_step,
            total=total_steps,
            desc="Iterating curriculum",
            unit="batch",
        )

    try:
        for global_step in range(total_steps):
            epoch = global_step // steps_per_epoch
            epoch_step = global_step % steps_per_epoch

            difficulty = _difficulty_for_step(step=global_step, total_steps=total_steps)
            if mixture_sampler is not None:
                items = mixture_sampler.sample_batch(difficulty=difficulty, batch_size=groups_per_step)
            else:
                eligible = get_task_indices(
                    task_ids=config.task_ids,
                    start_index=config.start_index,
                    end_index=config.end_index,
                    tasks=tasks,
                    curriculum=config.curriculum,
                    difficulty=difficulty,
                    seed=seed,
                )
                items = sampler.sample_batch(eligible=eligible, batch_size=groups_per_step)
            if global_step < initial_step:
                continue
            yield DatasetBatch(step=global_step, epoch=epoch, epoch_step=epoch_step, items=items)
            if progress_bar:
                progress_bar.update(1)
    finally:
        if progress_bar:
            progress_bar.close()


@limit_concurrency(256)
async def rollout_tau_bench_task(
    model: art.Model[TauBenchPolicyConfig],
    task_index: int,
    step: int = 0,
    phase: str = "train",
    reward_type: str = "real",
    is_shadow: bool = False,
) -> art.Trajectory:
    """
    Generate a trajectory for a single tau-bench task using the given model.
    This adapts the tau-bench evaluation loop for RL trajectory generation.
    Now truly async.
    """
    # print(f"Rolling out task {task_index} (step {step}, phase {phase})")
    config = copy.deepcopy(model.config.run_config)
    success_reward = 1.0 if config.env == "linear_algebra" else 1.0
    if is_shadow:
        config.model = "gpt-4.1"
        config.model_provider = "openai"
        config.api_key = None
        config.base_url = None

    # Get isolated environment for this task
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        task_split=phase,
        dataset_path=config.dataset_path,
        task_index=task_index,
    )
    if config.add_no_think:
        env.wiki += "\n/no_think"

    # Create agent with the trainable model
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )

    if not isinstance(agent, ToolCallingRLAgent):
        raise ValueError("Agent must be a ToolCallingRLAgent")

    # Create trajectory object
    traj = art.Trajectory(
        messages_and_choices=[],
        tools=env.tools_info,
        reward=0,
        metadata={
            "task_index": str(task_index),
            "env": config.env,
            "training_step": str(step),
            "phase": phase,
            "model": model.name,
            "reward_type": config.reward_type,
            "is_shadow": str(is_shadow),
        },
    )

    try:
        # Run the agent on the task (now async call)
        result = await agent.solve(
            env=env,
            task_index=task_index,
            max_assistant_turns=config.max_assistant_turns,
        )
        # optimal_trajectory: 1 if the entire trajectory is optimal (correct answer + optimal efficiency)
        optimal_trajectory = 1 if abs(result.reward - success_reward) <= 1e-6 else 0

        # Convert result to trajectory format
        traj.reward, explanation = await calculate_reward(result, config)

        # Build metrics dictionary
        reward_info = result.info.get("reward_info") or {}
        info = reward_info.get("info") or {}
        outputs = info.get("outputs") or {}
        has_valid_outputs = "correctness_score" in outputs
        total_completion_tokens = result.info.get("total_completion_tokens")
        if not isinstance(total_completion_tokens, (int, float)):
            total_completion_tokens = result.info["avg_completion_tokens"] * result.info["total_steps"]
        traj.metrics = {
            "total_steps": result.info["total_steps"],
            "final_prompt_tokens": result.info["final_prompt_tokens"],
            "total_completion_tokens": float(total_completion_tokens),
            "avg_completion_tokens": result.info["avg_completion_tokens"],
            "max_completion_tokens": result.info["max_completion_tokens"],
            "forced_stop": result.info["forced_stop"],
            "optimal_trajectory": optimal_trajectory,
            "valid_trajectory": 1 if has_valid_outputs else 0,
        }

        if has_valid_outputs:
            traj.metrics.update({
                "correctness_score": outputs["correctness_score"],
                "format_score": outputs["format_score"],
                "tool_success_score": outputs["tool_success_score"],
                "efficiency_penalty": outputs["efficiency_penalty"],
                "num_turns": outputs["num_turns"],
                "expected_turns": outputs["expected_turns"],
                "turn_deviation": outputs["num_turns"] - outputs["expected_turns"],
            })
        traj.metadata.update(result.info)
        traj.metadata["reward"] = "pending_general_rm" if config.reward_type == "general_rm" else traj.reward
        traj.metadata["optimal_trajectory"] = traj.metrics["optimal_trajectory"]
        traj.metadata["judge_explanation"] = explanation

        if config.messages_only:
            traj.messages_and_choices = clean_messages(result.messages)  # type: ignore
        else:
            traj.messages_and_choices = agent.create_messages_and_choices()  # type: ignore
    except Exception as e:
        print(f"Error in rollout for task {task_index}: {e}")
        traj.reward = -1.0
        traj.metadata["error"] = str(e)
        traj.metadata["traceback"] = traceback.format_exc()
        traj.messages_and_choices = agent.create_messages_and_choices()  # type: ignore
        result = SolveResult(
            reward=-1.0,
            info={"error": str(e)},
            messages=agent.messages,
            total_cost=0.0,
        )

    traj.finish()

    # Log to langfuse/openpipe
    try:
        await log_trajectory_to_openpipe(traj, result.messages)
    except Exception as e:
        print(f"Error logging trajectory to openpipe: {e}")

    # print(f"Finished rolling out task {task_index} (reward: {traj.reward})")
    return traj


async def evaluate_model(
    model: art.Model[TauBenchPolicyConfig],
    config: RunConfig,
    step: int,
    val_task_indices: list[int],
    split: str = "val",
) -> float:
    """Evaluate the model on a subset of tasks"""
    eval_retries = 1
    try:
        training_config = model.config.training_config
        if training_config is not None and getattr(training_config, "eval_retries", None) is not None:
            eval_retries = max(1, int(training_config.eval_retries))
    except Exception:
        eval_retries = 1

    print(f"Evaluating model on {len(val_task_indices)} tasks (passes={eval_retries})...")

    summaries: list[dict[str, float]] = []

    model_step = await model.get_step()
    eval_step = max(step, model_step)

    last_trajectories: list[art.Trajectory] = []
    for pass_idx in range(eval_retries):
        trajectories = await art.gather_trajectories(
            rollout_tau_bench_task(
                model=model,
                task_index=val_task_index,
                step=eval_step,
                phase=split,
                reward_type=config.reward_type,
            )
            for val_task_index in val_task_indices
        )
        last_trajectories = trajectories
        summaries.append(_summarize_trajectories(trajectories))
        if eval_retries > 1:
            print(f"Eval pass {pass_idx + 1}/{eval_retries}: reward={summaries[-1].get('reward', float('nan')):.4f}")

    aggregate = _aggregate_retry_summaries(summaries=summaries)
    _log_eval_aggregate(split=split, step=eval_step, aggregate=aggregate)

    print(
        f"[{split}] step={eval_step} n={int(aggregate.get('n', 0))} "
        f"reward_mean={aggregate.get('reward_mean', float('nan')):.4f} "
        f"reward_retry_std={aggregate.get('reward_std', float('nan')):.4f}"
    )
    if "optimal_trajectory_mean" in aggregate:
        print(f"[{split}] optimal_trajectory_mean={aggregate.get('optimal_trajectory_mean', float('nan')):.4f}")

    # Return mean reward across retries (for callers that use the float).
    return float(aggregate.get("reward_mean", float("nan")))


async def test(model: art.TrainableModel[TauBenchPolicyConfig]):
    """Main evaluation loop"""
    loop = asyncio.get_event_loop()
    big_pool = concurrent.futures.ThreadPoolExecutor(max_workers=50)
    loop.set_default_executor(big_pool)

    config = model.config.run_config
    training_config = model.config.training_config

    if training_config is None:
        raise ValueError("Training config is not set")

    register_kwargs = {}
    if model.config.training_config.chat_template is not None:
        register_kwargs["_openai_client_config"] = art.dev.OpenAIServerConfig(
            server_args=art.dev.ServerArgs(chat_template=model.config.training_config.chat_template)
        )

    with LocalBackend(in_process=config.in_process) as backend:
        # Resume/fork must happen *before* register() so the Unsloth/vLLM service
        # loads the correct LoRA adapter on startup.
        if model.config.run_config.resume:
            await backend._experimental_fork_checkpoint(
                model,
                from_model=model.config.run_config.resume_from,
                from_project=model.config.run_config.project,
                not_after_step=model.config.run_config.resume_step,
                verbose=True,
            )

        # Setup model with backend (starts the inference server + loads LoRA)
        await model.register(backend, **register_kwargs)

        config.api_key = model.inference_api_key
        config.base_url = model.inference_base_url
        config.base_model = model.base_model

        print("Loading training tasks...")

        # Load validation environment
        test_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            task_split="test",
            dataset_path=config.dataset_path,
        )

        test_task_indices = get_task_indices(
            task_ids=None,
            start_index=0,
            end_index=-1,
            tasks=test_env.tasks,
            curriculum=None,
            difficulty=None,
            seed=config.seed,
        )

        print(f"Validation on {len(test_task_indices)} tasks")

        # Final evaluation
        print("\n--- Final Evaluation ---")
        final_step = await model.get_step()
        final_reward = await evaluate_model(model, config, final_step, test_task_indices, split="test")
        print(f"Final average reward: {final_reward}")

        print("Evaluation complete!")


async def train(model: art.TrainableModel[TauBenchPolicyConfig]):
    """Main training loop adapted from art-e example"""
    loop = asyncio.get_event_loop()
    big_pool = concurrent.futures.ThreadPoolExecutor(max_workers=50)
    loop.set_default_executor(big_pool)

    config = model.config.run_config
    training_config = model.config.training_config

    if training_config is None:
        raise ValueError("Training config is not set")

    register_kwargs = {}
    if model.config.training_config.chat_template is not None:
        register_kwargs["_openai_client_config"] = art.dev.OpenAIServerConfig(
            server_args=art.dev.ServerArgs(chat_template=model.config.training_config.chat_template)
        )

    with LocalBackend(in_process=config.in_process) as backend:
        # Resume/fork must happen *before* register() so the Unsloth/vLLM service
        # loads the correct LoRA adapter on startup.
        if model.config.run_config.resume:
            await backend._experimental_fork_checkpoint(
                model,
                from_model=model.config.run_config.resume_from,
                from_project=model.config.run_config.project,
                not_after_step=model.config.run_config.resume_step,
                verbose=True,
            )
        else:
            print("Will continue training from previous latest checkpoint")

        # Setup model with backend (starts the inference server + loads LoRA)
        await model.register(backend, **register_kwargs)

        config.api_key = model.inference_api_key
        config.base_url = model.inference_base_url
        config.base_model = model.base_model

        print("Loading training tasks...")
        # Get environment to access tasks
        train_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            task_split="train",
            dataset_path=config.dataset_path,
        )

        # Load validation environment
        val_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            task_split="val",
            dataset_path=config.dataset_path,
        )

        train_task_indices = get_task_indices(
            task_ids=config.task_ids,
            start_index=config.start_index,
            end_index=config.end_index,
            tasks=train_env.tasks,
            curriculum=None,
            difficulty=None,
            seed=config.seed,
        )

        val_task_indices = get_task_indices(
            task_ids=config.val_task_ids,
            start_index=config.start_val_index,
            end_index=config.end_val_index,
            tasks=val_env.tasks,
            curriculum=None,
            difficulty=None,
            seed=config.seed,
        )

        print(f"Training on {len(train_task_indices)} tasks")
        print(f"Validation on {len(val_task_indices)} tasks")

        coverage: _CurriculumCoverageTracker | None = None
        if config.curriculum is not None and config.curriculum.enabled:
            tool_calls_by_index = {idx: len(train_env.tasks[idx].actions) for idx in train_task_indices}
            coverage = _CurriculumCoverageTracker(
                tool_calls_by_index=tool_calls_by_index,
                max_bucket_to_log=_COVERAGE_LOG_MAX_TOOL_CALLS_BUCKET,
            )

        initial_step = await model.get_step()
        base_epoch_size = len(train_task_indices)
        if config.curriculum is not None and config.curriculum.enabled:
            if coverage is not None:
                _prefill_coverage_tracker(
                    coverage=coverage,
                    initial_step=initial_step,
                    train_task_indices=train_task_indices,
                    tasks=train_env.tasks,
                    config=config,
                    training_config=training_config,
                )
            train_iterator = _iterate_curriculum(
                base_epoch_size=base_epoch_size,
                groups_per_step=training_config.groups_per_step,
                num_epochs=training_config.num_epochs,
                initial_step=initial_step,
                tasks=train_env.tasks,
                config=config,
                seed=config.seed,
            )
        else:
            # Training iterator
            train_iterator = iterate_dataset(
                train_task_indices,
                groups_per_step=training_config.groups_per_step,
                num_epochs=training_config.num_epochs,
                initial_step=initial_step,
            )

        for batch in train_iterator:
            print(f"\n--- Training Step {batch.step} (Epoch {batch.epoch}, Step {batch.epoch_step}) ---")

            if coverage is not None:
                coverage_metrics = coverage.update(step=batch.step, sampled_indices=list(batch.items))
                if wandb.run is not None:
                    wandb.log(coverage_metrics, step=batch.step)

            # Evaluation
            if batch.step % training_config.eval_steps == 0 and not config.skip_eval:
                print(f"\n--- Evaluating at Step {batch.step} ---")
                await evaluate_model(model, config, batch.step, val_task_indices)
                _archive_checkpoint(model=model, step=await model.get_step(), split="val")
                while True:
                    # proceed = input("Delete all previous checkpoints? (yes/no/exit): ").lower().strip()
                    proceed = "yes"

                    if proceed == "yes":
                        print("Deleting checkpoints...")
                        await _delete_checkpoints_keep_best(model)
                        break
                    elif proceed == "no":
                        print("Skipping checkpoint deletion.")
                        break
                    elif proceed == "exit":
                        print("Exiting...")
                        if wandb.run is not None:
                            wandb.finish()
                        return
                    else:
                        print("Please type 'yes', 'no', or 'exit'.")

            # Generate trajectory groups
            print(f"Generating trajectories for {len(batch.items)} tasks...")
            groups = await art.gather_trajectory_groups(
                art.TrajectoryGroup(
                    rollout_tau_bench_task(
                        model=model,
                        task_index=task_index,
                        step=batch.step,
                        phase="train",
                        reward_type=config.reward_type,
                        is_shadow=config.add_shadow_trajectory
                        and rollout_idx % training_config.trajectories_per_group == 0,
                    )
                    for rollout_idx in range(training_config.trajectories_per_group)
                )
                for task_index in batch.items
            )
            # await model.log(groups, split="train")
            _log_group_diversity(step=batch.step, groups=groups, split="train")

            if config.reward_type == "general_rm":
                print("Creating general RM trajectory groups...")
                updated_groups = await tqdm_asyncio.gather(
                    *[create_general_rm_trajectory_groups(group, config) for group in groups],
                    desc="Creating general RM trajectory groups",
                    total=len(groups),
                )
                groups = updated_groups

            # Training step
            print(f"Training on {len(groups)} trajectory groups...")
            dev_train_config: art.dev.TrainConfig = {
                "plot_tensors": config.plot_tensors,
                "importance_sampling_level": training_config.importance_sampling_level,
                "allow_training_without_logprobs": bool(config.messages_only),
                "scale_rewards": training_config.scale_rewards,
            }
            if training_config.epsilon is not None:
                dev_train_config["epsilon"] = training_config.epsilon
            if training_config.epsilon_high is not None:
                dev_train_config["epsilon_high"] = training_config.epsilon_high
            if training_config.max_negative_advantage_importance_sampling_weight is not None:
                dev_train_config["max_negative_advantage_importance_sampling_weight"] = (
                    training_config.max_negative_advantage_importance_sampling_weight
                )
            if training_config.truncated_importance_sampling is not None:
                dev_train_config["truncated_importance_sampling"] = training_config.truncated_importance_sampling
            await model.train(
                groups,
                config=art.TrainConfig(learning_rate=training_config.learning_rate, beta=training_config.beta),
                _config=dev_train_config,
            )
            if config.is_multi_gpu:
                await _delete_checkpoints_keep_best(model)

            # Log progress
            total_reward = sum(sum(traj.reward for traj in group.trajectories) for group in groups)
            num_trajectories = sum(len(group.trajectories) for group in groups)
            avg_reward = total_reward / num_trajectories if num_trajectories > 0 else 0
            print(f"Step {batch.step}: Average training reward = {avg_reward}")

        # Final evaluation
        print("\n--- Final Evaluation ---")
        final_step = await model.get_step()
        final_reward = await evaluate_model(model, config, final_step, val_task_indices)
        _archive_checkpoint(model=model, step=await model.get_step(), split="val")
        wandb.finish()
        print(f"Final average reward: {final_reward}")

        # Optional post-training upload to HF Hub (enabled when `HF_HUB_NAMESPACE` is set).
        try:
            await asyncio.to_thread(_push_experiment_dir_to_hf_sync, model=model)
        except Exception as e:
            print(f"[hf] Warning: post-training upload failed: {e}")

        print("Training completed!")


def main():
    """Entry point: expects a JSON-serialized TrainableModel (model_json) just like art-e/train.py"""

    parser = argparse.ArgumentParser(description="Run RL training for a serialized TrainableModel")
    parser.add_argument(
        "model_json",
        help="JSON string serialization of the TrainableModel to train",
    )
    args = parser.parse_args()

    print("Model JSON:", args.model_json)

    # Recreate the TrainableModel from the serialized JSON.
    model_dict = json.loads(args.model_json)

    # The nested `config` needs to be converted back into the proper pydantic model.
    model_dict["config"] = TauBenchPolicyConfig(**model_dict["config"])

    is_multi_gpu = False

    # the nested "_internal_config" needs to be converted back into the proper pydantic model.
    if "_internal_config" in model_dict and model_dict["_internal_config"] is not None:
        model_dict["_internal_config"] = art.dev.InternalModelConfig(**model_dict["_internal_config"])

    model: art.TrainableModel[TauBenchPolicyConfig] = art.TrainableModel(**model_dict)
    if model._internal_config is not None:
        is_multi_gpu = model._internal_config.get("engine_args", {}).get("tensor_parallel_size", 1) > 1
    model.config.run_config.model = model.name  # set run_config model name to model name
    model.config.run_config.is_multi_gpu = is_multi_gpu

    print(model)

    run_config = model.config.run_config

    print(f"Starting RL training for model: {model.name}")
    print(f"Base model: {model.base_model}")
    print(f"Environment: {run_config.env}")
    print(f"Task split: {run_config.task_split}")
    print(f"Reward type: {run_config.reward_type}")

    # Run training
    asyncio.run(train(model))


if __name__ == "__main__":
    main()
