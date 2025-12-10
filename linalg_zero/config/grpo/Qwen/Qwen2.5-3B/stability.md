If training remains unstable with your current conservative settings, here's what I would change in order of priority:

## Immediate Changes

1. **Reduce learning rate to 1e-6** - This is the most common stable setting across tau-bench examples [1](#13-0)
2. **Increase beta to 0.003** - Add moderate KL regularization while keeping learning efficient [2](#13-1)

## If Still Unstable

3. **Add gradient clipping** - Set max_grad_norm to 1e-7 (like model 003) or 1.0 for more moderate clipping [3](#13-2)
4. **Reduce batch size** - Lower groups_per_step to 4 to reduce variance in updates [4](#13-3)
5. **Increase beta to 0.01** - Strong regularization if KL divergence is high [5](#13-4)

## Advanced Options

6. **Use sequence-level importance sampling** - More stable than token-level for long trajectories [6](#13-5)
7. **Enable reward scaling** - Helps with reward variance issues [7](#13-6)

## Implementation Example

```python
# Most conservative stable configuration
await model.train(
    trajectory_groups,
    config=art.TrainConfig(learning_rate=1e-6, beta=0.003),
    _config=art.dev.TrainConfig(
        importance_sampling_level="sequence",
        scale_rewards=True
    )
)
```

The learning rate is the most critical parameter - most tau-bench models that achieved stability converged to 1e-6 or lower [8](#13-7) .

## Notes

- Model 003's extreme learning rate (1e-2) only works with very low gradient norm (1e-7) and is not recommended for general use [3](#13-2)
- Monitor KL divergence closely - if it exceeds 0.1, increase beta immediately
- Consider reducing trajectories_per_group to 6 if memory is causing instability
- The 32B model (011) uses 5e-7 learning rate, showing that smaller isn't always better for every model [9](#13-8)

Wiki pages you might want to explore:
- [Training System (OpenPipe/ART)](/wiki/OpenPipe/ART#3)
- [Tool Calling & Tau-Bench (OpenPipe/ART)](/wiki/OpenPipe/ART#6.2)

### Citations

**File:** dev/tau-bench/run_training.py (L20-170)
```python
trainable_models = {
    "001": art.TrainableModel(
        name="tau-bench-rl-001-tm",
        project="tau_bench_rl",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=TauBenchPolicyConfig(
            training_config=TauBenchTrainingConfig(
                trajectories_per_group=6,
                groups_per_step=10,
                learning_rate=1.2e-5,
                eval_steps=10,
                val_set_size=85,
                training_dataset_size=30,
                num_epochs=50,
                train_mode="sync_rl",
            ),
            run_config=RunConfig(
                model_provider="hosted_vllm",
                user_model_provider="openai",
                user_model="gpt-4o",
                agent_strategy="tool-calling-rl",
                temperature=1.0,
                task_split="test",
                log_dir="rl_results",
                skip_eval=True,
            ),
        ),
    )
}

trainable_models["002"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["002"].config.training_config is not None
trainable_models["002"].name = "tau-bench-rl-002-tm-main-32"
trainable_models["002"].config.training_config.trajectories_per_group = 32
trainable_models["002"].config.training_config.groups_per_step = 4
trainable_models["002"].config.training_config.training_dataset_size = 4
trainable_models["002"].config.training_config.learning_rate = 1e-6

# v high lr, v low gn, because twitter said so
trainable_models["003"] = trainable_models["002"].model_copy(deep=True)
assert trainable_models["003"].config.training_config is not None
trainable_models["003"].name = "tau-bench-rl-003-tm"
trainable_models["003"].config.training_config.learning_rate = 1e-2
trainable_models["003"]._internal_config = art.dev.InternalModelConfig(
    trainer_args=art.dev.TrainerArgs(
        max_grad_norm=1e-7,
    )
)

trainable_models["008"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["008"].config.training_config is not None
trainable_models["008"].name = "tau-bench-rl-008-tm-small-2"
trainable_models["008"].config.training_config.trajectories_per_group = 64
trainable_models["008"].config.training_config.groups_per_step = 4
trainable_models["008"].config.training_config.training_dataset_size = 4
trainable_models["008"].config.training_config.learning_rate = 1e-6
trainable_models["008"].config.run_config.skip_eval = False
trainable_models["008"].config.training_config.val_set_size = 60
trainable_models["008"].config.training_config.eval_steps = 8
trainable_models["008"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

trainable_models["009"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["009"].config.training_config is not None
trainable_models["009"].name = "tau-bench-rl-009-tm-too-big"
trainable_models["009"].config.training_config.trajectories_per_group = 64
trainable_models["009"].config.training_config.groups_per_step = 16
trainable_models["009"].config.training_config.training_dataset_size = 32
trainable_models["009"].config.training_config.learning_rate = 1e-6
trainable_models["009"].config.run_config.skip_eval = False
trainable_models["009"].config.training_config.val_set_size = 60
trainable_models["009"].config.training_config.eval_steps = 8
trainable_models["009"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

trainable_models["010"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["010"].config.training_config is not None
trainable_models["010"].name = "tau-bench-rl-010-tm-too-big"
trainable_models["010"].config.training_config.trajectories_per_group = 64
trainable_models["010"].config.training_config.groups_per_step = 16
trainable_models["010"].config.training_config.training_dataset_size = 32
trainable_models["010"].config.training_config.learning_rate = 1e-6
trainable_models["010"].config.run_config.skip_eval = False
trainable_models["010"].config.run_config.reward_type = "real+llm"
trainable_models["010"].config.run_config.judge_model = "o4-mini"
trainable_models["010"].config.training_config.val_set_size = 60
trainable_models["010"].config.training_config.eval_steps = 8
trainable_models["010"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

trainable_models["011"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["011"].config.training_config is not None
trainable_models["011"].name = "tau-bench-rl-011-tm-32b-4"
trainable_models["011"].base_model = "Qwen/Qwen2.5-32B-Instruct"
trainable_models["011"].config.training_config.trajectories_per_group = 64
trainable_models["011"].config.training_config.groups_per_step = 32
trainable_models["011"].config.training_config.training_dataset_size = 32
trainable_models["011"].config.training_config.learning_rate = 5e-7
trainable_models["011"].config.run_config.skip_eval = False
trainable_models["011"].config.run_config.reward_type = "real"
trainable_models["011"].config.run_config.user_model = "gpt-4.1"
trainable_models["011"].config.training_config.val_set_size = 60
trainable_models["011"].config.training_config.eval_steps = 8
trainable_models["011"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=8, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_32b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)
trainable_models["011"].config.run_config.plot_tensors = True

trainable_models["012"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["012"].config.training_config is not None
trainable_models["012"].name = "tau-bench-rl-012-airline-multi-2"
trainable_models["012"].config.training_config.trajectories_per_group = 32
trainable_models["012"].config.training_config.groups_per_step = 16
trainable_models["012"].config.training_config.training_dataset_size = 16
trainable_models["012"].config.training_config.learning_rate = 1e-6
trainable_models["012"].config.run_config.skip_eval = False
trainable_models["012"].config.training_config.val_set_size = 32
trainable_models["012"].config.training_config.eval_steps = 8
trainable_models["012"].config.run_config.reward_type = "real"
trainable_models["012"].config.run_config.user_model = "gpt-4.1"
trainable_models["012"].config.run_config.env = "airline"
trainable_models["012"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

trainable_models["015"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["015"].config.training_config is not None
trainable_models["015"].name = "tau-bench-rl-015-tm-2"
trainable_models["015"].base_model = "willcb/Qwen3-14B"
trainable_models["015"].config.training_config.trajectories_per_group = 64
trainable_models["015"].config.training_config.groups_per_step = 32
trainable_models["015"].config.training_config.training_dataset_size = 32
trainable_models["015"].config.training_config.learning_rate = 5e-6
```

**File:** dev/tau-bench/run_training.py (L184-204)
```python
trainable_models["016"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["016"].config.training_config is not None
trainable_models["016"].name = "tau-bench-rl-016-tm-14b-gspo-2"
trainable_models["016"].base_model = "Qwen/Qwen2.5-14B-Instruct"
trainable_models["016"].config.training_config.trajectories_per_group = 32
trainable_models["016"].config.training_config.groups_per_step = 32
trainable_models["016"].config.training_config.training_dataset_size = 32
trainable_models["016"].config.training_config.learning_rate = 5e-7
trainable_models["016"].config.training_config.importance_sampling_level = "sequence"
trainable_models["016"].config.run_config.skip_eval = False
trainable_models["016"].config.run_config.reward_type = "real"
trainable_models["016"].config.run_config.user_model = "gpt-4.1"
trainable_models["016"].config.training_config.val_set_size = 60
trainable_models["016"].config.training_config.eval_steps = 8
trainable_models["016"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=8, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

```

**File:** src/art/unsloth/train.py (L231-233)
```python
        if config.beta > 0.0:
            trainer._metrics["train"]["kl_div"].append(mean_kl.item())
        return mean_policy_loss + config.beta * mean_kl
```

**File:** src/art/dev/train.py (L20-24)
```python
    plot_tensors: bool
    precalculate_logprobs: bool
    scale_learning_rate_by_reward_std_dev: bool
    scale_rewards: bool
    truncated_importance_sampling: float | None
```
