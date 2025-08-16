# Based on linalg_zero/grpo/verl/examples/sglang_multiturn/run_qwen2.5-0.5b_gsm8k_multiturn_w_interaction.sh
# make sure your current working directory is the root of the linalg-zero project and run `make grpo-debug`

set -x

ulimit -n 65535

VERL_DIR="$(pwd)"
CONFIG_PATH="$VERL_DIR/../config/"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
OFFLOAD=${OFFLOAD:-False}

# To add a custom data loader:
# data.datagen.path=$VERL_DIR/../datagen.py\
# data.datagen.name=LinearAlgebraCurriculum \

# Possible models:
#  - Qwen/Qwen2.5-0.5B-Instruct
#  - Qwen/Qwen3-1.7B
#  - Qwen/Qwen3-0.6B

# Settings specific to Qwen3 models:
# actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode: ignore_strippable for Qwen3 models

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='linalg_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=$((1024 * 3)) \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$VERL_DIR/../config/linalg_tool_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='linalg_zero_grpo_rl' \
    trainer.experiment_name='qwen2.5-0.5b-linalg-zero-sgl' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=$HOME/data/linalg-zero/train.parquet \
    data.val_files=$HOME/data/linalg-zero/test.parquet \
    custom_reward_function.path="../compute_score.py" \
    custom_reward_function.name="calc_reward" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_PATH/linalg_interaction_config.yaml" \
    trainer.total_epochs=15 $@
