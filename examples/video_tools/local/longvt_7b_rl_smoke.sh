#!/bin/bash
set -euo pipefail

ROOT=${LONGVT_ROOT:-/data/caojingqi/LongVT}
MODEL=${LONGVT_RL_MODEL:-$ROOT/runs/longvt_7b_sft_8gpu_20261112_120910/merged-hf}
DATA=${LONGVT_RL_DATA:-$ROOT/data/annotations/longvt_rl_selfqa_1k6.parquet}
VAL_DATA=${LONGVT_RL_VAL_DATA:-$ROOT/data/annotations/longvt_rl_val_114.parquet}
TOOL_CONFIG=${LONGVT_TOOL_CONFIG:-$ROOT/examples/video_tools/config/local/mcp_tool_config.yaml}
OUTPUT=${LONGVT_RL_OUTPUT:-$ROOT/runs/rl-smoke}

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TRAIN_BATCH=${TRAIN_BATCH:-4}
ROLLOUT_N=${ROLLOUT_N:-2}
TOTAL_STEPS=${TOTAL_STEPS:-2}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
if [ "$ROLLOUT_NAME" = "sglang" ]; then
  SGLANG_VIDEO=${SGLANG_VIDEO:-true}
else
  SGLANG_VIDEO=${SGLANG_VIDEO:-false}
fi
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-2}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-2}

if [ -z "${LLM_AS_A_JUDGE_BASE:-}" ]; then
  echo "LLM_AS_A_JUDGE_BASE must point to the reward/judge OpenAI-compatible endpoint." >&2
  exit 1
fi

export LONGVT_VIDEO_DIR=${LONGVT_VIDEO_DIR:-$ROOT/media}
export LONGVT_MCP_MAX_FRAMES=${LONGVT_MCP_MAX_FRAMES:-32}
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-av}
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
python3 -m verl.trainer.main_ppo \
  --config-path="$ROOT/examples/video_tools/config" \
  --config-name=timer1_multiturn_grpo \
  algorithm.adv_estimator=grpo \
  data.train_batch_size="$TRAIN_BATCH" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=false \
  data.truncation=left \
  data.return_raw_chat=true \
  data.sglang_video="$SGLANG_VIDEO" \
  data.dataloader_num_workers=2 \
  data.return_multi_modal_inputs=false \
  data.train_files="$DATA" \
  data.val_files="$VAL_DATA" \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_tool_call="$MAX_TOOL_CALLS" \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  algorithm.use_kl_in_reward=false \
  custom_reward_function.path="$ROOT/custom_rewards/tool_reward.py" \
  +custom_reward_function.reward_kwargs.use_iou_reward=true \
  trainer.project_name=LongVT \
  trainer.experiment_name=longvt-rl-smoke \
  trainer.logger="['console']" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$OUTPUT" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.rollout_data_dir="$OUTPUT/rollout"
