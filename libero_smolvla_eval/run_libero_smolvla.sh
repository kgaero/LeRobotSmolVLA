#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/robotics/libero_smolvla_eval"

source "$PROJECT_DIR/.venv/bin/activate"
export MUJOCO_GL=egl
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$PROJECT_DIR/output"
cd "$PROJECT_DIR/lerobot"

python -m lerobot.scripts.lerobot_eval \
  --policy.path="$PROJECT_DIR/policies/HuggingFaceVLA_smolvla_libero" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids=[0] \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$PROJECT_DIR/output"
