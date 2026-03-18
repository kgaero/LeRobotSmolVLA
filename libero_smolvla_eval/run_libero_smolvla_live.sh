#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/robotics/libero_smolvla_eval"

source "$PROJECT_DIR/.venv/bin/activate"
export MUJOCO_GL=glx
export HF_HUB_DISABLE_TELEMETRY=1
export LD_LIBRARY_PATH="$PROJECT_DIR/local_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

python -u "$PROJECT_DIR/live_libero_smolvla.py" \
  --policy-path "$PROJECT_DIR/policies/HuggingFaceVLA_smolvla_libero" \
  --suite libero_spatial \
  --task-id 0 \
  --device cuda \
  --n-action-steps 10 \
  --viewer-backend mujoco \
  --viewer-camera frontview \
  --output-dir "$PROJECT_DIR/output_live"
