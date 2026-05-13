#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/robotics"
LIBERO_DIR="$PROJECT_DIR/libero_smolvla_eval"

cd "$PROJECT_DIR"
source "$LIBERO_DIR/.venv/bin/activate"

export MUJOCO_GL=glx
export LD_LIBRARY_PATH="$LIBERO_DIR/local_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

python "$PROJECT_DIR/teleop_libero_task0_keyboard.py" \
  --suite libero_spatial \
  --task-id 0 \
  --camera frontview \
  --output-dir "$LIBERO_DIR/libero_task0_demos" \
  --image-size 256 \
  --exit-after-save
