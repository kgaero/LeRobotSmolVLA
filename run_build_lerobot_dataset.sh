#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/robotics"
LIBERO_DIR="$PROJECT_DIR/libero_smolvla_eval"
INPUT_DIR="$LIBERO_DIR/libero_task0_demos"
OUTPUT_DIR="$PROJECT_DIR/lerobot_dataset_out"
TASK_NAME="${TASK_NAME:-spatial task 0}"
ROBOT_NAME="${ROBOT_NAME:-Panda}"
REPO_ID="${REPO_ID:-kgaero/libero-spatial-task0-withImages}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -f "$LIBERO_DIR/.venv/bin/activate" ]; then
  # Reuse the LIBERO environment because it already contains the LeRobot stack.
  # shellcheck disable=SC1091
  source "$LIBERO_DIR/.venv/bin/activate"
  PYTHON_BIN=python
fi

CMD=(
  "$PYTHON_BIN"
  "$PROJECT_DIR/build_lerobot_dataset.py"
  --input "$INPUT_DIR"
  --output "$OUTPUT_DIR"
  --task-name "$TASK_NAME"
  --robot-name "$ROBOT_NAME"
  --overwrite
)

if [ -n "$REPO_ID" ]; then
  CMD+=(--repo-id "$REPO_ID" --push-to-hub)
fi

CMD+=("$@")

printf 'Running command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
