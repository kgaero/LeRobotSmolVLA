#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/robotics/libero_smolvla_eval"
POLICY_DIR="$PROJECT_DIR/policies/smolvla-libero-colab-test"

source "$PROJECT_DIR/.venv/bin/activate"

export MUJOCO_GL=egl
export HF_HUB_DISABLE_TELEMETRY=1
export LD_LIBRARY_PATH="$PROJECT_DIR/local_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

mkdir -p "$PROJECT_DIR/policies"

echo "Checking policy..."

python - <<PY
from huggingface_hub import snapshot_download
import os

policy_dir = os.path.expanduser("$POLICY_DIR")

if not os.path.exists(policy_dir):
    print("Downloading SmolVLA policy from HuggingFace...")
    snapshot_download(
        repo_id="kgaero/smolvla-libero-colab-test",
        local_dir=policy_dir,
        local_dir_use_symlinks=False,
    )
else:
    print("Policy already exists. Skipping download.")
PY

echo "Running LIBERO task 0..."

python -u "$PROJECT_DIR/live_libero_smolvla.py" \
  --policy-path "$POLICY_DIR" \
  --suite libero_spatial \
  --task-id 0 \
  --device cuda \
  --n-action-steps 10 \
  --viewer-backend matplotlib \
  --output-dir "$PROJECT_DIR/output_live"
