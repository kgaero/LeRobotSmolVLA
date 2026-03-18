#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install \
  numpy \
  h5py \
  huggingface_hub \
  imageio \
  imageio-ffmpeg

echo
echo "Installed minimum dependencies for build_lerobot_dataset.py"
