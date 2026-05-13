# LIBERO to LeRobot Robotics Workflows

Utilities for collecting LIBERO keyboard teleoperation demonstrations, adding camera observations to robosuite HDF5 demos, converting those demos into a LeRobot-compatible dataset, and running local SmolVLA LIBERO evaluation.

The repository is organized around a practical pipeline:

1. Record a successful LIBERO task demonstration with keyboard teleoperation.
2. Replay saved simulator states to add image observations to `demo.hdf5`.
3. Convert one or more HDF5 demonstrations into a local dataset layout.
4. Optionally write a native LeRobot dataset and upload it to Hugging Face.
5. Run a local SmolVLA policy against LIBERO for evaluation or live viewing.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `teleop_libero_task0_keyboard.py` | Main native MuJoCo keyboard teleoperation script for LIBERO task collection. |
| `teleop_keyboard_libero.py` | Alternate teleoperation entry point with robosuite viewer support and MuJoCo fallback. |
| `libero_demo_hdf5_images.py` | Replays saved simulator states and writes image observations into robosuite HDF5 demos. |
| `build_lerobot_dataset.py` | Converts LIBERO / robosuite HDF5 demos into fallback dataset files and optional native LeRobot files. |
| `setup_lerobot_dataset_env.sh` | Installs the minimum Python dependencies required by the dataset converter. |
| `run_teleop_libero_task0.sh` | Convenience wrapper for task 0 keyboard collection using the local LIBERO environment. |
| `run_build_lerobot_dataset.sh` | Convenience wrapper for converting demos and optionally pushing the native dataset to Hugging Face. |
| `libero_smolvla_eval/` | Local LeRobot / LIBERO / SmolVLA evaluation workspace, including the LeRobot submodule. |
| `libero_task0_demos/` | Example or generated LIBERO demonstration sessions. |
| `lerobot_dataset_out/` | Generated dataset output from the conversion pipeline. |

Generated outputs, local virtual environments, large model files, and local runtime libraries are not part of the core source workflow and should generally stay out of commits unless intentionally versioned.

## Prerequisites

This project is intended for a Linux or WSL environment with GUI support when using live viewers.

Required for dataset conversion:

- Python 3.10 or newer
- `numpy`
- `h5py`
- `huggingface_hub`
- `imageio`
- `imageio-ffmpeg`

Required for teleoperation and evaluation:

- LIBERO
- robosuite
- MuJoCo
- Hugging Face LeRobot with LIBERO support
- PyTorch
- A local SmolVLA policy directory for evaluation
- A working GUI / OpenGL backend for viewer windows

The checked-in `libero_smolvla_eval/README.txt` records the local environment that was used for this workspace, including the Python version, LeRobot checkout, policy paths, rendering backend, and previous verification result.

## Setup

Install the minimum converter dependencies:

```bash
./setup_lerobot_dataset_env.sh
```

For teleoperation and evaluation, activate the local LIBERO / LeRobot environment:

```bash
source /home/kgaer/robotics/libero_smolvla_eval/.venv/bin/activate
```

If you are using WSLg or another desktop session, `MUJOCO_GL=glx` is used by the live viewer and teleoperation wrappers. Headless/offscreen workflows commonly use `MUJOCO_GL=egl`.

## Collect A LIBERO Demonstration

Run the task 0 teleoperation wrapper:

```bash
./run_teleop_libero_task0.sh
```

The wrapper:

- Activates `libero_smolvla_eval/.venv`.
- Sets `MUJOCO_GL=glx`.
- Loads local GUI compatibility libraries.
- Runs `teleop_libero_task0_keyboard.py` for `libero_spatial` task `0`.
- Saves demonstrations under `libero_smolvla_eval/libero_task0_demos/`.

Keyboard controls are printed by the teleoperation script when it starts. Successful episodes are compacted into `demo.hdf5` when you reset with `q`; the wrapper currently uses `--exit-after-save` to stop after the first successful saved episode.

## Add Images To HDF5 Demos

Image augmentation is called automatically during successful demo compaction. The helper in `libero_demo_hdf5_images.py` replays saved MuJoCo states and writes these observation datasets into each HDF5 demo:

- `obs/image` from the `agentview` camera
- `obs/wrist_image` from the `robot0_eye_in_hand` camera

The helper chooses `MUJOCO_GL=glx` when a display is available and `MUJOCO_GL=egl` for headless sessions unless `MUJOCO_GL` is already set.

## Build A LeRobot-Compatible Dataset

Use the wrapper for the default local paths:

```bash
./run_build_lerobot_dataset.sh
```

By default, the wrapper reads HDF5 demos from:

```text
/home/kgaer/robotics/libero_smolvla_eval/libero_task0_demos
```

and writes output to:

```text
/home/kgaer/robotics/lerobot_dataset_out
```

You can also call the converter directly:

```bash
python build_lerobot_dataset.py \
  --input ./libero_smolvla_eval/libero_task0_demos \
  --output ./lerobot_dataset_out \
  --task-name "spatial task 0" \
  --robot-name Panda \
  --repo-id kgaero/libero-spatial-task0-withImages \
  --overwrite
```

Important converter options:

| Option | Description |
| --- | --- |
| `--input` | HDF5 file or directory scanned recursively for `.hdf5` and `.h5` files. |
| `--output` | Dataset output root. Existing output requires `--overwrite`. |
| `--task-name` | Task label written into metadata and dataset card. |
| `--robot-name` | Robot label written into metadata and optional native LeRobot export. |
| `--repo-id` | Hugging Face dataset repository id used for metadata and upload. |
| `--push-to-hub` | Uploads the native LeRobot export to Hugging Face. |
| `--hf-token` | Token override for Hugging Face upload. Prefer this or environment variables over source edits. |
| `--private` | Creates the dataset repository as private when pushing. |
| `--primary-image-key` | Preferred image observation key when multiple image streams are present. |
| `--skip-videos` | Skips fallback MP4 creation. |
| `--disable-native-writer` | Writes only the fallback dataset layout. |
| `--fail-fast` | Stops on the first failed input file. |

The converter inspects HDF5 structure instead of assuming one fixed schema. It discovers episode-like groups, action/reward/done datasets, low-dimensional observations, and image observations, then aligns observation lengths to the action sequence.

## Dataset Output

The converter always writes a fallback layout:

```text
lerobot_dataset_out/
  episodes/
    episode_000000/
      actions.npy
      rewards.npy
      dones.npy
      meta.json
      obs_lowdim/
      obs_images/
  inspection/
  dataset_manifest.json
  dataset_summary.json
  files_processed.json
  validation_report.json
  README.md
```

When the local LeRobot API is importable and the discovered schema is consistent, it also writes native LeRobot files such as:

```text
lerobot_dataset_out/
  data/
  meta/
  images/
  videos/
```

Validation checks that required fallback episode files exist, action/reward/done lengths match, action shapes are consistent, and native LeRobot output can be loaded when the native writer was used.

## Push To Hugging Face

Set a token through the environment or pass it explicitly:

```bash
export HF_TOKEN=...
./run_build_lerobot_dataset.sh
```

or:

```bash
python build_lerobot_dataset.py \
  --input ./libero_smolvla_eval/libero_task0_demos \
  --output ./lerobot_dataset_out \
  --task-name "spatial task 0" \
  --robot-name Panda \
  --repo-id kgaero/libero-spatial-task0-withImages \
  --push-to-hub \
  --hf-token "$HF_TOKEN" \
  --overwrite
```

Do not commit real Hugging Face tokens. If a token has ever been committed, revoke it and create a replacement.

The upload path intentionally refuses fallback-only data. `--push-to-hub` requires a successful native LeRobot export so that the Hub receives the expected `data/`, `meta/`, image, and video structure.

## Run SmolVLA Evaluation

Use the batch evaluation wrapper:

```bash
./libero_smolvla_eval/run_libero_smolvla.sh
```

This runs `lerobot.scripts.lerobot_eval` with:

- Policy: `libero_smolvla_eval/policies/HuggingFaceVLA_smolvla_libero`
- Env type: `libero`
- Suite: `libero_spatial`
- Task ids: `[0]`
- Episodes: `1`
- Device: `cuda`
- Rendering backend: `MUJOCO_GL=egl`

Outputs are written under `libero_smolvla_eval/output/`.

## Run Live SmolVLA Viewing

Use the live viewer wrapper:

```bash
./libero_smolvla_eval/run_libero_smolvla_live.sh
```

The live script supports `robosuite`, `mujoco`, and `matplotlib` viewer backends. The current wrapper uses the native MuJoCo passive viewer with `MUJOCO_GL=glx`, writes an MP4, and writes a summary JSON under `libero_smolvla_eval/output_live/`.

Common direct invocation:

```bash
python -u ./libero_smolvla_eval/live_libero_smolvla.py \
  --policy-path ./libero_smolvla_eval/policies/HuggingFaceVLA_smolvla_libero \
  --suite libero_spatial \
  --task-id 0 \
  --device cuda \
  --n-action-steps 10 \
  --viewer-backend mujoco \
  --viewer-camera frontview
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Viewer window does not open | Confirm GUI support, `DISPLAY` / `WAYLAND_DISPLAY`, and `MUJOCO_GL`. Try `--viewer-backend mujoco` or `--viewer-backend matplotlib`. |
| Qt / OpenCV complains about `libqxcb.so`, `libSM`, or `libICE` | Use the wrapper scripts so local compatibility libraries are included in `LD_LIBRARY_PATH`. |
| Converter finds no episodes | Open the generated files in `lerobot_dataset_out/inspection/` to inspect the HDF5 tree and confirm action / observation datasets exist. |
| Native LeRobot writer is skipped | Check the logged reason. Common causes are missing LeRobot imports or inconsistent image shapes across episodes. |
| Hugging Face upload fails | Confirm `--repo-id`, token permissions, and that native LeRobot export succeeded. |
| CUDA evaluation fails | Confirm PyTorch CUDA availability and try `--device cpu` for a slower functional check. |

## Development Notes

- Prefer the wrapper scripts for the known local environment.
- Keep generated datasets, videos, model weights, and local caches out of source control unless there is a specific reason to version them.
- The repository currently has no root `LICENSE` file. Add one before publishing or sharing this as reusable open-source software.
- The repository currently has no automated test suite. For changes to the converter, use a small HDF5 demo and verify `validation_report.json` plus a native LeRobot load when available.
- Keep secrets out of source. Use `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `--hf-token` for uploads.

