#!/usr/bin/env python3
"""Convert LIBERO / robosuite teleoperation demos into a LeRobot-compatible dataset.

This script prioritizes:
- HDF5 schema inspection over hardcoded assumptions
- a fully implemented fallback intermediate dataset layout
- optional native LeRobot dataset writing when the local environment supports it
- explicit logging and validation for single-episode bring-up and future batching
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

# Hardcode a temporary token here for first-pass testing, then delete it later.
HF_TOKEN = "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"

LOGGER = logging.getLogger("build_lerobot_dataset")

HDF5_SUFFIXES = {".hdf5", ".h5"}
ACTION_CANDIDATES = ("actions", "action", "policy_actions")
REWARD_CANDIDATES = ("rewards", "reward")
DONE_CANDIDATES = ("dones", "done", "terminals", "terminal", "success", "successful")
OBS_GROUP_CANDIDATES = ("obs", "observations", "observation")
EPISODE_NAME_RE = re.compile(r"^(demo|episode|ep|traj|trajectory|rollout)[_\-]?\d+$", re.IGNORECASE)
IMAGE_KEY_HINTS = (
    "image",
    "rgb",
    "camera",
    "cam",
    "view",
    "eye",
    "wrist",
    "hand",
    "front",
    "agentview",
    "robot0_eye_in_hand",
)
PRIMARY_IMAGE_PRIORITIES = (
    "agentview_image",
    "agentview",
    "frontview_image",
    "frontview",
    "rgb",
    "image",
)


@dataclass
class ObservationSpec:
    """Description of one observation dataset discovered inside a demo group."""

    key: str
    dataset_path: str
    shape: tuple[int, ...]
    dtype: str
    is_image: bool


@dataclass
class DemoGroupSpec:
    """Description of one episode-like group inside a source HDF5 file."""

    group_path: str
    action_path: str | None
    reward_path: str | None
    done_path: str | None
    observation_specs: list[ObservationSpec]
    score: int
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeData:
    """In-memory episode payload extracted from one demo group."""

    episode_id: int
    source_file: str
    source_demo_path: str
    length: int
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    obs_lowdim: dict[str, np.ndarray]
    obs_images: dict[str, np.ndarray]
    primary_image_key: str | None
    extraction_notes: list[str]
    discovery: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class FileProcessResult:
    """Per-input-file processing result."""

    file_path: str
    success: bool
    discovered_demo_groups: list[str]
    created_episode_ids: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class NativeBackendResult:
    """Status of optional native LeRobot writing."""

    used: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Build a LeRobot-compatible dataset from LIBERO / robosuite demo HDF5 files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to one .hdf5/.h5 demo file or a directory to scan recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id, e.g. kgaero/libero-spatial-task0-test.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name to record in metadata and dataset card.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="unknown_robot",
        help="Robot name for metadata and optional native LeRobot export.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Playback / dataset FPS metadata.",
    )
    parser.add_argument(
        "--primary-image-key",
        type=str,
        default=None,
        help="Preferred image observation key, e.g. agentview_image.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the finished output folder to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional token override. If omitted, HF_TOKEN at the top of this file is used.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private on the Hub.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Do not write fallback MP4 videos even if image observations are present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete a previous output directory before rebuilding.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first per-file processing error instead of continuing.",
    )
    parser.add_argument(
        "--disable-native-writer",
        action="store_true",
        help="Disable the optional native LeRobot writer even if it is importable.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure process-wide logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dependency(module_name: str, install_hint: str) -> Any:
    """Import a required dependency with a clear installation hint."""

    try:
        return __import__(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing dependency '{module_name}'. Install it with: {install_hint}") from exc


def ensure_clean_output_dir(output_root: Path, overwrite: bool) -> None:
    """Prepare the output directory, optionally replacing a previous run."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. Re-run with --overwrite to rebuild it."
            )
        LOGGER.warning("Removing existing output directory: %s", output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def find_hdf5_files(input_path: Path) -> list[Path]:
    """Return one or more HDF5 files from a file or recursive directory scan."""

    if input_path.is_file():
        if input_path.suffix.lower() not in HDF5_SUFFIXES:
            raise ValueError(f"Input file is not an HDF5 file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(path for path in input_path.rglob("*") if path.suffix.lower() in HDF5_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under: {input_path}")
    return files


def to_jsonable(value: Any) -> Any:
    """Convert nested values into JSON-serializable forms."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sanitize_key(key: str) -> str:
    """Make observation keys safe for filenames and generic dataset feature names."""

    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", key).strip("._-")
    return sanitized or "unnamed"


def serialize_attrs(attrs: Any) -> dict[str, Any]:
    """Serialize HDF5 attribute mappings."""

    serialized: dict[str, Any] = {}
    for key in attrs.keys():
        serialized[str(key)] = to_jsonable(attrs[key])
    return serialized


def print_hdf5_tree(h5_file: Any, file_path: Path) -> list[str]:
    """Build and log a readable HDF5 tree."""

    lines = [f"HDF5 tree for {file_path}"]

    def visitor(name: str, obj: Any) -> None:
        depth = name.count("/")
        indent = "  " * depth
        node_name = name if name else "/"
        if hasattr(obj, "keys"):
            lines.append(f"{indent}[G] {node_name}")
        else:
            shape = tuple(int(dim) for dim in obj.shape)
            lines.append(f"{indent}[D] {node_name} shape={shape} dtype={obj.dtype}")

    h5_file.visititems(visitor)
    for line in lines:
        LOGGER.info("%s", line)
    return lines


def group_contains_named_dataset(group: Any, candidate_names: tuple[str, ...]) -> bool:
    """Return True if the group contains a direct child dataset with a candidate name."""

    for name in candidate_names:
        if name in group and not hasattr(group[name], "keys"):
            return True
    return False


def find_candidate_dataset(group: Any, candidate_names: tuple[str, ...]) -> str | None:
    """Find the best matching dataset path inside a demo group."""

    direct_hits: list[str] = []
    recursive_hits: list[str] = []

    for name in candidate_names:
        if name in group and not hasattr(group[name], "keys"):
            direct_hits.append(group[name].name)

    def visitor(name: str, obj: Any) -> None:
        if hasattr(obj, "keys"):
            return
        leaf = PurePosixPath(name).name
        if leaf in candidate_names:
            recursive_hits.append(obj.name)

    group.visititems(visitor)
    if direct_hits:
        return sorted(direct_hits, key=lambda item: (item.count("/"), item))[0]
    if recursive_hits:
        return sorted(recursive_hits, key=lambda item: (item.count("/"), item))[0]
    return None


def is_image_dataset(key: str, shape: tuple[int, ...], dtype_name: str) -> bool:
    """Heuristic classification for image observations."""

    key_l = key.lower()
    if any(hint in key_l for hint in IMAGE_KEY_HINTS):
        if len(shape) >= 3:
            return True
    if len(shape) == 4:
        if shape[-1] in (1, 3, 4):
            return True
        if shape[1] in (1, 3, 4):
            return True
    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            return True
        if shape[0] in (1, 3, 4):
            return True
    if dtype_name.startswith("|S") or dtype_name.startswith("<U") or dtype_name == "object":
        return False
    return False


def discover_observation_keys(demo_group: Any) -> list[ObservationSpec]:
    """Discover observation datasets inside a demo group."""

    specs: list[ObservationSpec] = []
    obs_roots: list[tuple[str, Any]] = []
    for candidate in OBS_GROUP_CANDIDATES:
        if candidate in demo_group and hasattr(demo_group[candidate], "keys"):
            obs_roots.append((candidate, demo_group[candidate]))

    if not obs_roots:
        obs_roots.append(("", demo_group))

    seen_paths: set[str] = set()
    skip_leaf_names = set(ACTION_CANDIDATES + REWARD_CANDIDATES + DONE_CANDIDATES)

    for root_prefix, root_group in obs_roots:
        def visitor(name: str, obj: Any) -> None:
            if hasattr(obj, "keys"):
                return
            if obj.name in seen_paths:
                return

            leaf = PurePosixPath(name).name
            if root_prefix == "" and leaf in skip_leaf_names:
                return

            rel_key = name if root_prefix == "" else name.removeprefix(f"{root_prefix}/")
            shape = tuple(int(dim) for dim in obj.shape)
            if len(shape) == 0:
                return

            spec = ObservationSpec(
                key=rel_key,
                dataset_path=obj.name,
                shape=shape,
                dtype=str(obj.dtype),
                is_image=is_image_dataset(rel_key, shape, str(obj.dtype)),
            )
            specs.append(spec)
            seen_paths.add(obj.name)

        root_group.visititems(visitor)

    specs.sort(key=lambda item: item.key)
    return specs


def discover_demo_groups(h5_file: Any) -> list[DemoGroupSpec]:
    """Locate episode-like groups robustly inside an HDF5 file."""

    candidates: list[DemoGroupSpec] = []

    def visitor(name: str, obj: Any) -> None:
        if not hasattr(obj, "keys"):
            return

        score = 0
        if group_contains_named_dataset(obj, ACTION_CANDIDATES):
            score += 5
        if any(candidate in obj and hasattr(obj[candidate], "keys") for candidate in OBS_GROUP_CANDIDATES):
            score += 2

        base_name = PurePosixPath(obj.name).name
        if EPISODE_NAME_RE.match(base_name):
            score += 2
        if PurePosixPath(obj.name).parent.name == "data":
            score += 1

        if score == 0:
            return

        action_path = find_candidate_dataset(obj, ACTION_CANDIDATES)
        reward_path = find_candidate_dataset(obj, REWARD_CANDIDATES)
        done_path = find_candidate_dataset(obj, DONE_CANDIDATES)
        observation_specs = discover_observation_keys(obj)
        if action_path is None and not observation_specs:
            return

        candidates.append(
            DemoGroupSpec(
                group_path=obj.name,
                action_path=action_path,
                reward_path=reward_path,
                done_path=done_path,
                observation_specs=observation_specs,
                score=score,
                attrs=serialize_attrs(obj.attrs),
            )
        )

    h5_file.visititems(visitor)
    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item.score, item.group_path.count("/"), item.group_path))
    final_candidates: list[DemoGroupSpec] = []
    for candidate in candidates:
        if any(
            other.group_path != candidate.group_path
            and other.group_path.startswith(candidate.group_path.rstrip("/") + "/")
            for other in candidates
        ):
            continue
        if candidate.group_path not in {item.group_path for item in final_candidates}:
            final_candidates.append(candidate)

    final_candidates.sort(key=lambda item: item.group_path)
    for candidate in final_candidates:
        LOGGER.info(
            "Discovered demo group: %s | action=%s | reward=%s | done=%s | obs=%d",
            candidate.group_path,
            candidate.action_path,
            candidate.reward_path,
            candidate.done_path,
            len(candidate.observation_specs),
        )
    return final_candidates


def pick_primary_image_key(image_keys: list[str], preferred_key: str | None) -> str | None:
    """Choose the primary image observation key."""

    if not image_keys:
        return None

    if preferred_key:
        for key in image_keys:
            if key == preferred_key or PurePosixPath(key).name == preferred_key:
                return key
            if sanitize_key(key) == sanitize_key(preferred_key):
                return key

    lowered = {key: key.lower() for key in image_keys}
    for preferred in PRIMARY_IMAGE_PRIORITIES:
        for key, value in lowered.items():
            if preferred in value:
                return key
    return sorted(image_keys)[0]


def infer_actions_array(raw_actions: np.ndarray, obs_specs: list[ObservationSpec]) -> np.ndarray:
    """Normalize the action dataset to shape (T, A)."""

    actions = np.asarray(raw_actions)
    if actions.ndim == 0:
        raise ValueError("Action dataset is scalar; expected a time series.")
    if actions.ndim == 1:
        obs_lengths = {spec.shape[0] for spec in obs_specs if spec.shape}
        if actions.shape[0] in obs_lengths:
            actions = actions.reshape(actions.shape[0], 1)
        else:
            actions = actions.reshape(1, actions.shape[0])
    actions = actions.astype(np.float32, copy=False)
    if actions.shape[0] == 0:
        raise ValueError("Action dataset is empty.")
    return actions


def align_array_length(name: str, array: np.ndarray, target_len: int, notes: list[str]) -> np.ndarray | None:
    """Align an observation-like array to the action length."""

    if array.ndim == 0:
        notes.append(f"Skipped scalar dataset '{name}'.")
        return None

    current_len = int(array.shape[0])
    if current_len == target_len:
        return array

    if current_len == target_len + 1:
        notes.append(f"Trimmed '{name}' from length {current_len} to {target_len} (T+1 -> T).")
        return array[:target_len]

    if target_len == 1 and array.ndim >= 1:
        reshaped = np.expand_dims(array, axis=0) if current_len != 1 else array
        if reshaped.shape[0] == 1:
            notes.append(f"Reshaped single-step dataset '{name}' to batch dimension 1.")
            return reshaped

    notes.append(
        f"Skipped '{name}' because leading dimension {current_len} does not match target length {target_len}."
    )
    return None


def normalize_image_sequence(key: str, frames: np.ndarray, target_len: int, notes: list[str]) -> np.ndarray | None:
    """Normalize image observations to uint8 frames of shape (T, H, W, C)."""

    array = np.asarray(frames)
    array = align_array_length(key, array, target_len, notes)
    if array is None:
        return None

    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            array = np.expand_dims(array, axis=0)
        elif array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
            array = np.expand_dims(array, axis=0)
        else:
            notes.append(f"Skipped image key '{key}' because shape {tuple(array.shape)} is ambiguous.")
            return None
    elif array.ndim == 4:
        if array.shape[-1] in (1, 3, 4):
            pass
        elif array.shape[1] in (1, 3, 4):
            array = np.transpose(array, (0, 2, 3, 1))
            notes.append(f"Transposed channel-first image key '{key}' to channel-last.")
        else:
            notes.append(f"Skipped image key '{key}' because shape {tuple(array.shape)} is not image-like.")
            return None
    else:
        notes.append(f"Skipped image key '{key}' because ndim={array.ndim} is not supported.")
        return None

    if array.shape[0] != target_len:
        notes.append(f"Skipped image key '{key}' after normalization because length != target length.")
        return None

    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        return np.clip(array * scale, 0, 255).astype(np.uint8)
    if np.issubdtype(array.dtype, np.bool_):
        return (array.astype(np.uint8) * 255).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def load_optional_vector(
    h5_file: Any,
    dataset_path: str | None,
    target_len: int,
    fill_value: float | int,
    fill_kind: str,
    notes: list[str],
) -> np.ndarray:
    """Load rewards / dones or synthesize defaults if absent."""

    if dataset_path is None:
        if fill_kind == "reward":
            notes.append("Rewards missing; filled with zeros.")
            return np.zeros(target_len, dtype=np.float32)
        notes.append("Dones missing; inferred last step as done.")
        dones = np.zeros(target_len, dtype=bool)
        dones[-1] = True
        return dones

    raw = np.asarray(h5_file[dataset_path][()])
    if raw.ndim == 0:
        raw = np.repeat(raw.reshape(1), target_len)
    if raw.ndim > 1:
        raw = raw.reshape(raw.shape[0], -1)
        if raw.shape[1] == 1:
            raw = raw[:, 0]
    aligned = align_array_length(dataset_path, raw, target_len, notes)
    if aligned is None:
        if fill_kind == "reward":
            notes.append(f"Reward dataset '{dataset_path}' was unusable; filled with zeros.")
            return np.zeros(target_len, dtype=np.float32)
        notes.append(f"Done dataset '{dataset_path}' was unusable; inferred last step as done.")
        dones = np.zeros(target_len, dtype=bool)
        dones[-1] = True
        return dones

    if fill_kind == "reward":
        return np.asarray(aligned, dtype=np.float32).reshape(target_len)
    return np.asarray(aligned).astype(bool).reshape(target_len)


def flatten_lowdim_observations(obs_lowdim: dict[str, np.ndarray]) -> tuple[np.ndarray | None, list[str]]:
    """Flatten all low-dimensional observation streams into one state matrix."""

    if not obs_lowdim:
        return None, []

    matrices: list[np.ndarray] = []
    names: list[str] = []
    for key in sorted(obs_lowdim):
        value = np.asarray(obs_lowdim[key], dtype=np.float32)
        if value.ndim == 1:
            value = value.reshape(value.shape[0], 1)
        else:
            value = value.reshape(value.shape[0], -1)
        matrices.append(value)
        if value.shape[1] == 1:
            names.append(key)
        else:
            names.extend(f"{key}[{idx}]" for idx in range(value.shape[1]))
    if not matrices:
        return None, []
    return np.concatenate(matrices, axis=1), names


def load_demo_episode(
    h5_file: Any,
    file_path: Path,
    demo_spec: DemoGroupSpec,
    episode_id: int,
    preferred_image_key: str | None,
) -> EpisodeData:
    """Extract one demo group into an EpisodeData payload."""

    notes: list[str] = []
    metadata = {
        "file_attrs": serialize_attrs(h5_file.attrs),
        "demo_attrs": demo_spec.attrs,
    }

    if demo_spec.action_path is None:
        raise KeyError(f"Required action dataset missing for demo group '{demo_spec.group_path}'.")

    actions = infer_actions_array(h5_file[demo_spec.action_path][()], demo_spec.observation_specs)
    target_len = int(actions.shape[0])
    rewards = load_optional_vector(h5_file, demo_spec.reward_path, target_len, 0.0, "reward", notes)
    dones = load_optional_vector(h5_file, demo_spec.done_path, target_len, 1, "done", notes)

    obs_lowdim: dict[str, np.ndarray] = {}
    obs_images: dict[str, np.ndarray] = {}
    skipped_observations: list[str] = []

    for spec in demo_spec.observation_specs:
        raw = np.asarray(h5_file[spec.dataset_path][()])
        if spec.is_image:
            normalized = normalize_image_sequence(spec.key, raw, target_len, notes)
            if normalized is not None:
                obs_images[spec.key] = normalized
            else:
                skipped_observations.append(spec.key)
        else:
            aligned = align_array_length(spec.key, raw, target_len, notes)
            if aligned is None:
                skipped_observations.append(spec.key)
            else:
                obs_lowdim[spec.key] = np.asarray(aligned)

    image_keys = sorted(obs_images.keys())
    lowdim_keys = sorted(obs_lowdim.keys())
    primary_image_key = pick_primary_image_key(image_keys, preferred_image_key)

    if target_len <= 0:
        raise ValueError(f"Episode {episode_id} extracted zero frames from demo group '{demo_spec.group_path}'.")

    discovery = {
        "action_path": demo_spec.action_path,
        "reward_path": demo_spec.reward_path,
        "done_path": demo_spec.done_path,
        "observation_keys": [spec.key for spec in demo_spec.observation_specs],
        "image_keys": image_keys,
        "lowdim_keys": lowdim_keys,
        "skipped_observation_keys": skipped_observations,
    }

    LOGGER.info(
        "Loaded episode %06d from %s:%s | length=%d | action_dim=%s | images=%s | lowdim=%s",
        episode_id,
        file_path,
        demo_spec.group_path,
        target_len,
        tuple(actions.shape[1:]),
        image_keys,
        lowdim_keys,
    )

    return EpisodeData(
        episode_id=episode_id,
        source_file=str(file_path),
        source_demo_path=demo_spec.group_path,
        length=target_len,
        actions=actions,
        rewards=rewards,
        dones=dones,
        obs_lowdim=obs_lowdim,
        obs_images=obs_images,
        primary_image_key=primary_image_key,
        extraction_notes=notes,
        discovery=discovery,
        metadata=metadata,
    )


def write_episode_video(video_path: Path, frames: np.ndarray, fps: int) -> None:
    """Write an MP4 video from uint8 frames."""

    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'imageio'. Install it with: pip install imageio imageio-ffmpeg") from exc

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video_path), fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def write_fallback_episode(episode: EpisodeData, output_root: Path, fps: int, write_videos: bool) -> dict[str, Any]:
    """Write one episode into the fully implemented fallback intermediate format."""

    episode_dir = output_root / "episodes" / f"episode_{episode.episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    np.save(episode_dir / "actions.npy", episode.actions)
    np.save(episode_dir / "rewards.npy", episode.rewards)
    np.save(episode_dir / "dones.npy", episode.dones.astype(bool))

    if episode.obs_lowdim:
        np.savez_compressed(episode_dir / "obs_state.npz", **episode.obs_lowdim)

    image_archive_paths: dict[str, str] = {}
    if episode.obs_images:
        image_dir = episode_dir / "obs_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for key, frames in episode.obs_images.items():
            archive_name = f"{sanitize_key(key)}.npz"
            archive_path = image_dir / archive_name
            np.savez_compressed(archive_path, frames=frames)
            image_archive_paths[key] = str(archive_path.relative_to(output_root))

    written_videos: dict[str, str] = {}
    if write_videos and episode.primary_image_key and episode.primary_image_key in episode.obs_images:
        try:
            primary_video_path = episode_dir / "rgb.mp4"
            write_episode_video(primary_video_path, episode.obs_images[episode.primary_image_key], fps)
            written_videos[episode.primary_image_key] = str(primary_video_path.relative_to(output_root))

            for key, frames in episode.obs_images.items():
                if key == episode.primary_image_key:
                    continue
                video_path = episode_dir / f"rgb_{sanitize_key(key)}.mp4"
                write_episode_video(video_path, frames, fps)
                written_videos[key] = str(video_path.relative_to(output_root))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Skipping fallback video export for episode %06d because video writing failed: %s",
                episode.episode_id,
                exc,
            )

    state_matrix, state_names = flatten_lowdim_observations(episode.obs_lowdim)
    episode_meta = {
        "episode_id": episode.episode_id,
        "source_file": episode.source_file,
        "source_demo_path": episode.source_demo_path,
        "length": episode.length,
        "action_shape": list(episode.actions.shape),
        "reward_available": episode.discovery["reward_path"] is not None,
        "done_available": episode.discovery["done_path"] is not None,
        "primary_image_key": episode.primary_image_key,
        "image_keys": sorted(episode.obs_images),
        "lowdim_keys": sorted(episode.obs_lowdim),
        "state_vector_dim": int(state_matrix.shape[1]) if state_matrix is not None else 0,
        "state_vector_names": state_names,
        "written_videos": written_videos,
        "written_image_archives": image_archive_paths,
        "discovery": episode.discovery,
        "metadata": episode.metadata,
        "extraction_notes": episode.extraction_notes,
    }
    write_json(episode_dir / "meta.json", episode_meta)
    return episode_meta


def build_dataset_manifest(
    episodes: list[EpisodeData],
    file_results: list[FileProcessResult],
    task_name: str,
    robot_name: str,
    fps: int,
    output_root: Path,
    native_backend: NativeBackendResult,
) -> dict[str, Any]:
    """Build a dataset manifest aggregated across all processed episodes."""

    unique_image_keys = sorted({key for episode in episodes for key in episode.obs_images})
    unique_lowdim_keys = sorted({key for episode in episodes for key in episode.obs_lowdim})
    action_shapes = sorted({tuple(episode.actions.shape[1:]) for episode in episodes})
    reward_available = all(np.any(np.abs(episode.rewards) > 0) or episode.discovery["reward_path"] is not None for episode in episodes)
    done_available = all(episode.discovery["done_path"] is not None for episode in episodes)

    manifest = {
        "task_name": task_name,
        "robot_name": robot_name,
        "fps": fps,
        "dataset_root": str(output_root),
        "episode_count": len(episodes),
        "total_frames": int(sum(episode.length for episode in episodes)),
        "action_shapes": [list(shape) for shape in action_shapes],
        "observation_keys_found": sorted({key for episode in episodes for key in episode.discovery["observation_keys"]}),
        "image_keys_found": unique_image_keys,
        "lowdim_keys_found": unique_lowdim_keys,
        "reward_availability": reward_available,
        "done_availability": done_available,
        "primary_image_keys": sorted({episode.primary_image_key for episode in episodes if episode.primary_image_key}),
        "files_processed": [asdict(result) for result in file_results],
        "native_backend": asdict(native_backend),
    }
    return manifest


def build_dataset_summary(manifest: dict[str, Any], episode_meta_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a concise human-readable dataset summary."""

    summary = {
        "episode_count": manifest["episode_count"],
        "total_frames": manifest["total_frames"],
        "fps": manifest["fps"],
        "robot_name": manifest["robot_name"],
        "task_name": manifest["task_name"],
        "action_shapes": manifest["action_shapes"],
        "image_keys_found": manifest["image_keys_found"],
        "lowdim_keys_found": manifest["lowdim_keys_found"],
        "per_episode": [
            {
                "episode_id": meta["episode_id"],
                "length": meta["length"],
                "action_shape": meta["action_shape"],
                "primary_image_key": meta["primary_image_key"],
                "image_keys": meta["image_keys"],
                "lowdim_keys": meta["lowdim_keys"],
            }
            for meta in episode_meta_list
        ],
    }
    return summary


def resolve_hf_token(explicit_token: str | None) -> str | None:
    """Resolve the Hugging Face token from CLI, env, or the top-level constant."""

    if explicit_token:
        return explicit_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token
    if HF_TOKEN and HF_TOKEN != "PUT_TEMP_TOKEN_HERE":
        return HF_TOKEN
    return None


def create_dataset_readme(output_root: Path, summary: dict[str, Any], repo_id: str | None) -> None:
    """Create a simple dataset card / README at the output root."""

    lines = [
        "---",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: data/*/*.parquet",
        "---",
        "",
        f"# {repo_id or summary['task_name']}",
        "",
        "Generated from LIBERO / robosuite teleoperation HDF5 demos.",
        "",
        "## Summary",
        "",
        f"- Task: `{summary['task_name']}`",
        f"- Robot: `{summary['robot_name']}`",
        f"- Episodes: `{summary['episode_count']}`",
        f"- Total frames: `{summary['total_frames']}`",
        f"- FPS: `{summary['fps']}`",
        f"- Action shapes: `{summary['action_shapes']}`",
        f"- Image keys: `{summary['image_keys_found']}`",
        f"- Low-dimensional keys: `{summary['lowdim_keys_found']}`",
        "",
        "## Files",
        "",
        "- `data/`: native LeRobot frame parquet files",
        "- `meta/`: native LeRobot metadata, episode metadata, stats, and task tables",
        "- `images/` and `videos/`: native LeRobot visual assets when image observations are present",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_import_lerobot(script_root: Path) -> tuple[Any | None, str]:
    """Try to import the local or installed LeRobot dataset writer."""

    local_src = script_root / "libero_smolvla_eval" / "lerobot" / "src"
    if local_src.exists():
        sys.path.insert(0, str(local_src))

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # noqa: BLE001
        return None, f"LeRobot import failed: {exc}"
    return LeRobotDataset, "LeRobotDataset imported successfully."


def build_state_schema(episodes: list[EpisodeData]) -> tuple[list[str], dict[int, np.ndarray]]:
    """Build a consistent flattened state vector schema across all episodes."""

    ordered_names: OrderedDict[str, None] = OrderedDict()
    per_episode_components: dict[int, dict[str, np.ndarray]] = {}

    for episode in episodes:
        components: dict[str, np.ndarray] = {}
        for key in sorted(episode.obs_lowdim):
            value = np.asarray(episode.obs_lowdim[key], dtype=np.float32)
            if value.ndim == 1:
                value = value.reshape(value.shape[0], 1)
            else:
                value = value.reshape(value.shape[0], -1)
            for idx in range(value.shape[1]):
                name = key if value.shape[1] == 1 else f"{key}[{idx}]"
                ordered_names.setdefault(name, None)
                components[name] = value[:, idx]
        per_episode_components[episode.episode_id] = components

    ordered_list = list(ordered_names.keys())
    state_matrices: dict[int, np.ndarray] = {}
    for episode in episodes:
        matrix = np.zeros((episode.length, len(ordered_list)), dtype=np.float32)
        components = per_episode_components[episode.episode_id]
        for col_idx, name in enumerate(ordered_list):
            if name in components:
                matrix[:, col_idx] = components[name]
        state_matrices[episode.episode_id] = matrix
    return ordered_list, state_matrices


def build_image_schema(episodes: list[EpisodeData]) -> tuple[dict[str, tuple[int, int, int]], list[str]]:
    """Build a consistent image feature schema across all episodes."""

    schema: dict[str, tuple[int, int, int]] = {}
    mismatches: list[str] = []
    for episode in episodes:
        for key, frames in episode.obs_images.items():
            frame_shape = tuple(int(dim) for dim in frames.shape[1:])
            if key not in schema:
                schema[key] = frame_shape
            elif schema[key] != frame_shape:
                mismatches.append(
                    f"Image key '{key}' has inconsistent frame shape: {schema[key]} vs {frame_shape}"
                )
    return schema, mismatches


def write_native_lerobot_dataset(
    episodes: list[EpisodeData],
    output_root: Path,
    repo_id: str | None,
    robot_name: str,
    task_name: str,
    fps: int,
    script_root: Path,
) -> NativeBackendResult:
    """Write a native LeRobot dataset if the local API is importable."""

    LeRobotDataset, reason = maybe_import_lerobot(script_root)
    if LeRobotDataset is None:
        LOGGER.warning("%s", reason)
        return NativeBackendResult(used=False, reason=reason)

    state_names, state_matrices = build_state_schema(episodes)
    image_schema, mismatches = build_image_schema(episodes)
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        LOGGER.warning("Skipping native LeRobot writer because image schema is inconsistent: %s", mismatch_text)
        return NativeBackendResult(used=False, reason=mismatch_text)

    action_dim = int(episodes[0].actions.shape[1])
    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{idx}" for idx in range(action_dim)],
        }
    }
    if state_names:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        }
    for key, shape in sorted(image_schema.items()):
        features[f"observation.images.{sanitize_key(key)}"] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    local_repo_id = repo_id or f"local/{sanitize_key(task_name)}"
    native_temp_root = output_root.parent / f"{output_root.name}.__native_tmp__"
    if native_temp_root.exists():
        shutil.rmtree(native_temp_root)
    LOGGER.info("Writing native LeRobot dataset via temp root %s with repo_id=%s", native_temp_root, local_repo_id)

    dataset = LeRobotDataset.create(
        repo_id=local_repo_id,
        root=native_temp_root,
        fps=fps,
        features=features,
        robot_type=robot_name,
        use_videos=False,
    )

    try:
        for episode in episodes:
            state_matrix = state_matrices.get(episode.episode_id)
            for step_idx in range(episode.length):
                frame: dict[str, Any] = {
                    "action": episode.actions[step_idx].astype(np.float32),
                    "task": task_name,
                }
                if state_matrix is not None and state_matrix.size > 0:
                    frame["observation.state"] = state_matrix[step_idx].astype(np.float32)

                for key, shape in image_schema.items():
                    feature_name = f"observation.images.{sanitize_key(key)}"
                    if key in episode.obs_images:
                        frame[feature_name] = episode.obs_images[key][step_idx]
                    else:
                        frame[feature_name] = np.zeros(shape, dtype=np.uint8)

                dataset.add_frame(frame)
            dataset.save_episode()
        dataset.finalize()
    except Exception as exc:  # noqa: BLE001
        try:
            dataset.finalize()
        except Exception:  # noqa: BLE001
            pass
        if native_temp_root.exists():
            shutil.rmtree(native_temp_root, ignore_errors=True)
        reason = f"Native LeRobot writer failed: {exc}"
        LOGGER.warning("%s", reason)
        return NativeBackendResult(used=False, reason=reason)

    for subdir_name in ("data", "meta", "images", "videos"):
        src = native_temp_root / subdir_name
        dst = output_root / subdir_name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    shutil.rmtree(native_temp_root, ignore_errors=True)

    validation_details = {
        "repo_id": local_repo_id,
        "features": list(features.keys()),
        "state_dim": len(state_names),
        "image_keys": sorted(image_schema.keys()),
    }
    return NativeBackendResult(used=True, reason="Native LeRobot dataset written successfully.", details=validation_details)


def validate_dataset(output_root: Path, manifest: dict[str, Any], native_backend: NativeBackendResult) -> dict[str, Any]:
    """Validate the generated dataset layout."""

    episodes_dir = output_root / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Missing fallback episodes directory: {episodes_dir}")

    episode_dirs = sorted(path for path in episodes_dir.iterdir() if path.is_dir())
    if not episode_dirs:
        raise ValueError("Validation failed: no episode directories were created.")

    metadata_path = output_root / "dataset_manifest.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Validation failed: missing metadata file {metadata_path}")

    action_shapes: set[tuple[int, ...]] = set()
    image_backed_episode_count = 0
    for episode_dir in episode_dirs:
        actions_path = episode_dir / "actions.npy"
        rewards_path = episode_dir / "rewards.npy"
        dones_path = episode_dir / "dones.npy"
        meta_path = episode_dir / "meta.json"

        for required_path in (actions_path, rewards_path, dones_path, meta_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Validation failed: missing required episode file {required_path}")

        actions = np.load(actions_path)
        rewards = np.load(rewards_path)
        dones = np.load(dones_path)
        if actions.shape[0] == 0 or rewards.shape[0] == 0 or dones.shape[0] == 0:
            raise ValueError(f"Validation failed: zero-length arrays found in {episode_dir}")
        if not (actions.shape[0] == rewards.shape[0] == dones.shape[0]):
            raise ValueError(f"Validation failed: inconsistent array lengths in {episode_dir}")

        action_shapes.add(tuple(int(dim) for dim in actions.shape[1:]))

        has_video = any(path.suffix == ".mp4" for path in episode_dir.iterdir())
        has_image_archive = (episode_dir / "obs_images").is_dir() and any((episode_dir / "obs_images").iterdir())
        if has_video or has_image_archive:
            image_backed_episode_count += 1

    if len(action_shapes) != 1:
        raise ValueError(f"Validation failed: inconsistent action shapes across episodes: {sorted(action_shapes)}")

    if manifest["image_keys_found"] and image_backed_episode_count == 0:
        raise ValueError("Validation failed: image observations were discovered but no image-backed outputs exist.")

    native_validated = False
    native_error: str | None = None
    if native_backend.used:
        try:
            LeRobotDataset, reason = maybe_import_lerobot(Path(__file__).resolve().parent)
            if LeRobotDataset is None:
                native_error = reason
            else:
                repo_id = native_backend.details.get("repo_id")
                dataset = LeRobotDataset(repo_id=repo_id, root=output_root)
                if len(dataset) == 0:
                    raise ValueError("Native LeRobot dataset loaded but contains zero frames.")
                native_validated = True
        except Exception as exc:  # noqa: BLE001
            native_error = str(exc)

    validation = {
        "episode_count": len(episode_dirs),
        "action_shape": list(next(iter(action_shapes))),
        "image_backed_episode_count": image_backed_episode_count,
        "native_backend_used": native_backend.used,
        "native_backend_validated": native_validated,
        "native_backend_validation_error": native_error,
    }
    LOGGER.info("Validation summary: %s", validation)
    return validation


def push_dataset_to_hub(
    output_root: Path,
    repo_id: str,
    token: str,
    private: bool,
    native_backend: NativeBackendResult,
) -> None:
    """Create or update a Hugging Face dataset repo and upload only native LeRobot files."""

    if not token:
        raise ValueError("No Hugging Face token available. Set --hf-token or edit HF_TOKEN near the top of the script.")
    if not native_backend.used:
        raise RuntimeError(
            "Refusing to push because native LeRobot export was not created successfully. "
            "This avoids uploading fallback-only data to the Hub."
        )

    hub_module = ensure_dependency("huggingface_hub", "pip install huggingface_hub")
    HfApi = hub_module.HfApi

    api = HfApi(token=token)
    LOGGER.info("Ensuring dataset repo exists on the Hub: %s", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    LOGGER.info("Uploading dataset folder to Hugging Face Hub from %s using upload_large_folder", output_root)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_root),
        ignore_patterns=[
            "episodes/**",
            "inspection/**",
            "dataset_manifest.json",
            "dataset_summary.json",
            "validation_report.json",
            "files_processed.json",
        ],
        token=token,
    )
    LOGGER.info("Upload completed for dataset repo: %s", repo_id)


def process_file(
    file_path: Path,
    episode_start_id: int,
    preferred_image_key: str | None,
    output_root: Path,
    fps: int,
    write_videos: bool,
) -> tuple[list[EpisodeData], list[dict[str, Any]], FileProcessResult]:
    """Inspect and convert one source HDF5 file."""

    h5py = ensure_dependency("h5py", "pip install h5py")
    result = FileProcessResult(file_path=str(file_path), success=False, discovered_demo_groups=[])

    LOGGER.info("Inspecting file: %s", file_path)
    with h5py.File(file_path, "r") as h5_file:
        tree_lines = print_hdf5_tree(h5_file, file_path)
        inspection_path = output_root / "inspection" / f"{sanitize_key(file_path.stem)}_tree.txt"
        inspection_path.parent.mkdir(parents=True, exist_ok=True)
        inspection_path.write_text("\n".join(tree_lines) + "\n", encoding="utf-8")

        demo_specs = discover_demo_groups(h5_file)
        result.discovered_demo_groups = [spec.group_path for spec in demo_specs]
        if not demo_specs:
            raise ValueError(f"No episode/demo groups discovered in file: {file_path}")

        episodes: list[EpisodeData] = []
        episode_metas: list[dict[str, Any]] = []
        next_episode_id = episode_start_id
        for demo_spec in demo_specs:
            episode = load_demo_episode(
                h5_file=h5_file,
                file_path=file_path,
                demo_spec=demo_spec,
                episode_id=next_episode_id,
                preferred_image_key=preferred_image_key,
            )
            episode_meta = write_fallback_episode(episode, output_root, fps=fps, write_videos=write_videos)
            episodes.append(episode)
            episode_metas.append(episode_meta)
            result.created_episode_ids.append(next_episode_id)
            next_episode_id += 1

    result.success = True
    return episodes, episode_metas, result


def main() -> None:
    """Main entry point."""

    args = parse_args()
    setup_logging(args.log_level)

    write_videos = not args.skip_videos
    token = resolve_hf_token(args.hf_token)
    script_root = Path(__file__).resolve().parent

    ensure_clean_output_dir(args.output, overwrite=args.overwrite)
    hdf5_files = find_hdf5_files(args.input)
    LOGGER.info("Found %d HDF5 file(s) to process.", len(hdf5_files))

    all_episodes: list[EpisodeData] = []
    all_episode_metas: list[dict[str, Any]] = []
    file_results: list[FileProcessResult] = []
    next_episode_id = 0

    for file_path in hdf5_files:
        try:
            episodes, episode_metas, result = process_file(
                file_path=file_path,
                episode_start_id=next_episode_id,
                preferred_image_key=args.primary_image_key,
                output_root=args.output,
                fps=args.fps,
                write_videos=write_videos,
            )
            all_episodes.extend(episodes)
            all_episode_metas.extend(episode_metas)
            file_results.append(result)
            next_episode_id += len(episodes)
        except Exception as exc:  # noqa: BLE001
            error_text = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Failed processing %s: %s", file_path, error_text)
            LOGGER.debug("Traceback for %s:\n%s", file_path, traceback.format_exc())
            failure_result = FileProcessResult(
                file_path=str(file_path),
                success=False,
                discovered_demo_groups=[],
                created_episode_ids=[],
                errors=[error_text],
            )
            file_results.append(failure_result)
            if args.fail_fast:
                raise

    if not all_episodes:
        raise RuntimeError("No episodes were created. Inspect the logs and the inspection/ tree dumps.")

    native_backend = NativeBackendResult(used=False, reason="Native writer was not attempted.")
    if args.disable_native_writer:
        native_backend = NativeBackendResult(used=False, reason="Disabled by --disable-native-writer.")
    else:
        native_backend = write_native_lerobot_dataset(
            episodes=all_episodes,
            output_root=args.output,
            repo_id=args.repo_id,
            robot_name=args.robot_name,
            task_name=args.task_name,
            fps=args.fps,
            script_root=script_root,
        )

    manifest = build_dataset_manifest(
        episodes=all_episodes,
        file_results=file_results,
        task_name=args.task_name,
        robot_name=args.robot_name,
        fps=args.fps,
        output_root=args.output,
        native_backend=native_backend,
    )
    summary = build_dataset_summary(manifest, all_episode_metas)
    create_dataset_readme(args.output, summary, args.repo_id)
    write_json(args.output / "dataset_manifest.json", manifest)
    write_json(args.output / "dataset_summary.json", summary)
    write_json(args.output / "files_processed.json", {"files": [asdict(result) for result in file_results]})

    validation = validate_dataset(args.output, manifest, native_backend)
    write_json(args.output / "validation_report.json", validation)

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo-id is required when using --push-to-hub.")
        push_dataset_to_hub(
            args.output,
            repo_id=args.repo_id,
            token=token or "",
            private=args.private,
            native_backend=native_backend,
        )

    LOGGER.info("Finished dataset build.")
    LOGGER.info("Output root: %s", args.output)
    LOGGER.info(
        "Final summary | episodes=%d | total_frames=%d | action_shapes=%s | native_backend_used=%s",
        manifest["episode_count"],
        manifest["total_frames"],
        manifest["action_shapes"],
        native_backend.used,
    )


if __name__ == "__main__":
    main()
