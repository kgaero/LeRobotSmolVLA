#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from libero.libero.envs.env_wrapper import ControlEnv


CAMERA_SPECS = (
    ("agentview", "image"),
    ("robot0_eye_in_hand", "wrist_image"),
)


def _ensure_mujoco_gl_backend() -> str:
    backend = os.environ.get("MUJOCO_GL")
    if backend:
        return backend

    # Offscreen replay needs an explicit backend here. On desktop / WSLg sessions
    # GLX works reliably, while headless environments should use EGL.
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        backend = "glx"
    else:
        backend = "egl"
    os.environ["MUJOCO_GL"] = backend
    return backend


def _build_replay_env(env_info: dict, image_size: int) -> ControlEnv:
    return ControlEnv(
        bddl_file_name=env_info["bddl"],
        robots=[env_info.get("robot", "Panda")],
        controller=env_info.get("controller", "OSC_POSE"),
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=[camera_name for camera_name, _ in CAMERA_SPECS],
        camera_heights=image_size,
        camera_widths=image_size,
        ignore_done=True,
        hard_reset=False,
    )


def _render_camera_sequences(env: ControlEnv, model_xml: str, states: np.ndarray) -> dict[str, np.ndarray]:
    sequences = {dataset_name: [] for _, dataset_name in CAMERA_SPECS}

    env.reset_from_xml_string(model_xml)
    env.sim.reset()

    for state in states:
        env.sim.set_state_from_flattened(np.asarray(state))
        env.sim.forward()
        # Camera observables are cached inside robosuite. When replaying saved states
        # without stepping the environment, force_update is required or every frame
        # can repeat the initial cached image.
        obs = env.env._get_observations(force_update=True)
        for camera_name, dataset_name in CAMERA_SPECS:
            frame = np.asarray(obs[f"{camera_name}_image"], dtype=np.uint8)
            sequences[dataset_name].append(frame)

    return {key: np.stack(frames, axis=0) for key, frames in sequences.items()}


def augment_demo_hdf5_with_images(hdf5_path: Path, env_info_json: str, image_size: int = 256) -> int:
    if not hdf5_path.exists():
        return 0

    backend = _ensure_mujoco_gl_backend()
    env_info = json.loads(env_info_json)
    print(f"[images] preparing offscreen replay with MUJOCO_GL={backend}")
    env = _build_replay_env(env_info, image_size=image_size)
    augmented_count = 0

    try:
        with h5py.File(hdf5_path, "r+") as hdf5_file:
            data_group = hdf5_file["data"]
            for demo_name in sorted(data_group.keys()):
                demo_group = data_group[demo_name]
                obs_group = demo_group.require_group("obs")
                if all(dataset_name in obs_group for _, dataset_name in CAMERA_SPECS):
                    print(f"[images] {demo_name}: image datasets already present, skipping")
                    continue

                model_xml = demo_group.attrs["model_file"]
                states = np.asarray(demo_group["states"][()])
                print(
                    f"[images] {demo_name}: rendering {len(states)} frames at "
                    f"{image_size}x{image_size} for {[dataset_name for _, dataset_name in CAMERA_SPECS]}"
                )
                image_sequences = _render_camera_sequences(env, model_xml, states)

                for dataset_name, frames in image_sequences.items():
                    if dataset_name in obs_group:
                        del obs_group[dataset_name]
                    obs_group.create_dataset(dataset_name, data=frames, compression="gzip")
                    print(f"[images] {demo_name}: wrote {dataset_name} with shape {frames.shape}")

                augmented_count += 1
                print(f"[images] {demo_name}: completed")
    finally:
        env.close()

    print(f"[images] augmentation finished for {augmented_count} demo(s)")
    return augmented_count
