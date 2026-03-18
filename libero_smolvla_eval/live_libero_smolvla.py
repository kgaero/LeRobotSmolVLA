#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video


class MatplotlibViewer:
    def __init__(self, title: str):
        import matplotlib

        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt

        self._plt = plt
        self.fig, self.ax = plt.subplots(num=title)
        self.ax.axis("off")
        self.image_artist = None
        plt.show(block=False)

    def show(self, frame) -> None:
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(frame)
        else:
            self.image_artist.set_data(frame)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self) -> None:
        self._plt.close(self.fig)


class MujocoPassiveViewer(AbstractContextManager):
    def __init__(self, control_env, camera_name: str = "frontview"):
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self._camera_name = camera_name
        self._ctx = mujoco.viewer.launch_passive(control_env.sim.model._model, control_env.sim.data._data)
        self._viewer = self._ctx.__enter__()
        self._apply_camera(control_env)
        self.sync()

    def _apply_camera(self, control_env) -> None:
        if self._camera_name == "free":
            self._viewer.cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE
            return
        try:
            camera_id = control_env.sim.model.camera_name2id(self._camera_name)
        except Exception:
            self._viewer.cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE
            return
        self._viewer.cam.type = self._mujoco.mjtCamera.mjCAMERA_FIXED
        self._viewer.cam.fixedcamid = camera_id

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def sync(self) -> None:
        self._viewer.sync()

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def add_batch_dim_to_observation(observation):
    if isinstance(observation, dict):
        return {key: add_batch_dim_to_observation(value) for key, value in observation.items()}
    if isinstance(observation, np.ndarray):
        return np.expand_dims(observation, axis=0)
    return observation


def load_policy_rename_map(policy_path: Path) -> dict[str, str]:
    rename_map: dict[str, str] = {}

    policy_preprocessor_path = policy_path / "policy_preprocessor.json"
    if policy_preprocessor_path.exists():
        config = json.loads(policy_preprocessor_path.read_text(encoding="utf-8"))
        for step in config.get("steps", []):
            if step.get("registry_name") == "rename_observations_processor":
                rename_map.update(step.get("config", {}).get("rename_map", {}))
                break

    train_config_path = policy_path / "train_config.json"
    if train_config_path.exists():
        config = json.loads(train_config_path.read_text(encoding="utf-8"))
        rename_map.update(config.get("rename_map", {}))

    # LIBERO's built-in env wrapper emits `image` and `image2`. Some training configs
    # store the wrist camera alias as `wrist_image`, so normalize that here.
    if "observation.images.wrist_image" in rename_map and "observation.images.image2" not in rename_map:
        rename_map["observation.images.image2"] = rename_map["observation.images.wrist_image"]

    return rename_map


def get_control_env(env):
    control_env = getattr(env, "_env", None)
    if control_env is None or not hasattr(control_env, "sim"):
        raise RuntimeError("Expected a LIBERO ControlEnv with a MuJoCo sim for on-screen viewing.")
    return control_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmolVLA on one LIBERO task with a live viewer window.")
    project_dir = Path.home() / "robotics" / "libero_smolvla_eval"
    parser.add_argument(
        "--policy-path",
        default=str(project_dir / "policies" / "HuggingFaceVLA_smolvla_libero"),
        help="Local path to the policy directory.",
    )
    parser.add_argument("--suite", default="libero_spatial", help="LIBERO suite name.")
    parser.add_argument("--task-id", type=int, default=0, help="Task id inside the suite.")
    parser.add_argument("--device", default="cuda", help="Torch device to use.")
    parser.add_argument("--seed", type=int, default=1000, help="Environment seed.")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override how many predicted actions are consumed before recomputing a new chunk. "
        "Higher values are much faster but less reactive.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the denoising steps used to predict an action chunk. Lower values are faster "
        "but may reduce policy quality.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap for testing the live viewer without running a full episode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "output_live"),
        help="Directory where the MP4 and summary JSON are written.",
    )
    parser.add_argument(
        "--viewer-backend",
        choices=("robosuite", "matplotlib", "mujoco"),
        default="robosuite",
        help="Live viewer backend. 'robosuite' uses LIBERO's on-screen renderer, "
        "'mujoco' uses the native MuJoCo passive viewer, and 'matplotlib' displays offscreen frames.",
    )
    parser.add_argument(
        "--viewer-camera",
        default="frontview",
        help="Camera name for the native MuJoCo viewer backend, or 'free'.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds after each environment step so the live viewer is easier to watch.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{args.suite}_{args.task_id}_live.mp4"
    summary_path = output_dir / f"{args.suite}_{args.task_id}_live.json"

    if args.viewer_backend in {"robosuite", "mujoco"}:
        # robosuite forces EGL when GPU rendering is enabled unless MUJOCO_GL is already glx or osmesa.
        # On WSLg, glx keeps the on-screen GLFW viewer path active.
        os.environ.setdefault("MUJOCO_GL", "glx")
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    print(f"Using MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)

    from lerobot.envs.libero import LiberoEnv as BaseLiberoEnv
    from lerobot.envs.libero import _get_suite

    class OnScreenLiberoEnv(BaseLiberoEnv):
        def _make_envs_task(self, task_suite, task_id: int = 0):
            from libero.libero import get_libero_path
            from libero.libero.envs.env_wrapper import ControlEnv

            task = task_suite.get_task(task_id)
            self.task = task.name
            self.task_description = task.language
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

            env = ControlEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=self.observation_height,
                camera_widths=self.observation_width,
                has_renderer=True,
                has_offscreen_renderer=True,
                render_camera="frontview",
            )
            env.reset()
            return env

    class RobosuiteOnScreenLiberoEnv(OnScreenLiberoEnv):
        def render(self):
            # Trigger robosuite's OpenCV window, then return the same frame used for MP4 export.
            self._env.env.render()
            return super().render()

    policy_path = Path(args.policy_path).expanduser().resolve()
    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    if args.n_action_steps is not None:
        policy_cfg.n_action_steps = args.n_action_steps
    if args.num_inference_steps is not None:
        policy_cfg.num_steps = args.num_inference_steps

    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task_id], control_mode="relative")

    rename_map = load_policy_rename_map(policy_path)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    print(f"Loaded policy from {policy_path}", flush=True)
    print(
        "Policy runtime config: "
        f"chunk_size={policy.config.chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"num_inference_steps={policy.config.num_steps}",
        flush=True,
    )
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    suite = _get_suite(args.suite)
    print(f"Loaded LIBERO suite {args.suite} task_id={args.task_id}", flush=True)
    env_cls = BaseLiberoEnv
    if args.viewer_backend == "robosuite":
        env_cls = RobosuiteOnScreenLiberoEnv
    elif args.viewer_backend == "mujoco":
        env_cls = OnScreenLiberoEnv
    env = env_cls(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=360,
        observation_height=360,
        init_states=True,
        n_envs=1,
        control_mode="relative",
    )

    policy.reset()
    obs, _ = env.reset(seed=args.seed)
    print("Environment reset complete", flush=True)

    frames = []
    success = False
    total_reward = 0.0
    steps = 0
    max_steps = env._max_episode_steps
    if args.max_steps is not None:
        max_steps = min(max_steps, args.max_steps)
    print(f"Running for at most {max_steps} steps", flush=True)

    first_frame = env.render()
    viewer = None
    if args.viewer_backend == "matplotlib":
        viewer = MatplotlibViewer(f"LeRobot LIBERO Live: {args.suite} task {args.task_id}")
        viewer.show(first_frame)
        print("Matplotlib live viewer opened", flush=True)
    elif args.viewer_backend == "mujoco":
        viewer = MujocoPassiveViewer(get_control_env(env), camera_name=args.viewer_camera)
        print("Native MuJoCo viewer opened", flush=True)
    else:
        print("robosuite on-screen viewer enabled", flush=True)
    frames.append(first_frame)

    with torch.inference_mode():
        while steps < max_steps and (viewer is None or not hasattr(viewer, "is_running") or viewer.is_running()):
            observation = preprocess_observation(add_batch_dim_to_observation(obs))
            observation["task"] = [env.task_description]
            observation = env_preprocessor(observation)
            observation = preprocessor(observation)

            action = policy.select_action(observation)
            action = postprocessor(action)
            action_transition = env_postprocessor({ACTION: action})
            action_np = action_transition[ACTION].to("cpu").numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += float(reward)
            steps += 1

            frame = env.render()
            if viewer is not None:
                if hasattr(viewer, "show"):
                    viewer.show(frame)
                if hasattr(viewer, "sync"):
                    viewer.sync()
            frames.append(frame)

            if info.get("is_success", False):
                success = True

            if terminated or truncated:
                break
            if args.step_delay > 0:
                time.sleep(args.step_delay)

    write_video(str(video_path), frames, fps=30)
    print(f"Saved video to {video_path}", flush=True)

    summary = {
        "suite": args.suite,
        "task_id": args.task_id,
        "success": success,
        "sum_reward": total_reward,
        "steps": steps,
        "video_path": str(video_path),
        "policy_path": str(policy_path),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "device": str(policy.config.device),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}", flush=True)

    if viewer is not None:
        viewer.close()
    env.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
