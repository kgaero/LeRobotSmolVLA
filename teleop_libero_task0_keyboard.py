#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer

from libero.libero.benchmark import get_benchmark, get_benchmark_dict
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.devices import Keyboard
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper


class TeleopEnvAdapter:
    def __init__(self, env):
        self.env = env

    def _check_success(self):
        return self.env.check_success()

    def __getattr__(self, name):
        return getattr(self.env, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keyboard teleoperation for an installed LIBERO task using a native MuJoCo viewer."
    )
    parser.add_argument("--suite", default="libero_spatial", help="LIBERO benchmark suite name.")
    parser.add_argument("--task-id", type=int, default=0, help="Task id inside the chosen suite.")
    parser.add_argument("--task-order-index", type=int, default=0, help="Task ordering index for 10-task suites.")
    parser.add_argument("--camera", default="frontview", help="Fixed MuJoCo camera name, or 'free'.")
    parser.add_argument("--controller", default="OSC_POSE", help="Robot controller name.")
    parser.add_argument("--robot", default="Panda", help="Robot name.")
    parser.add_argument("--output-dir", default="./libero_task0_demos", help="Base directory for saved demos.")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="Keyboard position sensitivity.")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="Keyboard rotation sensitivity.")
    parser.add_argument("--auto-quit-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser.parse_args()


def print_task_metadata(suite_name, task_id, benchmark, task):
    print("Resolved task:")
    print(f"  suite: {suite_name}")
    print(f"  task_id: {task_id}")
    print(f"  task_name: {task.name}")
    print(f"  language: {task.language}")
    print(f"  bddl: {benchmark.get_task_bddl_file_path(task_id)}")
    print(f"  demo_relpath: {benchmark.get_task_demonstration(task_id)}")
    print(f"  init_states: {len(benchmark.get_task_init_states(task_id))}")
    print("")


def print_controls():
    print("Keyboard teleop controls:")
    print("  w/a/s/d: move end-effector in x/y")
    print("  r/f: move end-effector up/down")
    print("  z/x: rotate about x-axis")
    print("  t/g: rotate about y-axis")
    print("  c/v: rotate about z-axis")
    print("  space: toggle gripper open/close")
    print("  q: end episode and reset to a new initial state")
    print("  Ctrl-C in terminal or close viewer window: quit")
    print("")
    print("Saving behavior:")
    print("  Successful episodes are compacted into demo.hdf5 whenever you reset with q.")
    print("  The same compaction runs once more on exit.")
    print("")
    print("Viewer:")
    print("  Native MuJoCo viewer window should open.")
    print("  Mouse interaction for camera is available in free-camera mode.")
    print("")


def make_session_dir(base_dir: Path, suite_name: str, task_id: int) -> Path:
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"{suite_name}_task{task_id}_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def build_env_info(args, benchmark, task):
    return json.dumps(
        {
            "suite": args.suite,
            "task_id": args.task_id,
            "task_name": task.name,
            "language": task.language,
            "bddl": benchmark.get_task_bddl_file_path(args.task_id),
            "robot": args.robot,
            "controller": args.controller,
            "camera": args.camera,
        }
    )


def compact_successes(raw_dir: Path, session_dir: Path, env_info: str) -> int:
    if not any(raw_dir.glob("ep_*/state_*.npz")):
        return 0
    gather_demonstrations_as_hdf5(str(raw_dir), str(session_dir), env_info)
    hdf5_path = session_dir / "demo.hdf5"
    if not hdf5_path.exists():
        return 0
    with h5py.File(hdf5_path, "r") as f:
        return len(f["data"].keys())


def apply_viewer_camera(viewer, env, camera_name: str):
    if camera_name == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        return
    try:
        camera_id = env.sim.model.camera_name2id(camera_name)
    except Exception:
        print(f"[warn] camera '{camera_name}' not found, falling back to free camera")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        return
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def reset_to_init_state(env, init_states, init_index: int):
    env.reset()
    env.set_init_state(init_states[init_index % len(init_states)])


def main():
    args = parse_args()
    available_suites = sorted(get_benchmark_dict().keys())
    if args.suite.lower() not in get_benchmark_dict():
        print(f"[error] suite '{args.suite}' not found. Available suites: {available_suites}", file=sys.stderr)
        return 2

    benchmark = get_benchmark(args.suite)(args.task_order_index)
    if not 0 <= args.task_id < benchmark.get_num_tasks():
        print(
            f"[error] task-id {args.task_id} out of range for suite '{args.suite}' "
            f"(0..{benchmark.get_num_tasks() - 1})",
            file=sys.stderr,
        )
        return 2

    task = benchmark.get_task(args.task_id)
    init_states = benchmark.get_task_init_states(args.task_id)
    bddl_path = benchmark.get_task_bddl_file_path(args.task_id)

    print_task_metadata(args.suite, args.task_id, benchmark, task)
    print_controls()

    output_dir = Path(args.output_dir).resolve()
    session_dir = make_session_dir(output_dir, args.suite, args.task_id)
    raw_dir = Path(tempfile.mkdtemp(prefix="libero_raw_", dir=str(session_dir)))
    env_info = build_env_info(args, benchmark, task)

    print(f"Session directory: {session_dir}")
    print(f"Raw episode directory: {raw_dir}")
    print("")

    env = ControlEnv(
        bddl_file_name=bddl_path,
        robots=[args.robot],
        controller=args.controller,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        ignore_done=True,
        hard_reset=False,
    )
    env = TeleopEnvAdapter(env)
    env = DataCollectionWrapper(env, str(raw_dir))

    device = Keyboard(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )

    init_index = 0
    reset_to_init_state(env, init_states, init_index)
    device.start_control()

    successful_printed = False
    save_count = 0

    try:
        with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data) as viewer:
            apply_viewer_camera(viewer, env, args.camera)
            viewer.sync()
            print("Viewer started. Use the terminal for logs and Ctrl-C to quit.")
            start_time = time.time()

            while viewer.is_running():
                if args.auto_quit_seconds > 0 and (time.time() - start_time) >= args.auto_quit_seconds:
                    print("[quit] auto-quit timer reached")
                    break

                action, _ = input2action(device=device, robot=env.robots[0], active_arm="right", env_configuration=None)

                if action is None:
                    save_count = compact_successes(raw_dir, session_dir, env_info)
                    print(f"[reset] successful demos saved so far: {save_count}")
                    init_index += 1
                    reset_to_init_state(env, init_states, init_index)
                    successful_printed = False
                    device.start_control()
                    viewer.sync()
                    continue

                env.step(action)
                viewer.sync()

                if env._check_success() and not successful_printed:
                    print("[success] task completed. Press q to reset and save this episode.")
                    successful_printed = True

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[quit] keyboard interrupt received, finalizing demo file...")
    finally:
        try:
            save_count = compact_successes(raw_dir, session_dir, env_info)
            print(f"[final] successful demos in {session_dir / 'demo.hdf5'}: {save_count}")
        finally:
            env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
