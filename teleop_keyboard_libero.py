#!/usr/bin/env python3
import argparse
import ctypes
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import mujoco.viewer

from libero_demo_hdf5_images import augment_demo_hdf5_with_images
from libero.libero.benchmark import get_benchmark, get_benchmark_dict
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.devices import Keyboard
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper


LOCAL_QT_LIBDIR = Path("/home/kgaer/robotics/.local_qt_libs/usr/lib/x86_64-linux-gnu")
SYSTEM_QT_FONTDIR = Path("/usr/share/fonts/truetype/dejavu")


def setup_local_qt_runtime():
    if not LOCAL_QT_LIBDIR.exists():
        pass
    else:
        libdir = str(LOCAL_QT_LIBDIR)
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = libdir if not current else f"{libdir}:{current}"
        for libname in ("libICE.so.6", "libSM.so.6"):
            libpath = LOCAL_QT_LIBDIR / libname
            if libpath.exists():
                ctypes.CDLL(str(libpath), mode=ctypes.RTLD_GLOBAL)

    if SYSTEM_QT_FONTDIR.exists():
        os.environ.setdefault("QT_QPA_FONTDIR", str(SYSTEM_QT_FONTDIR))


setup_local_qt_runtime()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teleoperate a LIBERO task with robosuite's Keyboard device and robosuite's onscreen render loop."
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--viewer", choices=["auto", "robosuite", "mujoco"], default="auto")
    parser.add_argument("--output-dir", default="./libero_task0_demos")
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


class TeleopEnvAdapter:
    def __init__(self, env):
        self.env = env

    def _check_success(self):
        return self.env.check_success()

    def render(self):
        return self.env.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)


def qxcb_missing():
    if (LOCAL_QT_LIBDIR / "libICE.so.6").exists() and (LOCAL_QT_LIBDIR / "libSM.so.6").exists():
        return False
    plugin = Path(
        "/home/kgaer/robotics/libero_smolvla_eval/.venv/lib/python3.11/site-packages/cv2/qt/plugins/platforms/libqxcb.so"
    )
    if not plugin.exists():
        return False
    try:
        import subprocess

        result = subprocess.run(["ldd", str(plugin)], check=False, capture_output=True, text=True)
    except Exception:
        return False
    return "libSM.so.6 => not found" in result.stdout or "libICE.so.6 => not found" in result.stdout


def make_env(args, bddl_path, has_renderer):
    env = ControlEnv(
        bddl_file_name=bddl_path,
        robots=[args.robot],
        controller=args.controller,
        use_camera_obs=False,
        has_renderer=has_renderer,
        has_offscreen_renderer=not has_renderer,
        render_camera=args.camera,
        ignore_done=True,
        hard_reset=False,
    )
    return TeleopEnvAdapter(env)


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
            "viewer": args.viewer,
            "image_size": args.image_size,
        }
    )


def compact_successes(raw_dir: Path, session_dir: Path, env_info: str, image_size: int):
    state_files = list(raw_dir.glob("ep_*/state_*.npz"))
    hdf5_path = session_dir / "demo.hdf5"
    if not state_files:
        return hdf5_path, 0
    gather_demonstrations_as_hdf5(str(raw_dir), str(session_dir), env_info)
    if not hdf5_path.exists():
        return hdf5_path, 0
    augmented = augment_demo_hdf5_with_images(hdf5_path, env_info, image_size=image_size)
    if augmented:
        print(f"[images] added image observations to {augmented} demo(s)")
    with h5py.File(hdf5_path, "r") as f:
        return hdf5_path, len(f["data"].keys())


def finalize_current_episode(env):
    if getattr(env, "has_interaction", False):
        env._flush()
        env.has_interaction = False


def run_loop_with_robosuite_viewer(env, device, init_states, session_dir, env_info, image_size: int):
    reset_index = 0
    saved_count = 0
    success_latched = False
    while True:
        env.reset()
        env.set_init_state(init_states[reset_index % len(init_states)])
        env.render()
        print("[ready] robosuite viewer active; focus the window to drive the robot")
        device.start_control()
        success_latched = False

        while True:
            action, _ = input2action(
                device=device,
                robot=env.robots[0],
                active_arm="right",
                env_configuration=None,
            )
            if action is None:
                finalize_current_episode(env)
                hdf5_path, new_count = compact_successes(Path(env.directory), session_dir, env_info, image_size)
                if new_count > saved_count:
                    print(f"[saved] saved successfully: {hdf5_path} demos={new_count}")
                else:
                    print(f"[saved] no new successful demo saved: {hdf5_path} demos={new_count}")
                saved_count = new_count
                reset_index += 1
                print(f"[reset] moving to init state {reset_index % len(init_states)}")
                break
            env.step(action)
            env.render()
            if env.check_success() and not success_latched:
                print("[success] task satisfied; press q to reset")
                success_latched = True


def run_loop_with_mujoco_viewer(env, device, init_states, session_dir, env_info, image_size: int):
    reset_index = 0
    saved_count = 0
    success_latched = False
    env.reset()
    env.set_init_state(init_states[0])
    device.start_control()

    with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data) as viewer:
        print("[warn] falling back to native MuJoCo viewer because robosuite's Qt/OpenCV viewer is unavailable")
        print("[ready] viewer active; focus the window to drive the robot")
        success_latched = False
        while viewer.is_running():
            action, _ = input2action(
                device=device,
                robot=env.robots[0],
                active_arm="right",
                env_configuration=None,
            )
            if action is None:
                finalize_current_episode(env)
                hdf5_path, new_count = compact_successes(Path(env.directory), session_dir, env_info, image_size)
                if new_count > saved_count:
                    print(f"[saved] saved successfully: {hdf5_path} demos={new_count}")
                else:
                    print(f"[saved] no new successful demo saved: {hdf5_path} demos={new_count}")
                saved_count = new_count
                reset_index += 1
                env.reset()
                env.set_init_state(init_states[reset_index % len(init_states)])
                device.start_control()
                print(f"[reset] moving to init state {reset_index % len(init_states)}")
                viewer.sync()
                success_latched = False
                continue
            env.step(action)
            viewer.sync()
            if env.check_success() and not success_latched:
                print("[success] task satisfied; press q to reset")
                success_latched = True
            time.sleep(0.01)


def main():
    args = parse_args()
    benchmark_map = get_benchmark_dict()
    if args.suite.lower() not in benchmark_map:
        print(f"[error] unknown suite {args.suite}. choices={sorted(benchmark_map)}", file=sys.stderr)
        return 2

    benchmark = benchmark_map[args.suite.lower()](args.task_order_index)
    if not 0 <= args.task_id < benchmark.get_num_tasks():
        print(f"[error] task id {args.task_id} out of range 0..{benchmark.get_num_tasks() - 1}", file=sys.stderr)
        return 2

    task = benchmark.get_task(args.task_id)
    bddl_path = benchmark.get_task_bddl_file_path(args.task_id)
    init_states = benchmark.get_task_init_states(args.task_id)

    print("Resolved task:")
    print(f"  suite={args.suite}")
    print(f"  task_id={args.task_id}")
    print(f"  task_name={task.name}")
    print(f"  language={task.language}")
    print(f"  bddl={bddl_path}")
    print("")
    print("Teleoperation path:")
    print("  device=robosuite.devices.Keyboard")
    print("  action_mapping=robosuite.utils.input_utils.input2action")
    print(f"  viewer_mode={args.viewer}")
    print("")
    output_dir = Path(args.output_dir).resolve()
    session_dir = make_session_dir(output_dir, args.suite, args.task_id)
    raw_dir = Path(tempfile.mkdtemp(prefix="libero_raw_", dir=str(session_dir)))
    env_info = build_env_info(args, benchmark, task)
    print(f"session_dir={session_dir}")
    print(f"raw_dir={raw_dir}")
    print("")
    device = Keyboard(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )
    if args.viewer == "mujoco" or (args.viewer == "auto" and qxcb_missing()):
        env = DataCollectionWrapper(make_env(args, bddl_path, has_renderer=False), str(raw_dir))
        run_loop_with_mujoco_viewer(env, device, init_states, session_dir, env_info, args.image_size)
        return 0

    env = DataCollectionWrapper(make_env(args, bddl_path, has_renderer=True), str(raw_dir))
    run_loop_with_robosuite_viewer(env, device, init_states, session_dir, env_info, args.image_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
