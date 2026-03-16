LeRobot LIBERO SmolVLA local setup

Project directory:
/home/kgaer/robotics/libero_smolvla_eval

What was installed:
- Python 3.11.15 in a local virtual environment at /home/kgaer/robotics/libero_smolvla_eval/.venv
- Hugging Face LeRobot repo checked out at tag v0.4.4 in /home/kgaer/robotics/libero_smolvla_eval/lerobot
- LeRobot installed editable with LIBERO support
- Extra runtime dependency added for the SmolVLA checkpoint: num2words
- SmolVLA LIBERO policy downloaded locally to /home/kgaer/robotics/libero_smolvla_eval/policies/HuggingFaceVLA_smolvla_libero
- LIBERO config written to /home/kgaer/.libero/config.yaml to avoid the first-import interactive prompt

Python version used:
- Python 3.11.15

How to activate the environment:
source /home/kgaer/robotics/libero_smolvla_eval/.venv/bin/activate

Rerun command:
/home/kgaer/robotics/libero_smolvla_eval/run_libero_smolvla.sh

Live interactive viewer command:
/home/kgaer/robotics/libero_smolvla_eval/run_libero_smolvla_live.sh

Evaluation that was run:
- Policy: /home/kgaer/robotics/libero_smolvla_eval/policies/HuggingFaceVLA_smolvla_libero
- Env type: libero
- Suite: libero_spatial
- Task ids: [0]
- Episodes: 1
- Batch size: 1
- Rendering backend: MUJOCO_GL=egl

Where videos are saved:
/home/kgaer/robotics/libero_smolvla_eval/output/videos/libero_spatial_0/eval_episode_0.mp4

Interactive rendering status:
- Saved MP4 output works.
- The default live command /home/kgaer/robotics/libero_smolvla_eval/run_libero_smolvla_live.sh now uses the reliable Matplotlib live window path.
- That live command uses MUJOCO_GL=egl and shows offscreen rollout frames in a desktop window while also saving MP4 output.
- Local libSM/libICE runtime libraries are loaded from /home/kgaer/robotics/libero_smolvla_eval/local_libs for GUI compatibility.
- The robosuite/OpenCV simulator window path is experimental on this machine and can hang on first render.
- If you want to try that experimental path manually:
  export MUJOCO_GL=glx
  python /home/kgaer/robotics/libero_smolvla_eval/live_libero_smolvla.py --viewer-backend robosuite

Verification result:
- One rollout on libero_spatial task id 0 finished successfully.
- Reported success rate: 100.0 percent for 1 episode.
