---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
---

# kgaero/libero-spatial-task0-test1

Generated from LIBERO / robosuite teleoperation HDF5 demos.

## Summary

- Task: `saptial task 0`
- Robot: `Panda`
- Episodes: `2`
- Total frames: `4826`
- FPS: `10`
- Action shapes: `[[7]]`
- Image keys: `[]`
- Low-dimensional keys: `['states']`

## Files

- `data/`: native LeRobot frame parquet files
- `meta/`: native LeRobot metadata, episode metadata, stats, and task tables
- `images/` and `videos/`: native LeRobot visual assets when image observations are present
