---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
---

# kgaero/libero-spatial-task0-withImages

Generated from LIBERO / robosuite teleoperation HDF5 demos.

## Summary

- Task: `spatial task 0`
- Robot: `Panda`
- Episodes: `1`
- Total frames: `1600`
- FPS: `10`
- Action shapes: `[[7]]`
- Image keys: `['image', 'wrist_image']`
- Low-dimensional keys: `[]`

## Files

- `data/`: native LeRobot frame parquet files
- `meta/`: native LeRobot metadata, episode metadata, stats, and task tables
- `images/` and `videos/`: native LeRobot visual assets when image observations are present
