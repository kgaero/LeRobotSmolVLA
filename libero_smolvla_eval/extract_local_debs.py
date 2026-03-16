#!/usr/bin/env python3
from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import zstandard as zstd


def extract_data_tar_zst(deb_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ar_out = subprocess.check_output(["ar", "p", str(deb_path), "data.tar.zst"])
    dctx = zstd.ZstdDecompressor()
    data = dctx.decompress(ar_out)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
        tf.extractall(out_dir)


def main() -> None:
    project_dir = Path.home() / "robotics" / "libero_smolvla_eval"
    deb_dir = project_dir / "local_debs"
    out_dir = project_dir / "local_libs"
    for deb_name in ("libice6_2-1.0.10-1build3_amd64.deb", "libsm6_2-1.2.3-1build3_amd64.deb"):
        extract_data_tar_zst(deb_dir / deb_name, out_dir)
    for path in sorted(out_dir.rglob("libICE.so*")):
        print(path)
    for path in sorted(out_dir.rglob("libSM.so*")):
        print(path)


if __name__ == "__main__":
    main()
