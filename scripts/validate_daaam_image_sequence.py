#!/usr/bin/env python3
"""Validate a local dataset for DAAAM ImageSequenceDataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a DAAAM ImageSequenceDataset directory.")
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("--depth-scale", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_dir.expanduser().resolve()
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    pose_file = root / "pose" / "poses.txt"
    camera_info_file = root / "camera_info.json"

    errors: list[str] = []
    for path, name in [
        (rgb_dir, "rgb/"),
        (depth_dir, "depth/"),
        (pose_file, "pose/poses.txt"),
        (camera_info_file, "camera_info.json"),
    ]:
        if not path.exists():
            errors.append(f"Missing {name}: {path}")

    if errors:
        for e in errors:
            print("ERROR:", e)
        return 2

    rgb_files = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix.lower() in {".npy", ".png", ".exr", ".tiff"}])
    pose_lines = [ln for ln in pose_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    print(f"dataset: {root}")
    print(f"rgb files:   {len(rgb_files)}")
    print(f"depth files: {len(depth_files)}")
    print(f"pose lines:  {len(pose_lines)}")

    if not rgb_files:
        errors.append("No RGB files found")
    if len(depth_files) != len(rgb_files):
        errors.append(f"RGB/depth count mismatch: {len(rgb_files)} vs {len(depth_files)}")
    if len(pose_lines) != len(rgb_files):
        errors.append(f"RGB/pose count mismatch: {len(rgb_files)} vs {len(pose_lines)}")

    info = json.loads(camera_info_file.read_text(encoding="utf-8"))
    print("camera_info:", json.dumps({k: info.get(k) for k in ["width", "height", "intrinsics"]}, indent=2))

    if rgb_files:
        rgb = cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR)
        if rgb is None:
            errors.append(f"Cannot read first RGB: {rgb_files[0]}")
        else:
            h, w = rgb.shape[:2]
            if int(info.get("width", -1)) != w or int(info.get("height", -1)) != h:
                errors.append(f"camera_info size {info.get('width')}x{info.get('height')} != RGB size {w}x{h}")
            print(f"first RGB shape: {rgb.shape}")

    if depth_files:
        first_depth = depth_files[0]
        if first_depth.suffix.lower() == ".npy":
            depth = np.load(first_depth)
        else:
            depth = cv2.imread(str(first_depth), cv2.IMREAD_ANYDEPTH).astype(np.float32) / args.depth_scale
        if rgb_files and 'rgb' in locals() and rgb is not None:
            if depth.shape[:2] != rgb.shape[:2]:
                errors.append(f"Depth shape {depth.shape[:2]} != RGB shape {rgb.shape[:2]}")
        finite = depth[np.isfinite(depth)]
        valid = finite[finite > 0]
        print(f"first depth shape: {depth.shape}, dtype={depth.dtype}, valid_pixels={valid.size}")
        if valid.size:
            print(f"first depth valid range: min={float(valid.min()):.3f}m max={float(valid.max()):.3f}m median={float(np.median(valid)):.3f}m")
        else:
            errors.append("First depth frame has no positive finite values")

    for i, ln in enumerate(pose_lines[:3]):
        vals = [float(x) for x in ln.split()]
        if len(vals) != 16:
            errors.append(f"Pose line {i} does not have 16 values")
            break
        T = np.array(vals).reshape(4, 4)
        if not np.allclose(T[3], np.array([0, 0, 0, 1], dtype=float), atol=1e-5):
            errors.append(f"Pose line {i} bottom row is not [0 0 0 1]")
            break

    if errors:
        print("\nFAILED")
        for e in errors:
            print("ERROR:", e)
        return 1

    print("\nOK: dataset matches DAAAM ImageSequenceDataset layout.")
    fps_arg = f" --fps {args.fps:g} --target-fps {args.fps:g}" if args.fps else " --fps <capture_fps> --target-fps <capture_fps>"
    print("\nDAAAM command:")
    print(
        f"python scripts/run_pipeline.py {root} "
        f"--dataset-type ImageSequenceDataset --depth-scale {args.depth_scale:g}{fps_arg} "
        "--output-dir output/zed_offline_test"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
