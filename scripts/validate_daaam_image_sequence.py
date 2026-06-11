#!/usr/bin/env python3
"""Validate a local dataset for DAAAM ImageSequenceDataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daaam_zed.validation import validate_dataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a DAAAM ImageSequenceDataset directory.")
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("--depth-scale", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_dataset(args.dataset_dir, depth_scale=args.depth_scale, fps=args.fps)
    print(f"dataset: {result.dataset_dir}")
    print(f"rgb files:   {result.rgb_count}")
    print(f"depth files: {result.depth_count}")
    print(f"pose lines:  {result.pose_count}")
    print(f"timestamps:  {result.timestamp_count}")
    print(f"manifest entries: {result.manifest_entries}")
    if result.camera_info is not None:
        print("camera_info:", json.dumps({k: result.camera_info.get(k) for k in ["width", "height", "intrinsics"]}, indent=2))
    if result.first_depth_range_m is not None:
        mn, med, mx = result.first_depth_range_m
        print(f"first depth valid_pixels={result.first_depth_valid_pixels}")
        print(f"first depth valid range: min={mn:.3f}m max={mx:.3f}m median={med:.3f}m")
    print(f"pose resets: {result.pose_reset_count}")
    print(f"large pose jumps: {result.large_pose_jump_count}")

    if not result.ok:
        print("\nFAILED")
        for error in result.errors:
            print("ERROR:", error)
        return 1

    print("\nOK: dataset matches DAAAM ImageSequenceDataset layout.")
    fps_arg = f" --fps {args.fps:g} --target-fps {args.fps:g}" if args.fps else " --fps <capture_fps> --target-fps <capture_fps>"
    print("\nDAAAM command:")
    print(
        f"python scripts/run_pipeline.py {result.dataset_dir} "
        f"--dataset-type ImageSequenceDataset --depth-scale {args.depth_scale:g}{fps_arg} "
        "--output-dir output/zed_offline_test"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
