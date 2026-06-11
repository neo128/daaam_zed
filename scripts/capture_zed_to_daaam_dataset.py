#!/usr/bin/env python3
"""Capture a ZED/ZED Mini sequence directly to DAAAM ImageSequenceDataset format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daaam_zed.models import CaptureSettings  # noqa: E402
from daaam_zed.zed_capture import run_capture  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture ZED camera data to DAAAM ImageSequenceDataset format without ROS."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory if it already exists.")
    parser.add_argument("--resolution", default="HD720", help="ZED SDK resolution enum name, e.g. HD720, HD1080, VGA.")
    parser.add_argument("--fps", type=int, default=15, help="Camera capture FPS and DAAAM dataset FPS.")
    parser.add_argument("--depth-mode", default="ULTRA", help="ZED SDK depth mode enum name, e.g. ULTRA, QUALITY, NEURAL.")
    parser.add_argument(
        "--coordinate-system",
        default="IMAGE",
        help="ZED coordinate system enum. IMAGE is x-right, y-down, z-forward.",
    )
    parser.add_argument("--svo-file", type=Path, default=None, help="Optional SVO input instead of live camera.")
    parser.add_argument("--width", type=int, default=640, help="Saved image width.")
    parser.add_argument("--height", type=int, default=480, help="Saved image height.")
    parser.add_argument(
        "--resize-mode",
        choices=["crop_resize", "resize", "none"],
        default="crop_resize",
        help="How to map native ZED images into saved resolution.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many saved frames. 0 means no limit.")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="Stop after this many seconds. 0 means no limit.")
    parser.add_argument("--stride", type=int, default=1, help="Save every N-th successfully grabbed frame after warmup.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Grab and track this many frames before saving.")
    parser.add_argument("--min-depth-m", type=float, default=0.05, help="Depth below this value is written as 0.")
    parser.add_argument("--max-depth-m", type=float, default=20.0, help="Depth above this value is written as 0.")
    parser.add_argument("--keep-invalid-depth", action="store_true", help="Keep NaN/Inf/out-of-range depth.")
    parser.add_argument(
        "--drop-bad-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop frames whose ZED positional tracking state is not OK.",
    )
    parser.add_argument(
        "--area-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ZED area memory for positional tracking.",
    )
    parser.add_argument("--show-preview", action="store_true", help="Show a small OpenCV preview while recording.")
    parser.add_argument("--record-svo", action="store_true", help="Also record raw ZED SVO to raw.svo in the output directory.")
    parser.add_argument("--png-compression", type=int, default=3, help="PNG compression for RGB images, 0-9.")
    return parser.parse_args(argv)


def build_settings_from_args(args: argparse.Namespace) -> CaptureSettings:
    return CaptureSettings(
        resolution=args.resolution,
        fps=int(args.fps),
        depth_mode=args.depth_mode,
        coordinate_system=args.coordinate_system,
        svo_file=args.svo_file,
        output_width=int(args.width),
        output_height=int(args.height),
        resize_mode=args.resize_mode,
        max_frames=int(args.max_frames),
        duration_sec=float(args.duration_sec),
        stride=int(args.stride),
        warmup_frames=int(args.warmup_frames),
        min_depth_m=float(args.min_depth_m),
        max_depth_m=float(args.max_depth_m),
        keep_invalid_depth=bool(args.keep_invalid_depth),
        drop_bad_tracking=bool(args.drop_bad_tracking),
        area_memory=bool(args.area_memory),
        png_compression=int(args.png_compression),
        enable_svo_recording=bool(args.record_svo),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = build_settings_from_args(args)
    try:
        summary = run_capture(
            output_dir=args.output_dir,
            settings=settings,
            overwrite=bool(args.overwrite),
            show_preview=bool(args.show_preview),
        )
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Install the ZED SDK Python API and ensure /usr/local/zed/lib is accessible.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Done. Saved {summary.frames_saved} frames to {Path(args.output_dir).expanduser().resolve()}")
    print(
        "DAAAM command hint: "
        f"python scripts/run_pipeline.py {Path(args.output_dir).expanduser().resolve()} "
        f"--dataset-type ImageSequenceDataset --depth-scale 1.0 --fps {settings.fps} --target-fps {settings.fps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
