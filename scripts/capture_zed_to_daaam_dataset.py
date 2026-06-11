#!/usr/bin/env python3
"""Capture a ZED/ZED Mini sequence directly to DAAAM ImageSequenceDataset format.

No ROS is used. The output directory is directly consumable by:

    python scripts/run_pipeline.py <output_dir> \
      --dataset-type ImageSequenceDataset \
      --depth-scale 1.0 \
      --fps <capture_fps>

Output layout:

    output_dir/
      rgb/000000.png
      depth/000000.npy          # float32 depth in meters
      pose/poses.txt            # one row-major 4x4 world_T_left_camera matrix per line
      pose/poses_7d.txt         # debug/helper: x y z qx qy qz qw
      camera_info.json          # intrinsics for the saved RGB/depth resolution
      timestamps.txt            # saved_idx, camera_timestamp_sec, tracking_state, confidence
      manifest.json             # capture settings and summary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


_STOP_REQUESTED = False


def _handle_signal(signum, frame) -> None:  # noqa: ANN001, ARG001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def parse_args() -> argparse.Namespace:
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
        help="ZED coordinate system enum. IMAGE is x-right, y-down, z-forward and matches pinhole depth projection.",
    )
    parser.add_argument("--svo-file", type=Path, default=None, help="Optional SVO input instead of live camera.")

    parser.add_argument("--width", type=int, default=640, help="Saved image width. Use <=0 with --resize-mode none for native.")
    parser.add_argument("--height", type=int, default=480, help="Saved image height. Use <=0 with --resize-mode none for native.")
    parser.add_argument(
        "--resize-mode",
        choices=["crop_resize", "resize", "none"],
        default="crop_resize",
        help="How to map native ZED images into saved resolution. crop_resize preserves aspect ratio by center-cropping first.",
    )

    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many saved frames. 0 means no frame limit.")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="Stop after this many seconds. 0 means no time limit.")
    parser.add_argument("--stride", type=int, default=1, help="Save every N-th successfully grabbed frame after warmup.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Grab and track this many frames before saving.")

    parser.add_argument("--min-depth-m", type=float, default=0.05, help="Depth below this value is written as 0.")
    parser.add_argument("--max-depth-m", type=float, default=20.0, help="Depth above this value is written as 0.")
    parser.add_argument("--keep-invalid-depth", action="store_true", help="Keep NaN/Inf/out-of-range depth instead of zeroing it.")

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
    parser.add_argument("--png-compression", type=int, default=3, help="PNG compression for RGB images, 0-9.")
    return parser.parse_args()


def enum_value(enum_cls: Any, name: str) -> Any:
    """Fetch an enum value by name, with a helpful error message."""
    if hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    valid = [k for k in dir(enum_cls) if k.isupper() and not k.startswith("_")]
    raise ValueError(f"Invalid enum value {name!r} for {enum_cls}. Valid examples: {valid[:20]}")


def ensure_output_dirs(output_dir: Path, overwrite: bool) -> Dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory exists and is not empty: {output_dir}. Use --overwrite.")
    dirs = {
        "root": output_dir,
        "rgb": output_dir / "rgb",
        "depth": output_dir / "depth",
        "pose": output_dir / "pose",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def enum_name(value: Any) -> str:
    text = str(value)
    # Works for strings like 'POSITIONAL_TRACKING_STATE.OK' and 'OK'.
    return text.split(".")[-1].split("::")[-1]


def vector_to_numpy(obj: Any, attrs: Iterable[str], size: int) -> np.ndarray:
    """Convert several pyzed vector-like objects to numpy."""
    if hasattr(obj, "get"):
        arr = np.asarray(obj.get(), dtype=np.float64).reshape(-1)
        if arr.size >= size:
            return arr[:size]
    values = []
    ok = True
    for attr in attrs:
        if hasattr(obj, attr):
            values.append(float(getattr(obj, attr)))
        else:
            ok = False
            break
    if ok and len(values) == size:
        return np.asarray(values, dtype=np.float64)
    try:
        arr = np.asarray(obj, dtype=np.float64).reshape(-1)
        if arr.size >= size:
            return arr[:size]
    except Exception:
        pass
    raise TypeError(f"Cannot convert object to vector of size {size}: {type(obj)}")


def get_pose_translation_orientation(zed_pose: Any, sl: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return translation [x,y,z] and quaternion [qx,qy,qz,qw] from a pyzed Pose."""
    try:
        translation_obj = zed_pose.get_translation()
    except TypeError:
        tmp = sl.Translation()
        translation_obj = zed_pose.get_translation(tmp)
    t = vector_to_numpy(translation_obj, ["tx", "ty", "tz"], 3)

    try:
        orientation_obj = zed_pose.get_orientation()
    except TypeError:
        tmp = sl.Orientation()
        orientation_obj = zed_pose.get_orientation(tmp)
    q = vector_to_numpy(orientation_obj, ["ox", "oy", "oz", "ow"], 4)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return t, q


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    x, y, z, w = q.astype(np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_matrix(quaternion)
    T[:3, 3] = translation
    return T


def get_timestamp_sec(zed: Any, sl: Any, zed_pose: Any) -> float:
    """Best-effort ZED image timestamp in seconds."""
    candidates = []
    try:
        candidates.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE))
    except Exception:
        pass
    if hasattr(zed_pose, "timestamp"):
        candidates.append(zed_pose.timestamp)
    for ts in candidates:
        for method in ("get_nanoseconds", "get_microseconds", "get_milliseconds", "get_seconds"):
            if hasattr(ts, method):
                value = float(getattr(ts, method)())
                if method == "get_nanoseconds":
                    return value * 1e-9
                if method == "get_microseconds":
                    return value * 1e-6
                if method == "get_milliseconds":
                    return value * 1e-3
                return value
        for attr, scale in (("data_ns", 1e-9), ("data_us", 1e-6), ("data_ms", 1e-3), ("data_s", 1.0)):
            if hasattr(ts, attr):
                return float(getattr(ts, attr)) * scale
    return time.time()


def get_camera_intrinsics(zed: Any) -> Tuple[int, int, np.ndarray, list[float], Dict[str, Any]]:
    info = zed.get_camera_information()
    cfg = info.camera_configuration
    res = cfg.resolution
    native_w, native_h = int(res.width), int(res.height)
    calib = cfg.calibration_parameters
    left = calib.left_cam
    fx = float(left.fx)
    fy = float(left.fy)
    cx = float(left.cx)
    cy = float(left.cy)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = []
    if hasattr(left, "disto"):
        try:
            distortion = [float(x) for x in left.disto]
        except Exception:
            distortion = []
    extra = {
        "serial_number": getattr(info, "serial_number", None),
        "camera_model": str(getattr(info, "camera_model", "")),
        "firmware_version": str(getattr(info, "camera_firmware_version", "")),
    }
    return native_w, native_h, K, distortion, extra


def compute_output_geometry(
    native_w: int,
    native_h: int,
    K_native: np.ndarray,
    out_w: int,
    out_h: int,
    mode: str,
) -> Tuple[int, int, Tuple[int, int, int, int], np.ndarray]:
    """Return output size, crop rectangle x0/y0/w/h, and adjusted K."""
    if mode == "none" or out_w <= 0 or out_h <= 0:
        return native_w, native_h, (0, 0, native_w, native_h), K_native.copy()

    if mode == "crop_resize":
        src_aspect = native_w / native_h
        dst_aspect = out_w / out_h
        if src_aspect > dst_aspect:
            crop_h = native_h
            crop_w = int(round(native_h * dst_aspect))
            x0 = int(round((native_w - crop_w) / 2))
            y0 = 0
        else:
            crop_w = native_w
            crop_h = int(round(native_w / dst_aspect))
            x0 = 0
            y0 = int(round((native_h - crop_h) / 2))
    elif mode == "resize":
        x0, y0, crop_w, crop_h = 0, 0, native_w, native_h
    else:
        raise ValueError(f"Unknown resize mode: {mode}")

    sx = out_w / crop_w
    sy = out_h / crop_h
    K = K_native.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - x0) * sx
    K[1, 2] = (K[1, 2] - y0) * sy
    return out_w, out_h, (x0, y0, crop_w, crop_h), K


def apply_crop_resize(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    out_w: int,
    out_h: int,
    crop: Tuple[int, int, int, int],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return bgr, depth_m
    x0, y0, crop_w, crop_h = crop
    bgr_crop = bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    depth_crop = depth_m[y0 : y0 + crop_h, x0 : x0 + crop_w]
    bgr_out = cv2.resize(bgr_crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    depth_out = cv2.resize(depth_crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return bgr_out, depth_out


def sanitize_depth(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    out = depth_m.astype(np.float32, copy=True)
    invalid = ~np.isfinite(out)
    if math.isfinite(min_depth):
        invalid |= out < min_depth
    if math.isfinite(max_depth):
        invalid |= out > max_depth
    out[invalid] = 0.0
    return out


def write_camera_info(path: Path, width: int, height: int, K: np.ndarray, distortion: list[float], extra: Dict[str, Any]) -> None:
    camera_info = {
        "width": int(width),
        "height": int(height),
        "intrinsics": K.tolist(),
        "distortion": distortion,
        "notes": {
            "depth_unit": "meter",
            "pose_frame": "world_T_left_camera",
            "image_frame": "ZED rectified left image / IMAGE coordinate system",
            "daaam_depth_scale": 1.0,
        },
        "zed": extra,
    }
    path.write_text(json.dumps(camera_info, indent=2, ensure_ascii=False), encoding="utf-8")


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()

    global cv2  # noqa: PLW0603
    cv2_import_error: Optional[ImportError] = None
    pyzed_import_error: Optional[ImportError] = None

    try:
        import cv2 as cv2_module  # type: ignore

        cv2 = cv2_module
    except ImportError as exc:
        cv2_import_error = exc

    try:
        import pyzed.sl as sl  # type: ignore
    except ImportError as exc:
        pyzed_import_error = exc

    if cv2_import_error is not None:
        print("ERROR: OpenCV is not installed for this Python environment.", file=sys.stderr)
        print("Install it with: python3 -m pip install opencv-python", file=sys.stderr)

    if pyzed_import_error is not None:
        print("ERROR: pyzed.sl is not installed. Install the ZED SDK Python API first.", file=sys.stderr)
        if platform.system() == "Darwin":
            print(
                "Detected macOS. The full ZED SDK Python workflow used here requires Stereolabs ZED SDK support; "
                "USB visibility alone is not enough for depth, tracking, or pose capture.",
                file=sys.stderr,
            )
            print(
                "For full RGB/depth/pose capture, run this script on a supported ZED SDK host "
                "(Windows, Ubuntu, or Jetson) with pyzed installed.",
                file=sys.stderr,
            )
        else:
            print("On Linux this is usually: cd /usr/local/zed && python3 get_python_api.py", file=sys.stderr)

    if cv2_import_error is not None or pyzed_import_error is not None:
        return 2

    dirs = ensure_output_dirs(args.output_dir, args.overwrite)
    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "args": vars(args).copy(),
        "format": "DAAAM ImageSequenceDataset",
        "depth_scale_for_daaam": 1.0,
    }
    manifest["args"]["output_dir"] = str(args.output_dir)
    manifest["args"]["svo_file"] = str(args.svo_file) if args.svo_file else None

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = enum_value(sl.RESOLUTION, args.resolution)
    init.camera_fps = int(args.fps)
    init.depth_mode = enum_value(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = enum_value(sl.COORDINATE_SYSTEM, args.coordinate_system)
    init.depth_minimum_distance = float(args.min_depth_m)
    init.depth_maximum_distance = float(args.max_depth_m)

    if args.svo_file is not None:
        if not args.svo_file.exists():
            raise FileNotFoundError(args.svo_file)
        init.set_from_svo_file(str(args.svo_file))

    print(f"Opening ZED: resolution={args.resolution}, fps={args.fps}, depth={args.depth_mode}, coord={args.coordinate_system}")
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: Failed to open ZED camera: {err}", file=sys.stderr)
        return 2

    tracking_params = sl.PositionalTrackingParameters()
    if hasattr(tracking_params, "enable_area_memory"):
        tracking_params.enable_area_memory = bool(args.area_memory)

    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        zed.close()
        print(f"ERROR: Failed to enable ZED positional tracking: {err}", file=sys.stderr)
        return 3

    native_w, native_h, K_native, distortion, zed_extra = get_camera_intrinsics(zed)
    out_w, out_h, crop, K_out = compute_output_geometry(
        native_w, native_h, K_native, args.width, args.height, args.resize_mode
    )
    write_camera_info(dirs["root"] / "camera_info.json", out_w, out_h, K_out, distortion, zed_extra)

    manifest.update(
        {
            "native_width": native_w,
            "native_height": native_h,
            "saved_width": out_w,
            "saved_height": out_h,
            "crop_xywh": list(crop),
            "intrinsics_saved": K_out.tolist(),
            "zed": zed_extra,
        }
    )

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    zed_pose = sl.Pose()
    runtime = sl.RuntimeParameters()

    poses_txt = (dirs["pose"] / "poses.txt").open("w", encoding="utf-8")
    poses_7d_txt = (dirs["pose"] / "poses_7d.txt").open("w", encoding="utf-8")
    timestamps_txt = (dirs["root"] / "timestamps.txt").open("w", encoding="utf-8")
    timestamps_txt.write("# saved_idx camera_timestamp_sec tracking_state pose_confidence\n")

    saved = 0
    grabbed = 0
    skipped_tracking = 0
    skipped_stride = 0
    start = time.time()
    print(
        "Recording. Stop with Ctrl+C. "
        f"native={native_w}x{native_h}, saved={out_w}x{out_h}, crop={crop}, output={dirs['root']}"
    )

    try:
        while not _STOP_REQUESTED:
            if args.max_frames > 0 and saved >= args.max_frames:
                break
            if args.duration_sec > 0 and (time.time() - start) >= args.duration_sec:
                break

            err = zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                # SVO reaches END_OF_SVOFILE; live cameras may transiently fail.
                if enum_name(err) in {"END_OF_SVOFILE", "END_OF_FILE"}:
                    print("Reached end of SVO file.")
                    break
                time.sleep(0.002)
                continue

            grabbed += 1
            if grabbed <= args.warmup_frames:
                # Still call get_position during warmup to initialize tracking state.
                zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
                continue

            if args.stride > 1 and ((grabbed - args.warmup_frames - 1) % args.stride != 0):
                skipped_stride += 1
                continue

            tracking_state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            tracking_state_name = enum_name(tracking_state)
            pose_confidence = getattr(zed_pose, "pose_confidence", None)
            if args.drop_bad_tracking and tracking_state_name != "OK":
                skipped_tracking += 1
                if skipped_tracking % 30 == 1:
                    print(f"Skipping frame: tracking_state={tracking_state_name}, confidence={pose_confidence}")
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

            bgra = image_mat.get_data()
            depth_m = depth_mat.get_data()
            if bgra is None or depth_m is None:
                continue
            bgr = cv2.cvtColor(np.asarray(bgra), cv2.COLOR_BGRA2BGR)
            depth_m = np.asarray(depth_m)
            if depth_m.ndim == 3:
                depth_m = depth_m[..., 0]
            depth_m = depth_m.astype(np.float32, copy=False)

            bgr_out, depth_out = apply_crop_resize(bgr, depth_m, out_w, out_h, crop, args.resize_mode)
            if not args.keep_invalid_depth:
                depth_out = sanitize_depth(depth_out, args.min_depth_m, args.max_depth_m)
            else:
                depth_out = depth_out.astype(np.float32, copy=False)

            translation, quat = get_pose_translation_orientation(zed_pose, sl)
            T = pose_to_matrix(translation, quat)
            stamp_sec = get_timestamp_sec(zed, sl, zed_pose)

            name = f"{saved:06d}"
            rgb_path = dirs["rgb"] / f"{name}.png"
            depth_path = dirs["depth"] / f"{name}.npy"
            ok = cv2.imwrite(str(rgb_path), bgr_out, [cv2.IMWRITE_PNG_COMPRESSION, int(args.png_compression)])
            if not ok:
                raise RuntimeError(f"Failed to write RGB image: {rgb_path}")
            np.save(depth_path, depth_out.astype(np.float32))

            poses_txt.write(" ".join(f"{v:.10g}" for v in T.reshape(-1)) + "\n")
            poses_7d_txt.write(" ".join(f"{v:.10g}" for v in np.r_[translation, quat]) + "\n")
            timestamps_txt.write(f"{saved} {stamp_sec:.9f} {tracking_state_name} {pose_confidence}\n")

            saved += 1
            if saved % 30 == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(f"saved={saved} grabbed={grabbed} avg_save_fps={saved / elapsed:.2f} tracking={tracking_state_name}")

            if args.show_preview:
                preview = bgr_out.copy()
                cv2.putText(preview, f"{name} {tracking_state_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("ZED DAAAM offline capture", preview)
                if cv2.waitKey(1) in (27, ord("q")):
                    break
    finally:
        poses_txt.close()
        poses_7d_txt.close()
        timestamps_txt.close()
        if args.show_preview:
            cv2.destroyAllWindows()
        zed.disable_positional_tracking()
        zed.close()

    manifest.update(
        {
            "finished_at": datetime.now().isoformat(),
            "frames_grabbed_after_open": grabbed,
            "frames_saved": saved,
            "frames_skipped_bad_tracking": skipped_tracking,
            "frames_skipped_stride": skipped_stride,
            "elapsed_wall_sec": time.time() - start,
            "run_pipeline_hint": (
                f"python scripts/run_pipeline.py {dirs['root']} --dataset-type ImageSequenceDataset "
                f"--depth-scale 1.0 --fps {args.fps} --target-fps {args.fps}"
            ),
        }
    )
    write_manifest(dirs["root"] / "manifest.json", manifest)
    print(f"Done. Saved {saved} frames to {dirs['root']}")
    print(f"DAAAM command hint: {manifest['run_pipeline_hint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
