from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ValidationResult:
    dataset_dir: Path
    ok: bool
    errors: list[str] = field(default_factory=list)
    rgb_count: int = 0
    depth_count: int = 0
    pose_count: int = 0
    timestamp_count: int = 0
    manifest_entries: int = 0
    pose_reset_count: int = 0
    large_pose_jump_count: int = 0
    first_depth_valid_pixels: int = 0
    first_depth_range_m: tuple[float, float, float] | None = None
    camera_info: dict | None = None


def validate_dataset(dataset_dir: Path, depth_scale: float = 1.0, fps: float | None = None) -> ValidationResult:
    root = dataset_dir.expanduser().resolve()
    errors: list[str] = []
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    pose_file = root / "pose" / "poses.txt"
    pose_7d_file = root / "pose" / "poses_7d.txt"
    timestamp_file = root / "timestamps.txt"
    camera_info_file = root / "camera_info.json"
    manifest_jsonl_file = root / "manifest.jsonl"

    for path, name in [
        (rgb_dir, "rgb/"),
        (depth_dir, "depth/"),
        (pose_file, "pose/poses.txt"),
        (timestamp_file, "timestamps.txt"),
        (camera_info_file, "camera_info.json"),
    ]:
        if not path.exists():
            errors.append(f"Missing {name}: {path}")

    rgb_files = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
    depth_files = sorted(depth_dir.glob("*.npy")) if depth_dir.exists() else []
    pose_lines = _data_lines(pose_file)
    pose_7d_lines = _data_lines(pose_7d_file)
    timestamp_rows = _timestamp_rows(timestamp_file)
    manifest_entries = _manifest_entries(manifest_jsonl_file)

    if len(depth_files) != len(rgb_files):
        errors.append(f"RGB/depth count mismatch: {len(rgb_files)} vs {len(depth_files)}")
    if len(pose_lines) != len(rgb_files):
        errors.append(f"RGB/pose count mismatch: {len(rgb_files)} vs {len(pose_lines)}")
    if len(timestamp_rows) != len(rgb_files):
        errors.append(f"RGB/timestamp count mismatch: {len(rgb_files)} vs {len(timestamp_rows)}")
    saved_manifest_entries = [entry for entry in manifest_entries if entry.get("frame_id") is not None]
    if saved_manifest_entries and len(saved_manifest_entries) != len(rgb_files):
        errors.append(f"RGB/manifest saved-frame count mismatch: {len(rgb_files)} vs {len(saved_manifest_entries)}")

    info = _load_json(camera_info_file)
    first_depth_valid_pixels = 0
    first_depth_range: tuple[float, float, float] | None = None
    if rgb_files and depth_files and info:
        rgb = cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR)
        if rgb is None:
            errors.append(f"Cannot read first RGB: {rgb_files[0]}")
        else:
            h, w = rgb.shape[:2]
            if int(info.get("width", -1)) != w or int(info.get("height", -1)) != h:
                errors.append(f"camera_info size {info.get('width')}x{info.get('height')} != RGB size {w}x{h}")
        depth = np.load(depth_files[0])
        if rgb is not None and depth.shape[:2] != rgb.shape[:2]:
            errors.append(f"Depth shape {depth.shape[:2]} != RGB shape {rgb.shape[:2]}")
        finite = depth[np.isfinite(depth)]
        valid = finite[finite > 0]
        first_depth_valid_pixels = int(valid.size)
        if valid.size:
            first_depth_range = (float(valid.min()), float(np.median(valid)), float(valid.max()))
        else:
            errors.append("First depth frame has no positive finite values")

    _validate_pose_lines(pose_lines, errors)
    _validate_pose_7d_lines(pose_7d_lines, errors)
    _validate_timestamps(timestamp_rows, errors)

    pose_reset_count = sum(1 for entry in manifest_entries if entry.get("pose_reset"))
    large_pose_jump_count = sum(
        1
        for entry in manifest_entries
        if (entry.get("pose_delta_translation_m") or 0.0) > 0.20
    )
    return ValidationResult(
        dataset_dir=root,
        ok=not errors,
        errors=errors,
        rgb_count=len(rgb_files),
        depth_count=len(depth_files),
        pose_count=len(pose_lines),
        timestamp_count=len(timestamp_rows),
        manifest_entries=len(manifest_entries),
        pose_reset_count=pose_reset_count,
        large_pose_jump_count=large_pose_jump_count,
        first_depth_valid_pixels=first_depth_valid_pixels,
        first_depth_range_m=first_depth_range,
        camera_info=info,
    )


def _data_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _timestamp_rows(path: Path) -> list[tuple[int, float, str, str | None]]:
    rows = []
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 2:
            continue
        state = parts[2] if len(parts) > 2 else ""
        confidence = parts[3] if len(parts) > 3 else None
        rows.append((int(parts[0]), float(parts[1]), state, confidence))
    return rows


def _manifest_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_pose_lines(pose_lines: list[str], errors: list[str]) -> None:
    for i, line in enumerate(pose_lines):
        try:
            vals = [float(x) for x in line.split()]
        except ValueError:
            errors.append(f"Pose line {i} contains non-numeric values")
            continue
        if len(vals) != 16:
            errors.append(f"Pose line {i} does not have 16 values")
            continue
        T = np.array(vals, dtype=float).reshape(4, 4)
        if not np.allclose(T[3], np.array([0, 0, 0, 1], dtype=float), atol=1e-5):
            errors.append(f"Pose line {i} bottom row is not [0 0 0 1]")


def _validate_pose_7d_lines(pose_7d_lines: list[str], errors: list[str]) -> None:
    for i, line in enumerate(pose_7d_lines):
        vals = [float(x) for x in line.split()]
        if len(vals) != 7:
            errors.append(f"Pose 7D line {i} does not have 7 values")
            continue
        q = np.array(vals[3:7], dtype=float)
        norm = float(np.linalg.norm(q))
        if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
            errors.append(f"Pose 7D line {i} quaternion norm is {norm:.6f}, expected 1")


def _validate_timestamps(timestamp_rows: list[tuple[int, float, str, str | None]], errors: list[str]) -> None:
    previous_ts = None
    for frame_id, ts, _state, _confidence in timestamp_rows:
        if previous_ts is not None and ts <= previous_ts:
            errors.append(f"Timestamps are not strictly increasing at frame {frame_id}: {ts} <= {previous_ts}")
            return
        previous_ts = ts
