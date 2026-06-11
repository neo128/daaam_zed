from __future__ import annotations

import math
from statistics import median

import cv2
import numpy as np

from .models import FrameQuality, QualityThresholds


def compute_depth_quality(depth_m: np.ndarray) -> FrameQuality:
    depth = np.asarray(depth_m, dtype=np.float32)
    total = int(depth.size)
    if total == 0:
        return FrameQuality()

    finite = np.isfinite(depth)
    positive = finite & (depth > 0)
    valid = depth[positive]
    stats = FrameQuality(
        depth_valid_ratio=float(valid.size / total),
        depth_nan_ratio=float((~finite).sum() / total),
        depth_zero_ratio=float((finite & (depth == 0)).sum() / total),
        provided_fields={"depth_valid_ratio", "depth_nan_ratio", "depth_zero_ratio"},
    )
    if valid.size:
        stats.depth_min_m = float(np.min(valid))
        stats.depth_median_m = float(np.median(valid))
        stats.depth_p95_m = float(np.percentile(valid, 95))
        stats.depth_max_m = float(np.max(valid))
        stats.provided_fields.update({"depth_min_m", "depth_median_m", "depth_p95_m", "depth_max_m"})
    return stats


def compute_rgb_quality(bgr: np.ndarray, dark_threshold: int = 30, saturation_threshold: int = 240) -> FrameQuality:
    img = np.asarray(bgr)
    if img.size == 0:
        return FrameQuality()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img.ndim == 3 else None
    total = int(gray.size)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    saturation_ratio = 0.0
    if hsv is not None:
        saturation_ratio = float((hsv[..., 1] >= saturation_threshold).sum() / total)
    return FrameQuality(
        rgb_mean_brightness=float(np.mean(gray)),
        rgb_dark_ratio=float((gray < dark_threshold).sum() / total),
        rgb_saturation_ratio=saturation_ratio,
        rgb_laplacian_sharpness=float(np.var(laplacian)),
        provided_fields={
            "rgb_mean_brightness",
            "rgb_dark_ratio",
            "rgb_saturation_ratio",
            "rgb_laplacian_sharpness",
        },
    )


def merge_quality(*qualities: FrameQuality) -> FrameQuality:
    merged = FrameQuality()
    for quality in qualities:
        provided = quality.provided_fields or {
            field_name for field_name, value in quality.__dict__.items() if field_name not in {"warnings", "provided_fields"} and value is not None
        }
        for field_name in provided:
            value = getattr(quality, field_name)
            if field_name == "provided_fields":
                continue
            setattr(merged, field_name, value)
            merged.provided_fields.add(field_name)
        if quality.warnings:
            merged.warnings.extend(quality.warnings)
            merged.provided_fields.add("warnings")
        if quality.pose_reset:
            merged.pose_reset = True
            merged.provided_fields.add("pose_reset")
    return merged


def classify_pose_delta(
    previous_world_T_camera: np.ndarray,
    current_world_T_camera: np.ndarray,
    thresholds: QualityThresholds | None = None,
) -> FrameQuality:
    thresholds = thresholds or QualityThresholds()
    previous = np.asarray(previous_world_T_camera, dtype=np.float64).reshape(4, 4)
    current = np.asarray(current_world_T_camera, dtype=np.float64).reshape(4, 4)
    delta_translation = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    delta_rotation = _rotation_delta_deg(previous[:3, :3], current[:3, :3])
    warnings: list[str] = []
    pose_reset = False
    if delta_translation > thresholds.reset_translation_delta_m:
        pose_reset = True
        warnings.append("POSE_RESET")
    elif delta_translation > thresholds.warning_translation_delta_m:
        warnings.append("POSE_DELTA_WARNING")
    return FrameQuality(
        pose_delta_translation_m=delta_translation,
        pose_delta_rotation_deg=delta_rotation,
        pose_reset=pose_reset,
        warnings=warnings,
        provided_fields={"pose_delta_translation_m", "pose_delta_rotation_deg", "pose_reset", "warnings"},
    )


def _rotation_delta_deg(previous_R: np.ndarray, current_R: np.ndarray) -> float:
    delta = current_R @ previous_R.T
    trace = float(np.trace(delta))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(math.degrees(math.acos(cos_angle)))


def build_quality_report(
    frame_count: int,
    timestamp_count: int,
    pose_count: int,
    manifest_count: int,
    qualities: list[FrameQuality],
    validation_errors: list[str],
) -> dict:
    depth_ratios = [q.depth_valid_ratio for q in qualities]
    brightness = [q.rgb_mean_brightness for q in qualities]
    translation_deltas = [q.pose_delta_translation_m for q in qualities if q.pose_delta_translation_m is not None]
    reset_count = sum(1 for q in qualities if q.pose_reset)
    warning_count = sum(len(q.warnings) for q in qualities)
    return {
        "ok": not validation_errors,
        "counts": {
            "frames": int(frame_count),
            "timestamps": int(timestamp_count),
            "poses": int(pose_count),
            "manifest_entries": int(manifest_count),
        },
        "pose": {
            "reset_count": int(reset_count),
            "warning_count": int(warning_count),
            "translation_delta_min_m": _safe_min(translation_deltas),
            "translation_delta_median_m": _safe_median(translation_deltas),
            "translation_delta_p95_m": _safe_percentile(translation_deltas, 95),
            "translation_delta_max_m": _safe_max(translation_deltas),
        },
        "depth": {
            "valid_ratio_min": _safe_min(depth_ratios),
            "valid_ratio_median": _safe_median(depth_ratios),
            "valid_ratio_max": _safe_max(depth_ratios),
        },
        "rgb": {
            "mean_brightness_min": _safe_min(brightness),
            "mean_brightness_median": _safe_median(brightness),
            "mean_brightness_max": _safe_max(brightness),
        },
        "errors": list(validation_errors),
    }


def _safe_min(values: list[float]) -> float | None:
    return float(min(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _safe_percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else None
