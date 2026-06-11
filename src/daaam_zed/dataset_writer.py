from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import CaptureSettings, FrameQuality, Pose7D
from .quality import build_quality_report


class DatasetWriter:
    def __init__(
        self,
        output_dir: Path,
        settings: CaptureSettings,
        camera_info: dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.settings = settings
        self.camera_info = camera_info
        self.frame_id = 0
        self.manifest_entries = 0
        self.qualities: list[FrameQuality] = []
        self.created_at = datetime.now().isoformat()

        if self.output_dir.exists():
            if overwrite:
                shutil.rmtree(self.output_dir)
            elif any(self.output_dir.iterdir()):
                raise FileExistsError(f"Output directory exists and is not empty: {self.output_dir}")

        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth"
        self.pose_dir = self.output_dir / "pose"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.pose_dir.mkdir(parents=True, exist_ok=True)

        self.pose_file = (self.pose_dir / "poses.txt").open("w", encoding="utf-8")
        self.pose_7d_file = (self.pose_dir / "poses_7d.txt").open("w", encoding="utf-8")
        self.timestamps_file = (self.output_dir / "timestamps.txt").open("w", encoding="utf-8")
        self.manifest_jsonl_file = (self.output_dir / "manifest.jsonl").open("w", encoding="utf-8")
        self.timestamps_file.write("# frame_id timestamp_sec tracking_state pose_confidence\n")

    def write_camera_info(self, measured_saved_fps: float | None = None, timestamp_source: str = "zed_image") -> None:
        data = dict(self.camera_info)
        data.setdefault("notes", {})
        data["notes"].update(
            {
                "depth_unit": "meter",
                "pose_frame": "world_T_left_camera",
                "image_frame": "ZED rectified left image / IMAGE coordinate system",
                "depth_frame": "ZED depth aligned to left image",
                "camera_optical_frame": "x right, y down, z forward",
                "daaam_depth_scale": 1.0,
            }
        )
        data["timestamp"] = {"source": timestamp_source, "unit": "second"}
        data["capture"] = {
            "configured_fps": int(self.settings.fps),
            "measured_saved_fps": measured_saved_fps,
            "resolution": self.settings.resolution,
            "depth_mode": self.settings.depth_mode,
            "resize_mode": self.settings.resize_mode,
            "output_width": self.settings.output_width,
            "output_height": self.settings.output_height,
        }
        _write_json(self.output_dir / "camera_info.json", data)

    def write_frame(
        self,
        bgr: np.ndarray,
        depth_m: np.ndarray,
        world_T_left_camera: np.ndarray,
        pose_7d: Pose7D,
        timestamp: float,
        timestamp_source: str,
        zed_tracking_state: str,
        pose_confidence: float | None,
        quality: FrameQuality,
        sequence_id: int,
        exposure: float | None = None,
        gain: float | None = None,
        white_balance: float | None = None,
    ) -> int:
        frame_id = self.frame_id
        name = f"{frame_id:06d}"
        rgb_path = self.rgb_dir / f"{name}.png"
        depth_path = self.depth_dir / f"{name}.npy"
        _atomic_write_png(rgb_path, bgr, self.settings.png_compression)
        _atomic_write_npy(depth_path, depth_m.astype(np.float32, copy=False))

        matrix = np.asarray(world_T_left_camera, dtype=np.float64).reshape(4, 4)
        self.pose_file.write(" ".join(f"{v:.10g}" for v in matrix.reshape(-1)) + "\n")
        self.pose_file.flush()
        self.pose_7d_file.write(" ".join(f"{v:.10g}" for v in pose_7d.as_list()) + "\n")
        self.pose_7d_file.flush()
        self.timestamps_file.write(f"{frame_id} {timestamp:.9f} {zed_tracking_state} {pose_confidence}\n")
        self.timestamps_file.flush()

        entry = _frame_entry(
            frame_id=frame_id,
            sequence_id=sequence_id,
            timestamp=timestamp,
            timestamp_source=timestamp_source,
            rgb_path=rgb_path.relative_to(self.output_dir),
            depth_path=depth_path.relative_to(self.output_dir),
            pose_7d=pose_7d,
            world_T_left_camera=matrix,
            zed_tracking_state=zed_tracking_state,
            pose_confidence=pose_confidence,
            quality=quality,
            exposure=exposure,
            gain=gain,
            white_balance=white_balance,
        )
        self._write_manifest_entry(entry)
        self.qualities.append(quality)
        self.frame_id += 1
        return frame_id

    def write_skipped_frame(
        self,
        grabbed_index: int,
        timestamp: float,
        skip_reason: str,
        zed_tracking_state: str,
        pose_confidence: float | None = None,
        quality: FrameQuality | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "frame_id": None,
            "grabbed_index": int(grabbed_index),
            "timestamp": float(timestamp),
            "skip_reason": skip_reason,
            "zed_tracking_state": zed_tracking_state,
            "pose_confidence": pose_confidence,
        }
        if quality is not None:
            entry.update(_quality_dict(quality))
        self._write_manifest_entry(entry)

    def close(self, frames_grabbed: int, extra: dict[str, Any] | None = None) -> None:
        if self.pose_file.closed:
            return
        self.pose_file.close()
        self.pose_7d_file.close()
        self.timestamps_file.close()
        self.manifest_jsonl_file.close()
        self.write_camera_info()

        manifest = {
            "created_at": self.created_at,
            "finished_at": datetime.now().isoformat(),
            "format": "DAAAM ImageSequenceDataset",
            "depth_scale_for_daaam": 1.0,
            "frames_grabbed_after_open": int(frames_grabbed),
            "frames_saved": int(self.frame_id),
            "manifest_entries": int(self.manifest_entries),
            "settings": self.settings.__dict__,
        }
        if extra:
            manifest.update(extra)
        _write_json(self.output_dir / "manifest.json", manifest)

        from .validation import validate_dataset

        validation = validate_dataset(self.output_dir)
        report = build_quality_report(
            frame_count=validation.rgb_count,
            timestamp_count=validation.timestamp_count,
            pose_count=validation.pose_count,
            manifest_count=validation.manifest_entries,
            qualities=self.qualities,
            validation_errors=validation.errors,
        )
        _write_json(self.output_dir / "quality_report.json", report)

    def _write_manifest_entry(self, entry: dict[str, Any]) -> None:
        self.manifest_jsonl_file.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")
        self.manifest_jsonl_file.flush()
        self.manifest_entries += 1


def _frame_entry(
    frame_id: int,
    sequence_id: int,
    timestamp: float,
    timestamp_source: str,
    rgb_path: Path,
    depth_path: Path,
    pose_7d: Pose7D,
    world_T_left_camera: np.ndarray,
    zed_tracking_state: str,
    pose_confidence: float | None,
    quality: FrameQuality,
    exposure: float | None,
    gain: float | None,
    white_balance: float | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "frame_id": int(frame_id),
        "sequence_id": int(sequence_id),
        "timestamp": float(timestamp),
        "timestamp_source": timestamp_source,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "pose_valid": True,
        "pose_path": "pose/poses.txt",
        "pose_7d": pose_7d.as_list(),
        "world_T_left_camera": world_T_left_camera.tolist(),
        "zed_tracking_state": zed_tracking_state,
        "pose_confidence": pose_confidence,
        "exposure": exposure,
        "gain": gain,
        "white_balance": white_balance,
    }
    entry.update(_quality_dict(quality))
    return entry


def _quality_dict(quality: FrameQuality) -> dict[str, Any]:
    return {
        "pose_reset": quality.pose_reset,
        "pose_delta_translation_m": quality.pose_delta_translation_m,
        "pose_delta_rotation_deg": quality.pose_delta_rotation_deg,
        "depth_valid_ratio": quality.depth_valid_ratio,
        "depth_min_m": quality.depth_min_m,
        "depth_median_m": quality.depth_median_m,
        "depth_p95_m": quality.depth_p95_m,
        "depth_max_m": quality.depth_max_m,
        "depth_nan_ratio": quality.depth_nan_ratio,
        "depth_zero_ratio": quality.depth_zero_ratio,
        "rgb_mean_brightness": quality.rgb_mean_brightness,
        "rgb_dark_ratio": quality.rgb_dark_ratio,
        "rgb_saturation_ratio": quality.rgb_saturation_ratio,
        "rgb_laplacian_sharpness": quality.rgb_laplacian_sharpness,
        "warnings": list(quality.warnings),
    }


def _atomic_write_png(path: Path, image: np.ndarray, compression: int) -> None:
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    ok = cv2.imwrite(str(tmp_path), image, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise RuntimeError(f"Failed to write RGB image: {path}")
    tmp_path.replace(path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
    tmp_path.replace(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
