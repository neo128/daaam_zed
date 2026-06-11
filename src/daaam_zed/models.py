from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptureSettings:
    resolution: str = "HD720"
    fps: int = 15
    depth_mode: str = "ULTRA"
    coordinate_system: str = "IMAGE"
    svo_file: Path | None = None
    output_width: int = 640
    output_height: int = 480
    resize_mode: str = "crop_resize"
    max_frames: int = 0
    duration_sec: float = 0.0
    stride: int = 1
    warmup_frames: int = 30
    min_depth_m: float = 0.05
    max_depth_m: float = 20.0
    keep_invalid_depth: bool = False
    drop_bad_tracking: bool = True
    area_memory: bool = True
    png_compression: int = 3
    enable_svo_recording: bool = False


@dataclass(frozen=True)
class QualityThresholds:
    p95_translation_delta_target_m: float = 0.05
    warning_translation_delta_m: float = 0.20
    reset_translation_delta_m: float = 0.25


@dataclass(frozen=True)
class Pose7D:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw]


@dataclass
class FrameQuality:
    depth_valid_ratio: float = 0.0
    depth_min_m: float | None = None
    depth_median_m: float | None = None
    depth_p95_m: float | None = None
    depth_max_m: float | None = None
    depth_nan_ratio: float = 0.0
    depth_zero_ratio: float = 0.0
    rgb_mean_brightness: float = 0.0
    rgb_dark_ratio: float = 0.0
    rgb_saturation_ratio: float = 0.0
    rgb_laplacian_sharpness: float = 0.0
    pose_delta_translation_m: float | None = None
    pose_delta_rotation_deg: float | None = None
    pose_reset: bool = False
    warnings: list[str] = field(default_factory=list)
    provided_fields: set[str] = field(default_factory=set, repr=False, compare=False)


@dataclass
class CameraFrame:
    left_bgr: Any
    right_bgr: Any | None
    depth_m: Any
    world_T_left_camera: Any
    pose_7d: Pose7D
    timestamp: float
    timestamp_source: str
    tracking_state: str
    pose_confidence: float | None
    exposure: float | None = None
    gain: float | None = None
    white_balance: float | None = None


@dataclass
class FrameRecord:
    frame_id: int
    sequence_id: int
    timestamp: float
    timestamp_source: str
    rgb_path: Path
    depth_path: Path
    pose_valid: bool
    pose_7d: Pose7D
    world_T_left_camera: list[list[float]]
    zed_tracking_state: str
    pose_confidence: float | None
    quality: FrameQuality
    exposure: float | None = None
    gain: float | None = None
    white_balance: float | None = None


@dataclass
class SkippedFrameRecord:
    grabbed_index: int
    timestamp: float
    skip_reason: str
    zed_tracking_state: str
    pose_confidence: float | None = None
    quality: FrameQuality | None = None


@dataclass
class CaptureSummary:
    output_dir: Path
    frames_saved: int = 0
    frames_grabbed: int = 0
    frames_skipped_bad_tracking: int = 0
    frames_skipped_stride: int = 0
    pose_reset_count: int = 0
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
