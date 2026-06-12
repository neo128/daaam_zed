from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .dataset_writer import DatasetWriter
from .geometry import apply_crop_resize, compute_output_geometry, pose_to_matrix, sanitize_depth
from .models import CameraFrame, CaptureSettings, CaptureSummary, Pose7D
from .quality import classify_pose_delta, compute_depth_quality, compute_rgb_quality, merge_quality


class CameraLike(Protocol):
    def open(self) -> dict[str, Any]:
        ...

    def grab(self) -> CameraFrame | None:
        ...

    def close(self) -> None:
        ...


def run_capture(
    output_dir: Path,
    settings: CaptureSettings,
    overwrite: bool = False,
    camera: CameraLike | None = None,
    show_preview: bool = False,
    progress_callback: Any | None = None,
) -> CaptureSummary:
    camera = camera or ZedCamera(settings)
    camera_info = camera.open()
    writer: DatasetWriter | None = None
    grabbed = 0
    skipped_tracking = 0
    skipped_stride = 0
    previous_pose: np.ndarray | None = None
    sequence_id = 0
    pose_reset_count = 0
    start = time.time()
    try:
        native_w = int(camera_info["native_width"])
        native_h = int(camera_info["native_height"])
        K_native = np.asarray(camera_info["intrinsics"], dtype=np.float64)
        out_w, out_h, crop, K_out = compute_output_geometry(
            native_w,
            native_h,
            K_native,
            settings.output_width,
            settings.output_height,
            settings.resize_mode,
        )
        writer_info = {
            "width": out_w,
            "height": out_h,
            "intrinsics": K_out.tolist(),
            "distortion": camera_info.get("distortion", []),
            "native_width": native_w,
            "native_height": native_h,
            "crop_xywh": list(crop),
            "zed": camera_info.get("zed", {}),
            "extrinsics": camera_info.get("extrinsics"),
        }
        writer = DatasetWriter(output_dir, settings, writer_info, overwrite=overwrite)
        if settings.enable_svo_recording and hasattr(camera, "enable_svo_recording"):
            camera.enable_svo_recording(writer.output_dir / "raw.svo")

        while True:
            if settings.max_frames > 0 and writer.frame_id >= settings.max_frames:
                break
            if settings.duration_sec > 0 and (time.time() - start) >= settings.duration_sec:
                break

            frame = camera.grab()
            if frame is None:
                break
            grabbed += 1

            if grabbed <= settings.warmup_frames:
                continue

            if settings.stride > 1 and ((grabbed - settings.warmup_frames - 1) % settings.stride != 0):
                skipped_stride += 1
                continue

            if settings.drop_bad_tracking and frame.tracking_state != "OK":
                skipped_tracking += 1
                writer.write_skipped_frame(
                    grabbed_index=grabbed,
                    timestamp=frame.timestamp,
                    skip_reason="BAD_TRACKING",
                    zed_tracking_state=frame.tracking_state,
                    pose_confidence=frame.pose_confidence,
                )
                continue

            bgr_out, depth_out = apply_crop_resize(
                frame.left_bgr,
                frame.depth_m,
                out_w,
                out_h,
                crop,
                settings.resize_mode,
            )
            if not settings.keep_invalid_depth:
                depth_out = sanitize_depth(depth_out, settings.min_depth_m, settings.max_depth_m)
            quality = merge_quality(compute_depth_quality(depth_out), compute_rgb_quality(bgr_out))
            pose_matrix = np.asarray(frame.world_T_left_camera, dtype=np.float64).reshape(4, 4)
            if previous_pose is not None:
                pose_quality = classify_pose_delta(previous_pose, pose_matrix)
                quality = merge_quality(quality, pose_quality)
                if pose_quality.pose_reset:
                    pose_reset_count += 1
                    sequence_id += 1
            previous_pose = pose_matrix.copy()

            writer.write_frame(
                bgr=bgr_out,
                depth_m=depth_out,
                world_T_left_camera=pose_matrix,
                pose_7d=frame.pose_7d,
                timestamp=frame.timestamp,
                timestamp_source=frame.timestamp_source,
                zed_tracking_state=frame.tracking_state,
                pose_confidence=frame.pose_confidence,
                quality=quality,
                sequence_id=sequence_id,
                exposure=frame.exposure,
                gain=frame.gain,
                white_balance=frame.white_balance,
            )
            if show_preview:
                cv2.imshow("ZED DAAAM offline capture", bgr_out)
                if cv2.waitKey(1) in (27, ord("q")):
                    break
            if progress_callback is not None:
                progress_callback(frame, quality)
    finally:
        if show_preview:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.close(
                frames_grabbed=grabbed,
                extra={
                    "frames_skipped_bad_tracking": skipped_tracking,
                    "frames_skipped_stride": skipped_stride,
                    "pose_reset_count": pose_reset_count,
                    "elapsed_wall_sec": time.time() - start,
                },
            )
        camera.close()

    return CaptureSummary(
        output_dir=Path(output_dir),
        frames_saved=writer.frame_id if writer is not None else 0,
        frames_grabbed=grabbed,
        frames_skipped_bad_tracking=skipped_tracking,
        frames_skipped_stride=skipped_stride,
        pose_reset_count=pose_reset_count,
    )


class ZedCamera:
    def __init__(self, settings: CaptureSettings) -> None:
        self.settings = settings
        self.sl: Any | None = None
        self.zed: Any | None = None
        self.pose: Any | None = None
        self.left_mat: Any | None = None
        self.right_mat: Any | None = None
        self.depth_mat: Any | None = None
        self.runtime: Any | None = None
        self._recording_enabled = False

    def open(self) -> dict[str, Any]:
        import pyzed.sl as sl  # type: ignore

        self.sl = sl
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = _enum_value(sl.RESOLUTION, self.settings.resolution)
        init.camera_fps = int(self.settings.fps)
        init.depth_mode = _enum_value(sl.DEPTH_MODE, self.settings.depth_mode)
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = _enum_value(sl.COORDINATE_SYSTEM, self.settings.coordinate_system)
        init.depth_minimum_distance = float(self.settings.min_depth_m)
        init.depth_maximum_distance = float(self.settings.max_depth_m)
        if self.settings.svo_file is not None:
            init.set_from_svo_file(str(self.settings.svo_file))

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        tracking_params = sl.PositionalTrackingParameters()
        if hasattr(tracking_params, "enable_area_memory"):
            tracking_params.enable_area_memory = bool(self.settings.area_memory)
        err = self.zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError(f"Failed to enable ZED positional tracking: {err}")

        self.pose = sl.Pose()
        self.left_mat = sl.Mat()
        self.right_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.runtime = sl.RuntimeParameters()
        return _camera_info(self.zed)

    def grab(self) -> CameraFrame | None:
        assert self.sl is not None and self.zed is not None
        err = self.zed.grab(self.runtime)
        if err != self.sl.ERROR_CODE.SUCCESS:
            return None

        tracking_state = self.zed.get_position(self.pose, self.sl.REFERENCE_FRAME.WORLD)
        self.zed.retrieve_image(self.left_mat, self.sl.VIEW.LEFT)
        self.zed.retrieve_image(self.right_mat, self.sl.VIEW.RIGHT)
        self.zed.retrieve_measure(self.depth_mat, self.sl.MEASURE.DEPTH)

        left_bgra = _sdk_mat_snapshot(self.left_mat.get_data())
        right_bgra = _sdk_mat_snapshot(self.right_mat.get_data())
        depth_raw = _sdk_mat_snapshot(self.depth_mat.get_data())
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., 0]
        depth_m = np.ascontiguousarray(depth_raw, dtype=np.float32)
        left_bgr = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2BGR)
        t, q = _pose_translation_orientation(self.pose, self.sl)
        T = pose_to_matrix(t, q)
        return CameraFrame(
            left_bgr=left_bgr,
            right_bgr=right_bgr,
            depth_m=depth_m,
            world_T_left_camera=T,
            pose_7d=Pose7D(float(t[0]), float(t[1]), float(t[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            timestamp=_timestamp_sec(self.zed, self.sl, self.pose),
            timestamp_source="zed_image",
            tracking_state=_enum_name(tracking_state),
            pose_confidence=getattr(self.pose, "pose_confidence", None),
            exposure=_camera_setting(self.zed, self.sl, "EXPOSURE"),
            gain=_camera_setting(self.zed, self.sl, "GAIN"),
            white_balance=_camera_setting(self.zed, self.sl, "WHITEBALANCE_TEMPERATURE"),
        )

    def close(self) -> None:
        if self.zed is not None:
            if self._recording_enabled:
                try:
                    self.zed.disable_recording()
                except Exception:
                    pass
            try:
                self.zed.disable_positional_tracking()
            except Exception:
                pass
            self.zed.close()

    def enable_svo_recording(self, path: Path) -> None:
        assert self.sl is not None and self.zed is not None
        compression = getattr(self.sl.SVO_COMPRESSION_MODE, "H264", None)
        if compression is None:
            compression = getattr(self.sl.SVO_COMPRESSION_MODE, "H264_LOSSLESS", None)
        params = self.sl.RecordingParameters(str(path), compression) if compression is not None else self.sl.RecordingParameters(str(path))
        err = self.zed.enable_recording(params)
        if err != self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable SVO recording: {err}")
        self._recording_enabled = True


def _sdk_mat_snapshot(data: Any) -> np.ndarray:
    return np.array(np.asarray(data), copy=True, order="C")


def _enum_value(enum_cls: Any, name: str) -> Any:
    if hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    valid = [k for k in dir(enum_cls) if k.isupper() and not k.startswith("_")]
    raise ValueError(f"Invalid enum value {name!r} for {enum_cls}. Valid examples: {valid[:20]}")


def _enum_name(value: Any) -> str:
    return str(value).split(".")[-1].split("::")[-1]


def _camera_info(zed: Any) -> dict[str, Any]:
    info = zed.get_camera_information()
    cfg = info.camera_configuration
    res = cfg.resolution
    calib = cfg.calibration_parameters
    left = calib.left_cam
    distortion: list[float] = []
    if hasattr(left, "disto"):
        try:
            distortion = [float(x) for x in left.disto]
        except Exception:
            distortion = []
    return {
        "native_width": int(res.width),
        "native_height": int(res.height),
        "intrinsics": [[float(left.fx), 0.0, float(left.cx)], [0.0, float(left.fy), float(left.cy)], [0.0, 0.0, 1.0]],
        "distortion": distortion,
        "zed": {
            "serial_number": getattr(info, "serial_number", None),
            "camera_model": str(getattr(info, "camera_model", "")),
            "firmware_version": str(getattr(info, "camera_firmware_version", "")),
            "sdk_version": _sdk_version(),
        },
    }


def _sdk_version() -> str | None:
    try:
        import pyzed.sl as sl  # type: ignore

        if hasattr(sl, "Camera") and hasattr(sl.Camera, "get_sdk_version"):
            return str(sl.Camera.get_sdk_version())
    except Exception:
        return None
    return None


def _pose_translation_orientation(zed_pose: Any, sl: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        translation_obj = zed_pose.get_translation()
    except TypeError:
        tmp = sl.Translation()
        translation_obj = zed_pose.get_translation(tmp)
    t = _vector_to_numpy(translation_obj, ["tx", "ty", "tz"], 3)
    try:
        orientation_obj = zed_pose.get_orientation()
    except TypeError:
        tmp = sl.Orientation()
        orientation_obj = zed_pose.get_orientation(tmp)
    q = _vector_to_numpy(orientation_obj, ["ox", "oy", "oz", "ow"], 4)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return t, q


def _vector_to_numpy(obj: Any, attrs: list[str], size: int) -> np.ndarray:
    if hasattr(obj, "get"):
        arr = np.asarray(obj.get(), dtype=np.float64).reshape(-1)
        if arr.size >= size:
            return arr[:size]
    values = []
    for attr in attrs:
        if not hasattr(obj, attr):
            break
        values.append(float(getattr(obj, attr)))
    if len(values) == size:
        return np.asarray(values, dtype=np.float64)
    arr = np.asarray(obj, dtype=np.float64).reshape(-1)
    if arr.size >= size:
        return arr[:size]
    raise TypeError(f"Cannot convert object to vector of size {size}: {type(obj)}")


def _timestamp_sec(zed: Any, sl: Any, pose: Any) -> float:
    try:
        ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
    except Exception:
        ts = getattr(pose, "timestamp", None)
    if ts is not None:
        for method, scale in (
            ("get_nanoseconds", 1e-9),
            ("get_microseconds", 1e-6),
            ("get_milliseconds", 1e-3),
            ("get_seconds", 1.0),
        ):
            if hasattr(ts, method):
                return float(getattr(ts, method)()) * scale
    return time.time()


def _camera_setting(zed: Any, sl: Any, setting_name: str) -> float | None:
    if not hasattr(sl, "VIDEO_SETTINGS") or not hasattr(sl.VIDEO_SETTINGS, setting_name):
        return None
    try:
        result = zed.get_camera_settings(getattr(sl.VIDEO_SETTINGS, setting_name))
        if isinstance(result, tuple) and len(result) >= 2:
            return float(result[1])
        if isinstance(result, (int, float)):
            return float(result)
    except Exception:
        return None
    return None
