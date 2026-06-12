from __future__ import annotations

import queue
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .qt_env import prepare_qt_environment

prepare_qt_environment()

from PySide6.QtCore import QRect, QSize, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import cv2

prepare_qt_environment()

from .dataset_writer import DatasetWriter
from .geometry import apply_crop_resize, compute_output_geometry, sanitize_depth
from .models import CameraFrame, CaptureSettings
from .quality import classify_pose_delta, compute_depth_quality, compute_rgb_quality, merge_quality
from .validation import validate_dataset
from .zed_capture import ZedCamera


class CaptureWorker(QThread):
    preview_frame = Signal(object, object, object)
    status_changed = Signal(str)
    metrics_changed = Signal(dict)
    pose_changed = Signal(object)
    quality_changed = Signal(object)
    warning = Signal(str)
    error = Signal(str)
    capture_finished = Signal(object)

    def __init__(self, camera_factory=ZedCamera) -> None:  # noqa: ANN001
        super().__init__()
        self._commands: queue.Queue[tuple[str, tuple[Any, ...]]] = queue.Queue()
        self._camera_factory = camera_factory
        self._camera: Any | None = None
        self._camera_info: dict[str, Any] | None = None
        self._writer: DatasetWriter | None = None
        self._settings = CaptureSettings()
        self._output_dir: Path | None = None
        self._running = True
        self._recording = False
        self._paused = False
        self._grabbed = 0
        self._skipped_tracking = 0
        self._skipped_stride = 0
        self._sequence_id = 0
        self._pose_reset_count = 0
        self._previous_pose: np.ndarray | None = None
        self._start_time = time.time()

    def open_device(self, settings: CaptureSettings) -> None:
        self._commands.put(("open", (settings,)))

    def start_capture(self, output_dir: Path, settings: CaptureSettings, overwrite: bool) -> None:
        self._commands.put(("start", (output_dir, settings, overwrite)))

    def pause_saving(self) -> None:
        self._commands.put(("pause", ()))

    def resume_saving(self) -> None:
        self._commands.put(("resume", ()))

    def stop_capture(self) -> None:
        self._commands.put(("stop", ()))

    def restart_device(self, settings: CaptureSettings) -> None:
        self._commands.put(("restart", (settings,)))

    def close_device(self) -> None:
        self._commands.put(("close", ()))

    def shutdown(self) -> None:
        self._commands.put(("shutdown", ()))

    def run(self) -> None:
        self.status_changed.emit("Closed")
        while self._running:
            self._drain_commands()
            if self._camera is not None:
                self._grab_preview_frame()
            else:
                self.msleep(50)
        self._close_capture()
        self._close_camera()

    def _drain_commands(self) -> None:
        while True:
            try:
                command, args = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                if command == "open":
                    self._open_camera(args[0])
                elif command == "start":
                    self._start_capture(args[0], args[1], args[2])
                elif command == "pause":
                    self._paused = True
                    self.status_changed.emit("Paused")
                elif command == "resume":
                    self._paused = False
                    self.status_changed.emit("Recording" if self._recording else "Previewing")
                elif command == "stop":
                    self._close_capture()
                    self.status_changed.emit("Previewing" if self._camera is not None else "Closed")
                elif command == "restart":
                    self._close_capture()
                    self._close_camera()
                    self._open_camera(args[0])
                elif command == "close":
                    self._close_capture()
                    self._close_camera()
                    self.status_changed.emit("Closed")
                elif command == "shutdown":
                    self._running = False
            except Exception as exc:
                self.status_changed.emit("Error")
                self.error.emit(str(exc))

    def _open_camera(self, settings: CaptureSettings) -> None:
        self.status_changed.emit("Opening")
        self._settings = settings
        self._camera = self._camera_factory(settings)
        self._camera_info = self._camera.open()
        self._grabbed = 0
        self._previous_pose = None
        self.status_changed.emit("Previewing")

    def _start_capture(self, output_dir: Path, settings: CaptureSettings, overwrite: bool) -> None:
        if self._camera is None:
            self._open_camera(settings)
        assert self._camera_info is not None
        self._settings = settings
        native_w = int(self._camera_info["native_width"])
        native_h = int(self._camera_info["native_height"])
        out_w, out_h, crop, K_out = compute_output_geometry(
            native_w,
            native_h,
            np.asarray(self._camera_info["intrinsics"], dtype=np.float64),
            settings.output_width,
            settings.output_height,
            settings.resize_mode,
        )
        writer_info = {
            "width": out_w,
            "height": out_h,
            "intrinsics": K_out.tolist(),
            "distortion": self._camera_info.get("distortion", []),
            "native_width": native_w,
            "native_height": native_h,
            "crop_xywh": list(crop),
            "zed": self._camera_info.get("zed", {}),
            "extrinsics": self._camera_info.get("extrinsics"),
        }
        self._writer = DatasetWriter(output_dir, settings, writer_info, overwrite=overwrite)
        self._output_dir = output_dir
        self._recording = True
        self._paused = False
        self._sequence_id = 0
        self._pose_reset_count = 0
        self._previous_pose = None
        self._start_time = time.time()
        self.status_changed.emit("Recording")

    def _grab_preview_frame(self) -> None:
        assert self._camera is not None
        frame = self._camera.grab()
        if frame is None:
            self.msleep(5)
            return
        self._grabbed += 1
        self.preview_frame.emit(_preview_snapshot(frame.left_bgr), _preview_snapshot(frame.right_bgr), _preview_snapshot(frame.depth_m))
        self.pose_changed.emit(frame.pose_7d)
        if self._recording and not self._paused and self._writer is not None:
            self._save_frame(frame)
        self._emit_metrics(frame)

    def _save_frame(self, frame: CameraFrame) -> None:
        assert self._writer is not None and self._camera_info is not None
        if self._settings.drop_bad_tracking and frame.tracking_state != "OK":
            self._skipped_tracking += 1
            self._writer.write_skipped_frame(
                grabbed_index=self._grabbed,
                timestamp=frame.timestamp,
                skip_reason="BAD_TRACKING",
                zed_tracking_state=frame.tracking_state,
                pose_confidence=frame.pose_confidence,
            )
            return
        native_w = int(self._camera_info["native_width"])
        native_h = int(self._camera_info["native_height"])
        out_w, out_h, crop, _K_out = compute_output_geometry(
            native_w,
            native_h,
            np.asarray(self._camera_info["intrinsics"], dtype=np.float64),
            self._settings.output_width,
            self._settings.output_height,
            self._settings.resize_mode,
        )
        bgr_out, depth_out = apply_crop_resize(frame.left_bgr, frame.depth_m, out_w, out_h, crop, self._settings.resize_mode)
        if not self._settings.keep_invalid_depth:
            depth_out = sanitize_depth(depth_out, self._settings.min_depth_m, self._settings.max_depth_m)
        quality = merge_quality(compute_depth_quality(depth_out), compute_rgb_quality(bgr_out))
        pose_matrix = np.asarray(frame.world_T_left_camera, dtype=np.float64).reshape(4, 4)
        if self._previous_pose is not None:
            pose_quality = classify_pose_delta(self._previous_pose, pose_matrix)
            quality = merge_quality(quality, pose_quality)
            if pose_quality.warnings:
                self.warning.emit(", ".join(pose_quality.warnings))
            if pose_quality.pose_reset:
                self._pose_reset_count += 1
                self._sequence_id += 1
        self._previous_pose = pose_matrix.copy()
        self.quality_changed.emit(quality)
        self._writer.write_frame(
            bgr=bgr_out,
            depth_m=depth_out,
            world_T_left_camera=pose_matrix,
            pose_7d=frame.pose_7d,
            timestamp=frame.timestamp,
            timestamp_source=frame.timestamp_source,
            zed_tracking_state=frame.tracking_state,
            pose_confidence=frame.pose_confidence,
            quality=quality,
            sequence_id=self._sequence_id,
            exposure=frame.exposure,
            gain=frame.gain,
            white_balance=frame.white_balance,
        )

    def _emit_metrics(self, frame: CameraFrame) -> None:
        saved = self._writer.frame_id if self._writer is not None else 0
        elapsed = max(time.time() - self._start_time, 1e-6)
        self.metrics_changed.emit(
            {
                "preview_fps": 1.0 / max(elapsed / max(self._grabbed, 1), 1e-6),
                "saved_fps": saved / elapsed,
                "saved_frames": saved,
                "grabbed_frames": self._grabbed,
                "skipped_tracking": self._skipped_tracking,
                "pose_reset_count": self._pose_reset_count,
                "sequence_id": self._sequence_id,
                "tracking_state": frame.tracking_state,
                "pose_confidence": frame.pose_confidence,
                "timestamp": frame.timestamp,
            }
        )

    def _close_capture(self) -> None:
        if self._writer is not None:
            self._writer.close(
                frames_grabbed=self._grabbed,
                extra={
                    "frames_skipped_bad_tracking": self._skipped_tracking,
                    "frames_skipped_stride": self._skipped_stride,
                    "pose_reset_count": self._pose_reset_count,
                },
            )
            self.capture_finished.emit({"output_dir": str(self._output_dir), "frames_saved": self._writer.frame_id})
            self._writer = None
        self._recording = False
        self._paused = False

    def _close_camera(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None
            self._camera_info = None


def _preview_snapshot(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    return np.ascontiguousarray(np.asarray(image)).copy()


def depth_to_bgr_preview(depth_m: np.ndarray, min_depth_m: float = 0.05, max_depth_m: float = 5.0) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth > min_depth_m)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if max_depth_m <= min_depth_m:
        max_depth_m = min_depth_m + 1.0
    clipped = np.clip(depth, min_depth_m, max_depth_m)
    normalized[valid] = (255.0 * (1.0 - (clipped[valid] - min_depth_m) / (max_depth_m - min_depth_m))).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


class PreviewLabel(QWidget):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._placeholder = text
        self._image: QImage | None = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setStyleSheet("background: #111; color: #ddd;")

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(320, 240)

    def has_image(self) -> bool:
        return self._image is not None and not self._image.isNull()

    def set_bgr_image(self, bgr: np.ndarray | None) -> None:
        if bgr is None:
            self._placeholder = "Unavailable"
            self._image = None
            self.update()
            return
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape
        self._image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.update()

    def set_depth_image(self, depth_m: np.ndarray | None) -> None:
        if depth_m is None:
            self._placeholder = "Depth unavailable"
            self._image = None
            self.update()
            return
        self.set_bgr_image(depth_to_bgr_preview(depth_m))

    def paintEvent(self, event) -> None:  # noqa: ANN001, N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image is None or self._image.isNull():
            painter.setPen(Qt.GlobalColor.lightGray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            painter.setPen(Qt.GlobalColor.darkGray)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
            return
        scaled = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawImage(QRect(x, y, scaled.width(), scaled.height()), self._image)
        painter.setPen(Qt.GlobalColor.darkGray)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class MainWindow(QMainWindow):
    def __init__(self, start_worker: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("DAAAM ZED Capture")
        self.worker: CaptureWorker | None = CaptureWorker() if start_worker else None
        self._build_ui()
        self._connect_signals()
        self._set_state("Closed")
        if self.worker is not None:
            self.worker.start()

    def build_settings(self) -> CaptureSettings:
        return CaptureSettings(
            resolution=self.resolution_combo.currentText(),
            fps=self.fps_spin.value(),
            depth_mode=self.depth_mode_combo.currentText(),
            output_width=self.width_spin.value(),
            output_height=self.height_spin.value(),
            resize_mode=self.resize_mode_combo.currentText(),
            max_frames=self.max_frames_spin.value(),
            drop_bad_tracking=self.drop_bad_tracking_check.isChecked(),
            enable_svo_recording=self.record_svo_check.isChecked(),
        )

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)

        preview_layout = QGridLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        preview_layout.setColumnStretch(0, 1)
        preview_layout.setColumnStretch(1, 1)
        preview_layout.setRowStretch(0, 1)
        preview_layout.setRowStretch(1, 1)
        self.depth_preview = PreviewLabel("Depth")
        self.left_preview = PreviewLabel("Left Camera")
        self.right_preview = PreviewLabel("Right Camera")
        preview_layout.addWidget(self._preview_panel("Depth", self.depth_preview), 0, 0, 1, 2)
        preview_layout.addWidget(self._preview_panel("Left RGB", self.left_preview), 1, 0)
        preview_layout.addWidget(self._preview_panel("Right RGB", self.right_preview), 1, 1)
        root.addLayout(preview_layout, 2)

        side = QVBoxLayout()
        status_box = QGroupBox("Status")
        status_form = QFormLayout(status_box)
        self.status_value = QLabel("Closed")
        self.preview_fps_value = QLabel("0.00")
        self.saved_fps_value = QLabel("0.00")
        self.saved_frames_value = QLabel("0")
        self.grabbed_frames_value = QLabel("0")
        self.tracking_value = QLabel("-")
        self.pose_value = QLabel("-")
        self.quality_value = QLabel("-")
        status_form.addRow("Device", self.status_value)
        status_form.addRow("Preview FPS", self.preview_fps_value)
        status_form.addRow("Saved FPS", self.saved_fps_value)
        status_form.addRow("Saved Frames", self.saved_frames_value)
        status_form.addRow("Grabbed Frames", self.grabbed_frames_value)
        status_form.addRow("Tracking", self.tracking_value)
        status_form.addRow("Pose", self.pose_value)
        status_form.addRow("Quality", self.quality_value)
        side.addWidget(status_box)

        settings_box = QGroupBox("Capture")
        settings_form = QFormLayout(settings_box)
        self.output_dir_edit = QLineEdit(str(Path.home() / "datasets" / "zedm_daaam_ui"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["HD720", "HD1080", "VGA"])
        self.depth_mode_combo = QComboBox()
        self.depth_mode_combo.addItems(["ULTRA", "QUALITY", "NEURAL"])
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 100)
        self.fps_spin.setValue(15)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 10000)
        self.width_spin.setValue(640)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 10000)
        self.height_spin.setValue(480)
        self.resize_mode_combo = QComboBox()
        self.resize_mode_combo.addItems(["crop_resize", "resize", "none"])
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 1_000_000)
        self.max_frames_spin.setValue(0)
        self.drop_bad_tracking_check = QCheckBox()
        self.drop_bad_tracking_check.setChecked(True)
        self.record_svo_check = QCheckBox()
        self.record_svo_check.setChecked(False)
        settings_form.addRow("Output", self.output_dir_edit)
        settings_form.addRow("Resolution", self.resolution_combo)
        settings_form.addRow("Depth", self.depth_mode_combo)
        settings_form.addRow("FPS", self.fps_spin)
        settings_form.addRow("Width", self.width_spin)
        settings_form.addRow("Height", self.height_spin)
        settings_form.addRow("Resize", self.resize_mode_combo)
        settings_form.addRow("Max Frames", self.max_frames_spin)
        settings_form.addRow("Drop Bad Tracking", self.drop_bad_tracking_check)
        settings_form.addRow("Record Raw SVO", self.record_svo_check)
        side.addWidget(settings_box)

        self.open_button = QPushButton("Open Device")
        self.start_button = QPushButton("Start Capture")
        self.pause_button = QPushButton("Pause Saving")
        self.resume_button = QPushButton("Resume Saving")
        self.stop_button = QPushButton("Stop Capture")
        self.restart_button = QPushButton("Restart Device")
        self.close_button = QPushButton("Close Device")
        self.validate_button = QPushButton("Validate Dataset")
        self.open_folder_button = QPushButton("Open Output Folder")
        self.choose_output_button = QPushButton("Choose Output")
        button_layout = QGridLayout()
        for i, button in enumerate(
            [
                self.open_button,
                self.start_button,
                self.pause_button,
                self.resume_button,
                self.stop_button,
                self.restart_button,
                self.close_button,
                self.validate_button,
                self.open_folder_button,
                self.choose_output_button,
            ]
        ):
            button_layout.addWidget(button, i // 2, i % 2)
        side.addLayout(button_layout)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        side.addWidget(self.log)
        root.addLayout(side, 1)
        self.setCentralWidget(central)

    def _preview_panel(self, title: str, preview: PreviewLabel) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(title_label)
        layout.addWidget(preview, 1)
        return panel

    def _connect_signals(self) -> None:
        self.choose_output_button.clicked.connect(self._choose_output)
        self.open_button.clicked.connect(lambda: self.worker and self.worker.open_device(self.build_settings()))
        self.start_button.clicked.connect(self._start_capture)
        self.pause_button.clicked.connect(lambda: self.worker and self.worker.pause_saving())
        self.resume_button.clicked.connect(lambda: self.worker and self.worker.resume_saving())
        self.stop_button.clicked.connect(lambda: self.worker and self.worker.stop_capture())
        self.restart_button.clicked.connect(lambda: self.worker and self.worker.restart_device(self.build_settings()))
        self.close_button.clicked.connect(lambda: self.worker and self.worker.close_device())
        self.validate_button.clicked.connect(self._validate_dataset)
        self.open_folder_button.clicked.connect(self._open_output_folder)
        if self.worker is not None:
            self.worker.preview_frame.connect(self._update_preview)
            self.worker.status_changed.connect(self._set_state)
            self.worker.metrics_changed.connect(self._update_metrics)
            self.worker.pose_changed.connect(self._update_pose)
            self.worker.quality_changed.connect(self._update_quality)
            self.worker.warning.connect(self._append_warning)
            self.worker.error.connect(self._append_error)
            self.worker.capture_finished.connect(lambda summary: self.log.append(f"Finished: {summary}"))

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output Directory", self.output_dir_edit.text())
        if path:
            self.output_dir_edit.setText(path)

    def _start_capture(self) -> None:
        if self.worker is not None:
            self.worker.start_capture(Path(self.output_dir_edit.text()).expanduser(), self.build_settings(), True)

    def _validate_dataset(self) -> None:
        result = validate_dataset(Path(self.output_dir_edit.text()).expanduser())
        if result.ok:
            self.log.append(f"Validation OK: {result.rgb_count} frames")
        else:
            self.log.append("Validation FAILED: " + "; ".join(result.errors))

    def _open_output_folder(self) -> None:
        path = Path(self.output_dir_edit.text()).expanduser()
        try:
            subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            self._append_error(str(exc))

    def _update_preview(self, left_bgr: object, right_bgr: object, depth_m: object) -> None:
        self.left_preview.set_bgr_image(left_bgr if isinstance(left_bgr, np.ndarray) else None)
        self.right_preview.set_bgr_image(right_bgr if isinstance(right_bgr, np.ndarray) else None)
        self.depth_preview.set_depth_image(depth_m if isinstance(depth_m, np.ndarray) else None)

    def _set_state(self, state: str) -> None:
        self.status_value.setText(state)
        is_closed = state == "Closed"
        is_recording = state == "Recording"
        is_paused = state == "Paused"
        self.open_button.setEnabled(is_closed)
        self.start_button.setEnabled(state in {"Previewing", "Closed"})
        self.pause_button.setEnabled(is_recording)
        self.resume_button.setEnabled(is_paused)
        self.stop_button.setEnabled(is_recording or is_paused)
        self.restart_button.setEnabled(state in {"Previewing", "Recording", "Paused", "Error"})
        self.close_button.setEnabled(not is_closed)

    def _update_metrics(self, metrics: dict) -> None:
        self.preview_fps_value.setText(f"{metrics.get('preview_fps', 0.0):.2f}")
        self.saved_fps_value.setText(f"{metrics.get('saved_fps', 0.0):.2f}")
        self.saved_frames_value.setText(str(metrics.get("saved_frames", 0)))
        self.grabbed_frames_value.setText(str(metrics.get("grabbed_frames", 0)))
        self.tracking_value.setText(str(metrics.get("tracking_state", "-")))

    def _update_pose(self, pose: object) -> None:
        if hasattr(pose, "as_list"):
            self.pose_value.setText(" ".join(f"{v:.3f}" for v in pose.as_list()))

    def _update_quality(self, quality: object) -> None:
        self.quality_value.setText(
            f"depth={getattr(quality, 'depth_valid_ratio', 0.0):.2f} "
            f"bright={getattr(quality, 'rgb_mean_brightness', 0.0):.1f} "
            f"resets={getattr(quality, 'pose_reset', False)}"
        )

    def _append_warning(self, message: str) -> None:
        self.log.append(f"WARNING: {message}")

    def _append_error(self, message: str) -> None:
        self.log.append(f"ERROR: {message}")

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        if self.worker is not None:
            self.worker.shutdown()
            self.worker.wait(3000)
        super().closeEvent(event)


def run_app() -> int:
    prepare_qt_environment()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1280, 720)
    window.show()
    return app.exec()
