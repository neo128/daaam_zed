import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_constructs_with_required_controls():
    from PySide6.QtWidgets import QApplication, QPushButton

    from daaam_zed.ui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(start_worker=False)

    assert window.windowTitle() == "DAAAM ZED Capture"
    assert window.status_value.text() == "Closed"
    button_texts = {button.text() for button in window.findChildren(QPushButton)}
    assert {
        "Open Device",
        "Start Capture",
        "Pause Saving",
        "Resume Saving",
        "Stop Capture",
        "Restart Device",
        "Close Device",
        "Validate Dataset",
        "Open Output Folder",
    }.issubset(button_texts)
    assert window.left_preview.minimumWidth() >= 320
    assert window.right_preview.minimumWidth() >= 320
    assert window.depth_preview.minimumWidth() >= 320
    app.processEvents()
    window.close()


def test_preview_widget_size_hint_stays_stable_across_frame_updates():
    import numpy as np
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QApplication

    from daaam_zed.ui import PreviewLabel

    app = QApplication.instance() or QApplication([])
    preview = PreviewLabel("Preview")
    assert preview.sizeHint() == QSize(320, 240)

    preview.resize(640, 480)
    preview.set_bgr_image(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert preview.sizeHint() == QSize(320, 240)

    preview.set_bgr_image(np.zeros((480, 640, 3), dtype=np.uint8))
    assert preview.sizeHint() == QSize(320, 240)
    app.processEvents()


def test_main_window_updates_left_right_and_depth_previews():
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from daaam_zed.ui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(start_worker=False)

    left = np.zeros((8, 12, 3), dtype=np.uint8)
    right = np.full((8, 12, 3), 80, dtype=np.uint8)
    depth = np.linspace(0.0, 2.0, 96, dtype=np.float32).reshape(8, 12)
    window._update_preview(left, right, depth)

    assert window.left_preview.has_image()
    assert window.right_preview.has_image()
    assert window.depth_preview.has_image()
    app.processEvents()
    window.close()


def test_capture_worker_emits_preview_frames_as_independent_snapshots():
    import numpy as np

    from daaam_zed.models import CameraFrame, Pose7D
    from daaam_zed.ui import CaptureWorker

    class FakeCamera:
        def __init__(self, frame):
            self.frame = frame

        def grab(self):
            return self.frame

    left = np.zeros((4, 6, 3), dtype=np.uint8)
    right = np.ones((4, 6, 3), dtype=np.uint8)
    depth = np.ones((4, 6), dtype=np.float32)
    frame = CameraFrame(
        left_bgr=left,
        right_bgr=right,
        depth_m=depth,
        world_T_left_camera=np.eye(4, dtype=np.float64),
        pose_7d=Pose7D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        timestamp=1.0,
        timestamp_source="test",
        tracking_state="OK",
        pose_confidence=None,
    )
    worker = CaptureWorker()
    worker._camera = FakeCamera(frame)
    captured = []
    worker.preview_frame.connect(lambda left_img, right_img, depth_img: captured.append((left_img, right_img, depth_img)))

    worker._grab_preview_frame()

    assert len(captured) == 1
    left_snapshot, right_snapshot, depth_snapshot = captured[0]
    assert left_snapshot.flags.c_contiguous
    assert right_snapshot.flags.c_contiguous
    assert depth_snapshot.flags.c_contiguous
    assert not np.shares_memory(left_snapshot, left)
    assert not np.shares_memory(right_snapshot, right)
    assert not np.shares_memory(depth_snapshot, depth)


def test_main_window_builds_capture_settings_from_controls():
    from PySide6.QtWidgets import QApplication

    from daaam_zed.ui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(start_worker=False)
    window.fps_spin.setValue(30)
    window.max_frames_spin.setValue(120)
    window.drop_bad_tracking_check.setChecked(False)
    window.record_svo_check.setChecked(True)

    settings = window.build_settings()

    assert settings.fps == 30
    assert settings.max_frames == 120
    assert settings.drop_bad_tracking is False
    assert settings.enable_svo_recording is True
    app.processEvents()
    window.close()
