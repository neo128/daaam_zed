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
    app.processEvents()
    window.close()


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
