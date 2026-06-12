from __future__ import annotations

import os
from pathlib import Path


def test_prepare_qt_environment_removes_opencv_plugin_paths(monkeypatch):
    cv2_plugin_path = "/tmp/project/.venv/lib/python3.12/site-packages/cv2/qt/plugins"
    cv2_font_path = "/tmp/project/.venv/lib/python3.12/site-packages/cv2/qt/fonts"
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", cv2_plugin_path)
    monkeypatch.setenv("QT_QPA_FONTDIR", cv2_font_path)

    from daaam_zed.qt_env import prepare_qt_environment

    plugin_root = prepare_qt_environment()

    platform_path = Path(plugin_root)
    assert platform_path.exists()
    assert platform_path.name == "plugins"
    assert "cv2" not in platform_path.parts
    assert "cv2" not in Path(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]).parts
    assert "QT_QPA_FONTDIR" not in os.environ
