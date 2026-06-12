from __future__ import annotations

import os
from pathlib import Path


def _is_opencv_qt_path(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.replace("\\", "/").lower()
    return "/cv2/" in normalized and "/qt/" in normalized


def _pyside6_plugin_root() -> Path | None:
    try:
        import PySide6
    except Exception:
        return None
    plugin_root = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    return plugin_root if plugin_root.exists() else None


def _remove_opencv_entries(value: str) -> str:
    entries = [entry for entry in value.split(os.pathsep) if entry and not _is_opencv_qt_path(entry)]
    return os.pathsep.join(entries)


def prepare_qt_environment() -> Path | None:
    """Keep PySide6 from loading OpenCV's bundled Qt platform plugins."""

    plugin_root = _pyside6_plugin_root()

    if _is_opencv_qt_path(os.environ.get("QT_QPA_FONTDIR")):
        os.environ.pop("QT_QPA_FONTDIR", None)

    current_platform_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if plugin_root is not None and (not current_platform_path or _is_opencv_qt_path(current_platform_path)):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_root)
    elif _is_opencv_qt_path(current_platform_path):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    current_plugin_path = os.environ.get("QT_PLUGIN_PATH")
    if current_plugin_path:
        cleaned = _remove_opencv_entries(current_plugin_path)
        if cleaned:
            os.environ["QT_PLUGIN_PATH"] = cleaned
        else:
            os.environ.pop("QT_PLUGIN_PATH", None)

    return plugin_root
