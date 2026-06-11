#!/usr/bin/env python3
"""Report whether the local machine can run the ZED capture script.

This intentionally uses only the Python standard library so it can run before
opencv-python or pyzed are installed.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def print_header(title: str) -> None:
    print(f"\n== {title} ==")


def import_status(module: str) -> bool:
    try:
        ok = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        ok = False
    print(f"{module}: {'OK' if ok else 'missing'}")
    return ok


def run_text_command(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=20)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}"


def print_matching_lines(text: str, needles: tuple[str, ...], context: int = 2) -> bool:
    lines = text.splitlines()
    matches: set[int] = set()
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(needle in lower for needle in needles):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                matches.add(j)
    for i in sorted(matches):
        print(lines[i])
    return bool(matches)


def check_macos_devices() -> None:
    print_header("macOS Camera Devices")
    camera_text = run_text_command(["system_profiler", "SPCameraDataType"])
    if not print_matching_lines(camera_text, ("zed", "stereo", "camera", "相机"), context=1):
        print("No camera-like entries reported by SPCameraDataType.")

    print_header("macOS USB Devices")
    usb_text = run_text_command(["ioreg", "-p", "IOUSB", "-w0"])
    if not print_matching_lines(usb_text, ("zed", "stereo", "2b03", "camera", "uvc"), context=3):
        print("No ZED-like USB entries found.")


def main() -> int:
    print_header("Host")
    print(f"platform: {platform.platform()}")
    print(f"python:   {sys.executable}")
    print(f"version:  {sys.version.split()[0]}")

    print_header("Python Modules")
    numpy_ok = import_status("numpy")
    cv2_ok = import_status("cv2")
    pyzed_pkg_ok = import_status("pyzed")
    pyzed_sl_ok = import_status("pyzed.sl")

    print_header("ZED SDK Files")
    zed_root = Path("/usr/local/zed")
    print(f"/usr/local/zed: {'exists' if zed_root.exists() else 'missing'}")
    for tool in ("ZED_Explorer", "ZED_Depth_Viewer", "ZED_Diagnostic"):
        print(f"{tool}: {shutil.which(tool) or 'not on PATH'}")

    if platform.system() == "Darwin":
        check_macos_devices()
        print_header("macOS Note")
        print(
            "A ZED USB/HID entry only proves that macOS can see the device. "
            "The capture script still needs pyzed.sl from the Stereolabs ZED SDK "
            "to save depth, tracking, and pose."
        )

    print_header("Summary")
    if numpy_ok and cv2_ok and pyzed_pkg_ok and pyzed_sl_ok:
        print("READY: Python dependencies for full ZED capture are importable.")
        return 0

    print("NOT READY: full RGB/depth/pose capture cannot run in this Python environment yet.")
    if not cv2_ok:
        print("- Install OpenCV for this Python environment: python3 -m pip install opencv-python")
    if not pyzed_sl_ok:
        print("- Install the ZED SDK Python API for this Python environment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
