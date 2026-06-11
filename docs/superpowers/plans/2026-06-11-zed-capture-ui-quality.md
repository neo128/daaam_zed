# ZED Capture UI Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable ZED capture core, per-frame quality metadata, end-of-capture validation, and PySide6 desktop UI described in `docs/superpowers/specs/2026-06-11-zed-capture-ui-design.md`.

**Architecture:** Extract the current CLI behavior into `src/daaam_zed` modules with clear boundaries: geometry/pose math, quality statistics, dataset writing, ZED camera access, validation, and UI worker/main window. The CLI and UI will both use the same library so DAAAM layout, timestamps, pose handling, and quality metadata stay consistent.

**Tech Stack:** Python 3.12, `numpy`, `opencv-python`, `pyzed.sl`, `PySide6`, `pytest`.

---

## File Structure

- Create `src/daaam_zed/__init__.py`: package marker and version.
- Create `src/daaam_zed/models.py`: dataclasses for capture settings, frame payloads, pose, quality stats, manifest entries, and capture summaries.
- Create `src/daaam_zed/geometry.py`: crop/resize, intrinsics adjustment, depth sanitization, quaternion-to-matrix, pose conversion.
- Create `src/daaam_zed/quality.py`: depth/RGB statistics, pose delta classification, quality report generation.
- Create `src/daaam_zed/dataset_writer.py`: atomic dataset writes, aligned timestamps, `manifest.json`, `manifest.jsonl`, `quality_report.json`.
- Create `src/daaam_zed/zed_capture.py`: ZED SDK wrapper for open/grab/retrieve/pose/camera metadata.
- Create `src/daaam_zed/validation.py`: reusable dataset validation with legacy layout and new metadata checks.
- Create `src/daaam_zed/ui.py`: PySide6 main window and worker thread.
- Modify `scripts/capture_zed_to_daaam_dataset.py`: keep CLI arguments, delegate capture to the library.
- Modify `scripts/validate_daaam_image_sequence.py`: delegate validation to `src/daaam_zed/validation.py`.
- Create `scripts/zed_capture_ui.py`: UI entry point.
- Modify `requirements.txt`: add `PySide6` and `pytest`.
- Create `tests/`: unit tests for geometry, quality, writer, validation, and UI import.

## Task 1: Test Harness And Package Skeleton

**Files:**
- Create: `src/daaam_zed/__init__.py`
- Create: `src/daaam_zed/models.py`
- Create: `tests/test_imports.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing import tests**

```python
def test_package_imports():
    import daaam_zed
    assert daaam_zed.__version__


def test_models_import():
    from daaam_zed.models import CaptureSettings
    settings = CaptureSettings()
    assert settings.resolution == "HD720"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_imports.py -v`

Expected: FAIL because `daaam_zed` does not exist.

- [ ] **Step 3: Add minimal package and dependencies**

Create `src/daaam_zed/__init__.py` with `__version__ = "0.1.0"`.

Create `src/daaam_zed/models.py` with a minimal `CaptureSettings` dataclass.

Add `PySide6` and `pytest` to `requirements.txt`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_imports.py -v`

Expected: PASS.

## Task 2: Geometry And Pose Utilities

**Files:**
- Create: `src/daaam_zed/geometry.py`
- Create: `tests/test_geometry.py`

- [ ] **Step 1: Write failing tests for crop/resize, depth sanitization, and pose conversion**
- [ ] **Step 2: Run tests and confirm missing functions fail**
- [ ] **Step 3: Move equivalent logic from the CLI script into `geometry.py`**
- [ ] **Step 4: Run geometry tests and confirm pass**

## Task 3: Quality Statistics And Pose Reset Detection

**Files:**
- Create: `src/daaam_zed/quality.py`
- Create: `tests/test_quality.py`

- [ ] **Step 1: Write failing tests for depth stats, RGB stats, pose delta, warning threshold, reset threshold, and quality report**
- [ ] **Step 2: Run tests and confirm missing functions fail**
- [ ] **Step 3: Implement `quality.py` using deterministic numpy operations**
- [ ] **Step 4: Run quality tests and confirm pass**

## Task 4: Atomic Dataset Writer And Metadata

**Files:**
- Create: `src/daaam_zed/dataset_writer.py`
- Create: `tests/test_dataset_writer.py`

- [ ] **Step 1: Write failing tests for aligned RGB/depth/pose/timestamp writes, `manifest.jsonl`, skipped-frame entries, camera metadata, and `quality_report.json`**
- [ ] **Step 2: Run tests and confirm missing writer fails**
- [ ] **Step 3: Implement writer with temporary files and final rename before appending text rows**
- [ ] **Step 4: Run writer tests and confirm pass**

## Task 5: Validation Library

**Files:**
- Create: `src/daaam_zed/validation.py`
- Modify: `scripts/validate_daaam_image_sequence.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing tests for count mismatch, timestamp monotonicity, invalid pose matrix, non-normalized quaternion, and pose reset summary**
- [ ] **Step 2: Run tests and confirm missing validation fails**
- [ ] **Step 3: Implement reusable validation and update script wrapper**
- [ ] **Step 4: Run validation tests and confirm pass**

## Task 6: ZED Capture Core And CLI Delegation

**Files:**
- Create: `src/daaam_zed/zed_capture.py`
- Modify: `scripts/capture_zed_to_daaam_dataset.py`
- Create: `tests/test_capture_cli.py`

- [ ] **Step 1: Write failing tests for CLI argument-to-settings mapping and capture runner dependency injection**
- [ ] **Step 2: Run tests and confirm missing runner fails**
- [ ] **Step 3: Implement ZED camera wrapper and a capture loop that can use a fake camera in tests**
- [ ] **Step 4: Update the CLI to delegate to the capture runner**
- [ ] **Step 5: Run CLI tests and confirm pass**

## Task 7: PySide6 UI

**Files:**
- Create: `src/daaam_zed/ui.py`
- Create: `scripts/zed_capture_ui.py`
- Create: `tests/test_ui_import.py`

- [ ] **Step 1: Write failing import and widget construction tests using offscreen Qt**
- [ ] **Step 2: Run tests and confirm UI module missing fails**
- [ ] **Step 3: Implement main window, controls, metric panels, worker thread API, and image conversion helpers**
- [ ] **Step 4: Run UI import tests and confirm pass**

## Task 8: Documentation And End-To-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with CLI, UI, group-permission launch, metadata, and quality report usage**
- [ ] **Step 2: Run all automated tests**

Run: `PYTHONPATH=src QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -v`

Expected: all tests pass.

- [ ] **Step 3: Run script help/import smoke tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/capture_zed_to_daaam_dataset.py --help
PYTHONPATH=src .venv/bin/python scripts/validate_daaam_image_sequence.py --help
PYTHONPATH=src QT_QPA_PLATFORM=offscreen .venv/bin/python -c "from daaam_zed.ui import MainWindow; print(MainWindow)"
```

Expected: each command exits 0.

- [ ] **Step 4: Commit implementation**

Commit all implementation, tests, and docs with a message describing the UI and quality metadata work.
