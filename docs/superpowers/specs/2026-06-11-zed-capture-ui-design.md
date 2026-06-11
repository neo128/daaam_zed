# ZED Capture UI Design

Date: 2026-06-11

## Goal

Build a local PySide6 desktop UI for ZED-M data collection. The UI should let an operator open the camera, preview the left and right streams, start recording, pause only file saving, resume recording, stop recording, restart the device, and inspect runtime information such as FPS, frame count, tracking state, and camera pose.

The UI must preserve the existing DAAAM `ImageSequenceDataset` output format:

- `rgb/000000.png`
- `depth/000000.npy`
- `pose/poses.txt`
- `pose/poses_7d.txt`
- `camera_info.json`
- `timestamps.txt`
- `manifest.json`

## Chosen Approach

Use approach 2: extract reusable capture logic from `scripts/capture_zed_to_daaam_dataset.py` into a small library module, then build a PySide6 UI around that library.

This keeps the working CLI behavior as the reference implementation while giving the UI direct access to preview frames, FPS, pose, tracking state, and save controls. The CLI script should remain available for batch capture and smoke tests.

## Non-Goals

- No ROS integration.
- No rosbag or topic forwarding.
- No browser UI in the first version.
- No full dataset browser or DAAAM pipeline runner in the first version.
- No multi-camera support in the first version.

## User Workflow

1. The operator launches the UI from the project virtual environment.
2. The operator chooses or types an output directory.
3. The operator clicks `Open Device`.
4. The UI opens the ZED camera and starts live preview.
5. The operator clicks `Start Capture`.
6. The UI writes frames to the selected DAAAM dataset directory.
7. The operator clicks `Pause Saving` to keep preview running without writing frames.
8. The operator clicks `Resume Saving` to continue writing into the same dataset.
9. The operator clicks `Stop Capture` to close pose/timestamp files and write `manifest.json`.
10. The operator can click `Restart Device` to close and reopen the ZED camera after an error or device reset.

## Main Window Layout

The UI is a dense operational tool, not a landing page.

Left content area:

- A left camera preview panel.
- A right camera preview panel.
- Each panel has a stable aspect ratio and a compact status overlay with stream name and resolution.

Right status/control area:

- Device status: `Closed`, `Opening`, `Previewing`, `Recording`, `Paused`, `Stopping`, `Error`.
- Output directory selector.
- Capture settings:
  - resolution, default `HD720`
  - FPS, default `15`
  - depth mode, default `ULTRA`
  - saved width and height, default `640x480`
  - resize mode, default `crop_resize`
  - maximum frames, default `0` for unlimited
  - drop bad tracking, default enabled
- Runtime metrics:
  - preview FPS
  - saved frame count
  - grabbed frame count
  - skipped tracking count
  - tracking state
  - pose confidence
  - timestamp
  - pose as `x y z qx qy qz qw`
  - first-order depth stats for the current frame, such as valid pixel ratio and median depth

Bottom or right-side command area:

- `Open Device`
- `Start Capture`
- `Pause Saving`
- `Resume Saving`
- `Stop Capture`
- `Restart Device`
- `Close Device`
- `Validate Dataset`
- `Open Output Folder`

Button enablement follows the state machine so invalid actions are disabled.

## State Machine

`Closed`

- Camera is not open.
- `Open Device` is enabled.
- capture, pause, resume, stop, and validate are disabled unless an existing dataset path can be validated.

`Opening`

- UI is waiting for camera initialization.
- All device-changing controls are disabled.

`Previewing`

- Camera is open and live preview is running.
- `Start Capture` and `Restart Device` are enabled.
- No files are open for writing.

`Recording`

- Camera preview runs.
- RGB, depth, pose, and timestamp files are written.
- `Pause Saving`, `Stop Capture`, and `Restart Device` are enabled.

`Paused`

- Camera preview runs.
- No frames are saved.
- `Resume Saving`, `Stop Capture`, and `Restart Device` are enabled.

`Stopping`

- UI is closing files and writing `manifest.json`.
- Controls are disabled until the transition completes.

`Error`

- UI shows the most recent error message.
- `Restart Device` and `Close Device` are enabled when possible.

`Close Device` stops any active capture, finalizes open files, closes positional tracking, closes the ZED camera, and returns the UI to `Closed`.

## Capture Library Design

Create a Python package under `src/daaam_zed/` with focused modules:

- `zed_capture.py`
  - owns ZED camera open, close, grab, retrieve left/right/depth, tracking, and pose.
- `dataset_writer.py`
  - owns output directory creation, RGB/depth/pose/timestamp writing, camera info, and manifest writing.
- `geometry.py`
  - owns resize/crop, intrinsics adjustment, depth sanitization, quaternion conversion, and pose matrix conversion.
- `models.py`
  - contains lightweight dataclasses for capture settings, frame payloads, runtime status, and manifest summary.

The existing CLI script should be updated to call the same library. This avoids duplicate camera and writer logic between CLI and UI.

## UI Threading Model

The UI must not call blocking ZED SDK operations on the Qt main thread.

Use a `QThread` worker that owns the ZED camera and dataset writer. The worker emits Qt signals:

- `preview_frame(left_bgr, right_bgr)`
- `status_changed(status)`
- `metrics_changed(metrics)`
- `pose_changed(pose)`
- `error(message)`
- `capture_finished(summary)`

The main thread updates widgets only from signals.

Commands from the UI to the worker:

- `open_device(settings)`
- `start_capture(output_dir, settings)`
- `pause_saving()`
- `resume_saving()`
- `stop_capture()`
- `restart_device(settings)`
- `close_device()`
- `shutdown()`

## Preview Behavior

The preview should retrieve both `sl.VIEW.LEFT` and `sl.VIEW.RIGHT`.

Images are converted from BGRA/BGR to a `QImage` for display. Preview display may be downscaled for UI performance, but saved RGB/depth data must use the configured dataset geometry.

If the right image cannot be retrieved, the UI should keep the left preview and mark the right panel as unavailable instead of crashing.

## Saving Behavior

The writer should save only when the worker is in `Recording`.

When paused:

- ZED grab continues.
- preview continues.
- tracking and pose metrics continue to update.
- no RGB/depth/pose/timestamp rows are written.
- saved frame numbering does not skip values.

When stopped:

- open file handles are closed.
- manifest is written.
- camera remains open and returns to `Previewing`.

## Restart Device Behavior

`Restart Device` performs:

1. If recording or paused, stop capture and finalize the dataset.
2. Close positional tracking.
3. Close the ZED camera.
4. Reopen the ZED camera with the current settings.
5. Return to `Previewing` if successful, or `Error` if not.

## Error Handling

The UI should show errors in a persistent status area and keep a short log panel.

Important errors:

- `pyzed.sl` import failure.
- missing ZED SDK shared library.
- camera open failure.
- positional tracking failure.
- output directory exists and is not empty without overwrite permission.
- failed image/depth write.
- bad or unsupported enum values.
- permission errors for `/dev/video*` or `/usr/local/zed`.

Error messages should include the next useful action when known, for example: re-login to refresh `video` and `zed` groups, or run the UI through `sg zed -c 'sg video -c ...'`.

## Dataset Validation

`Validate Dataset` runs the existing validation logic against the current output directory and reports:

- RGB file count
- depth file count
- pose line count
- camera info size
- first depth valid pixel count and range
- final OK or FAILED status

Validation can run in a background task to avoid freezing the UI.

## Launch Command

Add a script entry point:

```bash
.venv/bin/python scripts/zed_capture_ui.py
```

During the current session, if group membership has not refreshed, the operator can launch:

```bash
sg zed -c 'sg video -c ".venv/bin/python scripts/zed_capture_ui.py"'
```

After logout and login, the plain command should be enough.

## Testing And Verification

Automated checks:

- geometry unit tests for crop/resize intrinsics and pose conversion
- dataset writer unit test using synthetic frames
- import smoke test for UI modules without opening a camera

Manual hardware verification:

- run `scripts/check_zed_environment.py`
- open the UI
- open the device
- verify left and right preview
- record 10 frames to `/tmp/zed_ui_smoke`
- pause and confirm saved frame count stops while preview continues
- resume and confirm saved frame count increases
- stop capture
- run dataset validation and confirm OK

## Implementation Notes

The first implementation should favor reliability over visual decoration. Keep the UI compact, readable, and operational. Avoid nested cards and oversized hero-style layouts. Use icon buttons where appropriate, but text labels are acceptable for critical commands such as start, pause, stop, and restart.
