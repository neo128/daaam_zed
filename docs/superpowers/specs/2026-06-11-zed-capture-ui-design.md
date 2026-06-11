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

The UI should also add first-class capture quality metadata:

- `manifest.jsonl`, one JSON object per saved or intentionally skipped frame
- `quality_report.json`, written at capture end
- optional raw ZED `.svo`, when enabled by the operator and supported by the SDK configuration

The main data-quality problem to address is pose relocalization/reset-style jumps. The capture program must detect, record, warn about, and optionally split sequences at these events instead of relying on manual inspection after collection.

## Chosen Approach

Use approach 2: extract reusable capture logic from `scripts/capture_zed_to_daaam_dataset.py` into a small library module, then build a PySide6 UI around that library.

This keeps the working CLI behavior as the reference implementation while giving the UI direct access to preview frames, FPS, pose, tracking state, and save controls. The CLI script should remain available for batch capture and smoke tests.

## Non-Goals

- No ROS integration.
- No rosbag or topic forwarding.
- No browser UI in the first version.
- No full dataset browser or DAAAM pipeline runner in the first version.
- No multi-camera support in the first version.
- No ROS bag export in the first version. The first version keeps image sequence output and may retain raw ZED `.svo` as the replayable source artifact.

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
  - actual saved FPS over the current recording window
  - saved frame count
  - grabbed frame count
  - skipped tracking count
  - pose jump warning count
  - active sequence id
  - tracking state
  - pose confidence
  - timestamp
  - pose as `x y z qx qy qz qw`
  - first-order depth stats for the current frame, such as valid pixel ratio and median depth
  - first-order RGB stats for the current frame, such as mean brightness, dark ratio, saturation ratio, and sharpness

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
  - owns output directory creation, atomic RGB/depth/pose/timestamp writing, camera info, `manifest.json`, and `manifest.jsonl` writing.
- `geometry.py`
  - owns resize/crop, intrinsics adjustment, depth sanitization, quaternion conversion, and pose matrix conversion.
- `quality.py`
  - owns per-frame RGB/depth statistics, pose delta computation, pose jump classification, and end-of-capture quality report generation.
- `models.py`
  - contains lightweight dataclasses for capture settings, frame payloads, per-frame metadata, runtime status, quality thresholds, and manifest summary.

The existing CLI script should be updated to call the same library. This avoids duplicate camera and writer logic between CLI and UI.

## UI Threading Model

The UI must not call blocking ZED SDK operations on the Qt main thread.

Use a `QThread` worker that owns the ZED camera and dataset writer. The worker emits Qt signals:

- `preview_frame(left_bgr, right_bgr)`
- `status_changed(status)`
- `metrics_changed(metrics)`
- `pose_changed(pose)`
- `quality_changed(quality)`
- `warning(message)`
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

RGB, depth, pose, timestamp, and per-frame metadata must use the same `frame_id`. A frame is committed only after all required modal data is available and written successfully. If any required modality is missing, the writer either skips the entire frame and records a skipped-frame entry in `manifest.jsonl`, or fails the capture when the error is not recoverable. It must not leave mismatched RGB/depth/pose numbering.

File writes should be atomic at frame granularity. The writer should write frame files to temporary names, then rename them into final paths only after the frame payload is complete. Pose and timestamp text rows are appended only after image/depth files are committed.

When paused:

- ZED grab continues.
- preview continues.
- tracking and pose metrics continue to update.
- pose jump and quality warnings can still be displayed as preview diagnostics.
- no RGB/depth/pose/timestamp rows are written.
- saved frame numbering does not skip values.

When stopped:

- open file handles are closed.
- manifest is written.
- quality report is written.
- camera remains open and returns to `Previewing`.

## Timestamp And Metadata Requirements

`timestamps.txt` must contain one real timestamp row per saved frame, aligned with the saved `frame_id`. The timestamp should come from the ZED image timestamp when available, not wall-clock time unless the SDK timestamp is unavailable.

`camera_info.json` must record:

- saved width and height
- native width and height
- saved intrinsics
- distortion values if available
- camera serial number and model
- ZED SDK version if available
- configured FPS and measured saved FPS summary
- timestamp source and timestamp unit
- depth unit, initially `meter`
- pose convention, initially `world_T_left_camera`
- image frame convention, depth frame convention, and camera optical frame notes
- crop/resize transform from native frame to saved frame
- extrinsics if available through the SDK

Depth exported by this program remains `float32` meters in the first version, so DAAAM should continue to use `--depth-scale 1.0`. If a future exporter writes millimeters, the metadata must explicitly say so and the generated DAAAM command must use `--depth-scale 1000`.

## Per-Frame Manifest Requirements

`manifest.jsonl` contains one JSON object for each saved frame and for each intentionally skipped frame when the reason is useful for auditing. Saved-frame entries include:

- `frame_id`
- `sequence_id`
- `timestamp`
- `timestamp_source`
- `rgb_path`
- `depth_path`
- `pose_valid`
- `pose_path`
- `pose_7d`
- `world_T_left_camera`
- `zed_tracking_state`
- `pose_confidence`
- `pose_reset`
- `pose_delta_translation_m`
- `pose_delta_rotation_deg`
- `depth_valid_ratio`
- `depth_min_m`
- `depth_median_m`
- `depth_p95_m`
- `depth_max_m`
- `depth_nan_ratio`
- `depth_zero_ratio`
- `rgb_mean_brightness`
- `rgb_dark_ratio`
- `rgb_saturation_ratio`
- `rgb_laplacian_sharpness`
- `exposure`
- `gain`
- `white_balance`
- `warnings`

If exposure, gain, or white-balance values cannot be read reliably from the SDK, fields are still present with `null` and a metadata note records that the values were unavailable.

Skipped-frame entries include:

- `frame_id: null`
- `grabbed_index`
- `timestamp`
- `skip_reason`
- `zed_tracking_state`
- `pose_confidence`
- any quality fields that were available before the skip decision

## Pose Reset And Sequence Splitting

The worker computes pose delta between adjacent valid saved poses.

Default thresholds:

- p95 translation delta target at 30 Hz: less than `0.05 m`
- warning threshold: single-frame translation delta greater than `0.20 m`
- hard reset threshold: single-frame translation delta greater than `0.25 m`

For 15 Hz capture, the thresholds remain conservative by default. The first version exposes threshold constants in the capture settings or quality settings, while the UI shows the defaults without advanced threshold editing.

When a warning threshold is crossed:

- the UI shows a persistent warning
- the frame `manifest.jsonl` entry includes `pose_reset=false` and a warning code
- the final `quality_report.json` includes the event

When the hard reset threshold is crossed, or when the ZED SDK reports a tracking reset, relocalization, or tracking lost/recovered boundary:

- the frame `manifest.jsonl` entry includes `pose_reset=true`
- the event is recorded with previous and current pose summaries
- the UI shows a stronger warning
- the capture either starts a new `sequence_id` within the same dataset or marks the boundary, depending on the selected policy

First-version policy: mark event boundaries in `manifest.jsonl` and `quality_report.json`, and increment `sequence_id` for subsequent saved frames. Physical subdirectories per sequence are explicitly out of scope for the first version.

## Quality Statistics

Depth quality is computed per saved frame:

- valid ratio
- min, median, p95, and max valid depth in meters
- NaN ratio
- zero ratio

RGB quality is computed per saved left RGB frame:

- mean brightness
- dark pixel ratio
- saturation ratio
- Laplacian sharpness

These are programmatic quality-control fields. The operator should not need to inspect every frame manually to identify unusable data.

## End-Of-Capture Quality Report

When capture stops, the program automatically runs a complete integrity and quality check and writes `quality_report.json`.

The report verifies:

- RGB, depth, pose, timestamp, and manifest counts agree
- frame ids are contiguous within the saved dataset
- RGB and depth resolutions match `camera_info.json`
- pose matrices are 4x4 and have bottom row `[0, 0, 0, 1]`
- quaternions are normalized within tolerance
- timestamps are monotonic
- large pose jumps and reset boundaries are summarized
- depth valid ratio distribution is summarized
- RGB brightness and sharpness distributions are summarized

If checks fail, the UI shows a failed quality status and points to `quality_report.json`. The dataset is still left on disk for inspection unless a file write failed before finalization.

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
- timestamp monotonicity
- large pose jump count
- reset boundary count
- final OK or FAILED status

Validation can run in a background task to avoid freezing the UI.

The existing validation script should be extended or complemented so it can read `manifest.jsonl` and `quality_report.json`. The older image sequence layout remains valid for DAAAM, but the new metadata becomes the primary audit trail.

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
- per-frame manifest writer test for saved and skipped frame entries
- pose delta and reset classification tests
- depth and RGB quality statistics tests
- quality report tests for count mismatch, non-monotonic timestamps, invalid pose matrix, non-normalized quaternion, and pose jump detection
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
- inspect `manifest.jsonl` and confirm per-frame tracking, pose, depth, and RGB quality fields exist
- inspect `quality_report.json` and confirm pose jump summary, count checks, timestamp checks, and quality distributions exist

## Implementation Notes

The first implementation should favor reliability over visual decoration. Keep the UI compact, readable, and operational. Avoid nested cards and oversized hero-style layouts. Use icon buttons where appropriate, but text labels are acceptable for critical commands such as start, pause, stop, and restart.
