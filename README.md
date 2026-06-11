# DAAAM ZED Offline Capture

Pure offline ZED/ZED Mini capture for DAAAM. No ROS, no topic forwarding, no rosbag.

The collector writes the exact folder layout expected by DAAAM's `ImageSequenceDataset`:

```text
my_zed_scene/
├── rgb/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── depth/
│   ├── 000000.npy          # float32 depth, meters
│   ├── 000001.npy
│   └── ...
├── pose/
│   ├── poses.txt           # one row-major 4x4 world_T_left_camera matrix per frame
│   └── poses_7d.txt        # debug: x y z qx qy qz qw
├── camera_info.json        # width, height, 3x3 intrinsics, distortion
├── timestamps.txt          # provenance/debug
├── manifest.json           # capture settings and DAAAM command hint
├── manifest.jsonl          # one per-frame audit record
└── quality_report.json     # end-of-capture integrity and quality summary
```

## Install

Install the ZED SDK and its Python API first. The capture script depends on `pyzed.sl`, not just USB camera visibility.

On Linux, Stereolabs usually installs the helper script here:

```bash
cd /usr/local/zed
python3 get_python_api.py
```

Then install normal Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

For the desktop UI, use the project virtual environment created with Python 3.12:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, run the local diagnostic first:

```bash
python3 scripts/check_zed_environment.py
```

If macOS shows a `ZED-M Hid Device` over USB but `pyzed.sl` is missing, this script cannot capture full ZED depth/tracking/pose on that host. Use a supported ZED SDK host such as Ubuntu, Windows, or Jetson for the full DAAAM dataset path below.

## Capture from ZED Mini / ZED M

```bash
python3 scripts/capture_zed_to_daaam_dataset.py \
  --output-dir $HOME/datasets/zedm_daaam_room01 \
  --resolution HD720 \
  --fps 15 \
  --depth-mode ULTRA \
  --width 640 \
  --height 480 \
  --resize-mode crop_resize \
  --max-frames 900 \
  --overwrite
```

This records about 60 seconds at 15 FPS if `--max-frames 900` is used.

Stop anytime with `Ctrl+C`; the script will close files and write `manifest.json`.

The CLI now writes the extra audit files `manifest.jsonl` and `quality_report.json`.
Each saved frame has one aligned `frame_id` across RGB, depth, pose, timestamps, and per-frame metadata. If a required modality is missing, the frame is skipped rather than leaving mismatched numbering.

Add `--record-svo` when you also want a replayable raw ZED recording:

```bash
python scripts/capture_zed_to_daaam_dataset.py \
  --output-dir $HOME/datasets/zedm_daaam_room01 \
  --max-frames 900 \
  --record-svo \
  --overwrite
```

This writes `raw.svo` inside the output directory when supported by the active ZED SDK configuration.

## Capture with the desktop UI

```bash
source .venv/bin/activate
python scripts/zed_capture_ui.py
```

If your current shell has not picked up the `video` and `zed` groups yet, launch with:

```bash
sg zed -c 'sg video -c ".venv/bin/python scripts/zed_capture_ui.py"'
```

The UI provides:

- left and right camera preview
- output directory and capture settings
- open device, start capture, pause saving, resume saving, stop capture, restart device, close device
- live FPS, saved frame count, tracking state, pose, and quality status
- dataset validation from the selected output directory
- optional raw SVO recording through the `Record Raw SVO` checkbox

Pause keeps the camera preview running and stops only file saving. Resume continues writing into the same dataset directory.

## Validate the dataset

```bash
python3 scripts/validate_daaam_image_sequence.py \
  $HOME/datasets/zedm_daaam_room01 \
  --depth-scale 1.0 \
  --fps 15
```

## Run DAAAM offline

From the DAAAM repo root:

```bash
python scripts/run_pipeline.py $HOME/datasets/zedm_daaam_room01 \
  --dataset-type ImageSequenceDataset \
  --depth-scale 1.0 \
  --fps 15 \
  --target-fps 15 \
  --output-dir output/zedm_room01 \
  --hydra-config-path /path/to/daaam_ros/config/hydra_config/coda_dataset_khronos.yaml \
  --sam-model fastsam/FastSAM-s.pt \
  --depth-lb 0.05 \
  --depth-ub 20.0
```

Notes:

- Depth is saved as `.npy` float32 meters, so DAAAM uses `--depth-scale 1.0`.
- `camera_info.json` records depth unit, timestamp source/unit, configured FPS, camera metadata, coordinate notes, and crop/resize geometry.
- `timestamps.txt` stores real per-frame timestamps aligned to saved frame ids.
- `manifest.jsonl` records per-frame paths, pose validity, ZED tracking state, pose confidence, pose reset flags, depth quality, RGB quality, and exposure/gain/white-balance when available.
- `quality_report.json` summarizes count checks, timestamp monotonicity, pose reset boundaries, depth valid ratio, and RGB quality distributions.
- The ZED SDK coordinate system is set to `IMAGE` by default: x right, y down, z forward. This matches the OpenCV-style projection used with `fx/fy/cx/cy` and depth images.
- The pose written to `pose/poses.txt` is `world_T_left_camera`, one 4x4 matrix per RGB/depth frame. DAAAM converts this into `[x, y, z, qx, qy, qz, qw]` internally.
- `--resize-mode crop_resize` center-crops then resizes, preserving aspect ratio. It also updates the saved intrinsics in `camera_info.json`.
- The capture core computes adjacent pose deltas. Single-frame translation deltas over 0.20 m produce warnings; deltas over 0.25 m are marked as pose reset events and start a new `sequence_id` in `manifest.jsonl`.
- If ZED tracking is unstable, frames are dropped by default. Disable this only for debugging:

```bash
python3 scripts/capture_zed_to_daaam_dataset.py \
  --output-dir /tmp/zed_debug \
  --no-drop-bad-tracking
```

## Recommended first test

Use a short, slow, static-object indoor sequence:

```bash
python3 scripts/capture_zed_to_daaam_dataset.py \
  --output-dir $HOME/datasets/zedm_daaam_smoke01 \
  --fps 10 \
  --max-frames 300 \
  --show-preview \
  --overwrite
```

Then validate:

```bash
python3 scripts/validate_daaam_image_sequence.py $HOME/datasets/zedm_daaam_smoke01 --fps 10
```

## Developer verification

```bash
PYTHONPATH=src QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests -v
PYTHONPATH=src .venv/bin/python scripts/capture_zed_to_daaam_dataset.py --help
PYTHONPATH=src .venv/bin/python scripts/validate_daaam_image_sequence.py --help
PYTHONPATH=src QT_QPA_PLATFORM=offscreen .venv/bin/python -c "from daaam_zed.ui import MainWindow; print(MainWindow)"
```
