import numpy as np


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False
        self.svo_path = None
        self.info = {
            "native_width": 4,
            "native_height": 3,
            "intrinsics": [[1.0, 0.0, 2.0], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]],
            "distortion": [],
            "zed": {"serial_number": 42, "camera_model": "FAKE"},
        }

    def open(self):
        return self.info

    def grab(self):
        if not self.frames:
            return None
        return self.frames.pop(0)

    def close(self):
        self.closed = True

    def enable_svo_recording(self, path):
        self.svo_path = path


def _frame(timestamp, x=0.0):
    from daaam_zed.models import CameraFrame, Pose7D

    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return CameraFrame(
        left_bgr=np.full((3, 4, 3), 100, dtype=np.uint8),
        right_bgr=np.full((3, 4, 3), 120, dtype=np.uint8),
        depth_m=np.ones((3, 4), dtype=np.float32),
        world_T_left_camera=pose,
        pose_7d=Pose7D(x, 0, 0, 0, 0, 0, 1),
        timestamp=timestamp,
        timestamp_source="zed_image",
        tracking_state="OK",
        pose_confidence=100.0,
    )


def test_run_capture_writes_dataset_and_marks_pose_reset(tmp_path):
    from daaam_zed.models import CaptureSettings
    from daaam_zed.validation import validate_dataset
    from daaam_zed.zed_capture import run_capture

    camera = FakeCamera([_frame(1.0, x=0.0), _frame(2.0, x=0.30)])

    summary = run_capture(
        output_dir=tmp_path / "scene",
        settings=CaptureSettings(warmup_frames=0, max_frames=2, output_width=4, output_height=3, resize_mode="none"),
        overwrite=True,
        camera=camera,
    )

    assert summary.frames_saved == 2
    assert summary.pose_reset_count == 1
    assert camera.closed is True

    manifest_lines = (tmp_path / "scene" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 2
    assert '"sequence_id": 0' in manifest_lines[0]
    assert '"sequence_id": 1' in manifest_lines[1]
    assert '"pose_reset": true' in manifest_lines[1]

    result = validate_dataset(tmp_path / "scene")
    assert result.ok is True
    assert result.pose_reset_count == 1


def test_run_capture_skips_bad_tracking_without_number_gap(tmp_path):
    from daaam_zed.models import CaptureSettings
    from daaam_zed.zed_capture import run_capture

    bad = _frame(1.0)
    bad.tracking_state = "SEARCHING"
    good = _frame(2.0)
    camera = FakeCamera([bad, good])

    summary = run_capture(
        output_dir=tmp_path / "scene",
        settings=CaptureSettings(warmup_frames=0, max_frames=1, output_width=4, output_height=3, resize_mode="none"),
        overwrite=True,
        camera=camera,
    )

    assert summary.frames_saved == 1
    assert (tmp_path / "scene" / "rgb" / "000000.png").exists()
    assert not (tmp_path / "scene" / "rgb" / "000001.png").exists()
    manifest_text = (tmp_path / "scene" / "manifest.jsonl").read_text(encoding="utf-8")
    assert '"skip_reason": "BAD_TRACKING"' in manifest_text


def test_run_capture_enables_optional_svo_recording(tmp_path):
    from daaam_zed.models import CaptureSettings
    from daaam_zed.zed_capture import run_capture

    camera = FakeCamera([_frame(1.0)])

    run_capture(
        output_dir=tmp_path / "scene",
        settings=CaptureSettings(
            warmup_frames=0,
            max_frames=1,
            output_width=4,
            output_height=3,
            resize_mode="none",
            enable_svo_recording=True,
        ),
        overwrite=True,
        camera=camera,
    )

    assert camera.svo_path == tmp_path / "scene" / "raw.svo"
