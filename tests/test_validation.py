import numpy as np


def _write_one_frame_dataset(root, timestamp=1.0):
    from daaam_zed.dataset_writer import DatasetWriter
    from daaam_zed.models import CaptureSettings, FrameQuality, Pose7D

    writer = DatasetWriter(
        output_dir=root,
        settings=CaptureSettings(),
        camera_info={"width": 4, "height": 3, "intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        overwrite=True,
    )
    writer.write_frame(
        bgr=np.full((3, 4, 3), 80, dtype=np.uint8),
        depth_m=np.ones((3, 4), dtype=np.float32),
        world_T_left_camera=np.eye(4),
        pose_7d=Pose7D(0, 0, 0, 0, 0, 0, 1),
        timestamp=timestamp,
        timestamp_source="zed_image",
        zed_tracking_state="OK",
        pose_confidence=100.0,
        quality=FrameQuality(depth_valid_ratio=1.0),
        sequence_id=0,
    )
    writer.close(frames_grabbed=1)


def test_validate_dataset_accepts_writer_output(tmp_path):
    from daaam_zed.validation import validate_dataset

    root = tmp_path / "scene"
    _write_one_frame_dataset(root)

    result = validate_dataset(root, depth_scale=1.0, fps=15)

    assert result.ok is True
    assert result.rgb_count == 1
    assert result.depth_count == 1
    assert result.pose_count == 1
    assert result.timestamp_count == 1
    assert result.manifest_entries == 1
    assert result.errors == []


def test_validate_dataset_reports_count_mismatch(tmp_path):
    from daaam_zed.validation import validate_dataset

    root = tmp_path / "scene"
    _write_one_frame_dataset(root)
    (root / "depth" / "000000.npy").unlink()

    result = validate_dataset(root)

    assert result.ok is False
    assert any("RGB/depth count mismatch" in error for error in result.errors)


def test_validate_dataset_reports_non_monotonic_timestamps(tmp_path):
    from daaam_zed.validation import validate_dataset

    root = tmp_path / "scene"
    _write_one_frame_dataset(root)
    with (root / "timestamps.txt").open("a", encoding="utf-8") as handle:
        handle.write("1 0.500000000 OK 100.0\n")

    result = validate_dataset(root)

    assert result.ok is False
    assert any("Timestamps are not strictly increasing" in error for error in result.errors)


def test_validate_dataset_reports_invalid_pose_matrix(tmp_path):
    from daaam_zed.validation import validate_dataset

    root = tmp_path / "scene"
    _write_one_frame_dataset(root)
    pose_file = root / "pose" / "poses.txt"
    pose_file.write_text(" ".join(["0"] * 16) + "\n", encoding="utf-8")

    result = validate_dataset(root)

    assert result.ok is False
    assert any("bottom row" in error for error in result.errors)
