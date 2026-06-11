import json

import cv2
import numpy as np


def test_dataset_writer_commits_aligned_frame_metadata(tmp_path):
    from daaam_zed.dataset_writer import DatasetWriter
    from daaam_zed.models import CaptureSettings, FrameQuality, Pose7D

    writer = DatasetWriter(
        output_dir=tmp_path / "scene",
        settings=CaptureSettings(fps=15),
        camera_info={
            "width": 6,
            "height": 4,
            "intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "zed": {"serial_number": 123},
        },
        overwrite=True,
    )

    bgr = np.full((4, 6, 3), 127, dtype=np.uint8)
    depth = np.full((4, 6), 2.5, dtype=np.float32)
    pose = np.eye(4, dtype=np.float64)
    quality = FrameQuality(depth_valid_ratio=1.0, rgb_mean_brightness=127.0)

    frame_id = writer.write_frame(
        bgr=bgr,
        depth_m=depth,
        world_T_left_camera=pose,
        pose_7d=Pose7D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        timestamp=123.456,
        timestamp_source="zed_image",
        zed_tracking_state="OK",
        pose_confidence=98.0,
        quality=quality,
        sequence_id=0,
        exposure=12.0,
        gain=4.0,
        white_balance=4600.0,
    )
    writer.write_skipped_frame(
        grabbed_index=2,
        timestamp=124.0,
        skip_reason="BAD_TRACKING",
        zed_tracking_state="SEARCHING",
        pose_confidence=10.0,
    )
    writer.close(frames_grabbed=2, extra={"test": True})

    root = tmp_path / "scene"
    assert frame_id == 0
    assert cv2.imread(str(root / "rgb" / "000000.png")) is not None
    np.testing.assert_array_equal(np.load(root / "depth" / "000000.npy"), depth)

    pose_lines = (root / "pose" / "poses.txt").read_text(encoding="utf-8").splitlines()
    pose_7d_lines = (root / "pose" / "poses_7d.txt").read_text(encoding="utf-8").splitlines()
    timestamp_lines = (root / "timestamps.txt").read_text(encoding="utf-8").splitlines()
    assert len(pose_lines) == 1
    assert len(pose_7d_lines) == 1
    assert timestamp_lines[1].startswith("0 123.456000000 OK 98.0")

    manifest_entries = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_entries[0]["frame_id"] == 0
    assert manifest_entries[0]["sequence_id"] == 0
    assert manifest_entries[0]["rgb_path"] == "rgb/000000.png"
    assert manifest_entries[0]["depth_valid_ratio"] == 1.0
    assert manifest_entries[0]["exposure"] == 12.0
    assert manifest_entries[1]["frame_id"] is None
    assert manifest_entries[1]["skip_reason"] == "BAD_TRACKING"

    camera_info = json.loads((root / "camera_info.json").read_text(encoding="utf-8"))
    assert camera_info["timestamp"]["source"] == "zed_image"
    assert camera_info["timestamp"]["unit"] == "second"
    assert camera_info["notes"]["depth_unit"] == "meter"
    assert camera_info["capture"]["configured_fps"] == 15

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames_saved"] == 1
    assert manifest["frames_grabbed_after_open"] == 2
    assert manifest["test"] is True

    quality_report = json.loads((root / "quality_report.json").read_text(encoding="utf-8"))
    assert quality_report["ok"] is True
    assert quality_report["counts"]["frames"] == 1
    assert quality_report["counts"]["manifest_entries"] == 2
