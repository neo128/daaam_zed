import numpy as np


def test_depth_quality_stats_use_positive_finite_depths():
    from daaam_zed.quality import compute_depth_quality

    depth = np.array([[0.0, 1.0, 2.0, np.nan], [np.inf, 0.0, 5.0, 10.0]], dtype=np.float32)

    stats = compute_depth_quality(depth)

    assert stats.depth_valid_ratio == 0.5
    assert stats.depth_nan_ratio == 0.25
    assert stats.depth_zero_ratio == 0.25
    assert stats.depth_min_m == 1.0
    assert stats.depth_median_m == 3.5
    assert stats.depth_max_m == 10.0
    assert stats.depth_p95_m is not None
    assert 9.0 < stats.depth_p95_m <= 10.0


def test_rgb_quality_stats_measure_dark_saturation_and_sharpness():
    from daaam_zed.quality import compute_rgb_quality

    dark = np.full((8, 8, 3), 10, dtype=np.uint8)
    dark_stats = compute_rgb_quality(dark)

    assert dark_stats.rgb_mean_brightness == 10.0
    assert dark_stats.rgb_dark_ratio == 1.0
    assert dark_stats.rgb_saturation_ratio == 0.0
    assert dark_stats.rgb_laplacian_sharpness == 0.0

    checker = np.indices((8, 8)).sum(axis=0) % 2
    checker_bgr = np.dstack([checker * 255, np.zeros_like(checker), np.zeros_like(checker)]).astype(np.uint8)
    checker_stats = compute_rgb_quality(checker_bgr)

    assert checker_stats.rgb_saturation_ratio == 0.5
    assert checker_stats.rgb_laplacian_sharpness > 0.0


def test_pose_delta_classifies_warning_and_reset_thresholds():
    from daaam_zed.quality import classify_pose_delta

    previous = np.eye(4, dtype=np.float64)
    warning_pose = np.eye(4, dtype=np.float64)
    warning_pose[0, 3] = 0.21
    reset_pose = np.eye(4, dtype=np.float64)
    reset_pose[0, 3] = 0.26

    warning = classify_pose_delta(previous, warning_pose)
    reset = classify_pose_delta(previous, reset_pose)

    assert warning.pose_delta_translation_m == 0.21
    assert warning.pose_reset is False
    assert "POSE_DELTA_WARNING" in warning.warnings
    assert reset.pose_delta_translation_m == 0.26
    assert reset.pose_reset is True
    assert "POSE_RESET" in reset.warnings


def test_quality_report_summarizes_counts_and_pose_jumps():
    from daaam_zed.models import FrameQuality
    from daaam_zed.quality import build_quality_report

    qualities = [
        FrameQuality(depth_valid_ratio=0.8, rgb_mean_brightness=20.0),
        FrameQuality(depth_valid_ratio=0.4, rgb_mean_brightness=50.0, pose_delta_translation_m=0.3, pose_reset=True, warnings=["POSE_RESET"]),
    ]

    report = build_quality_report(
        frame_count=2,
        timestamp_count=2,
        pose_count=2,
        manifest_count=2,
        qualities=qualities,
        validation_errors=[],
    )

    assert report["counts"]["frames"] == 2
    assert report["counts"]["manifest_entries"] == 2
    assert report["pose"]["reset_count"] == 1
    assert report["pose"]["translation_delta_p95_m"] == 0.3
    assert report["depth"]["valid_ratio_min"] == 0.4
    assert report["depth"]["valid_ratio_median"] == 0.6000000000000001
    assert report["rgb"]["mean_brightness_median"] == 35.0
    assert report["ok"] is True


def test_merge_quality_preserves_depth_and_rgb_fields():
    from daaam_zed.quality import compute_depth_quality, compute_rgb_quality, merge_quality

    depth_stats = compute_depth_quality(np.ones((2, 2), dtype=np.float32))
    rgb_stats = compute_rgb_quality(np.full((2, 2, 3), 50, dtype=np.uint8))

    merged = merge_quality(depth_stats, rgb_stats)

    assert merged.depth_valid_ratio == 1.0
    assert merged.depth_median_m == 1.0
    assert merged.rgb_mean_brightness == 50.0
    assert merged.rgb_dark_ratio == 0.0
