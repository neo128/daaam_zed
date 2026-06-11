import numpy as np


def test_crop_resize_updates_intrinsics_for_center_crop():
    from daaam_zed.geometry import compute_output_geometry

    native_K = np.array(
        [
            [734.0, 0.0, 640.0],
            [0.0, 734.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    out_w, out_h, crop, adjusted = compute_output_geometry(
        native_w=1280,
        native_h=720,
        K_native=native_K,
        out_w=640,
        out_h=480,
        mode="crop_resize",
    )

    assert (out_w, out_h) == (640, 480)
    assert crop == (160, 0, 960, 720)
    np.testing.assert_allclose(adjusted[0, 0], 489.3333333333)
    np.testing.assert_allclose(adjusted[1, 1], 489.3333333333)
    np.testing.assert_allclose(adjusted[0, 2], 320.0)
    np.testing.assert_allclose(adjusted[1, 2], 240.0)


def test_sanitize_depth_zeros_nan_inf_and_out_of_range():
    from daaam_zed.geometry import sanitize_depth

    depth = np.array([[np.nan, np.inf, 0.03], [0.1, 2.0, 25.0]], dtype=np.float32)

    sanitized = sanitize_depth(depth, min_depth=0.05, max_depth=20.0)

    np.testing.assert_array_equal(
        sanitized,
        np.array([[0.0, 0.0, 0.0], [0.1, 2.0, 0.0]], dtype=np.float32),
    )


def test_pose_to_matrix_uses_translation_and_normalized_quaternion():
    from daaam_zed.geometry import pose_to_matrix

    translation = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    quaternion = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float64)

    matrix = pose_to_matrix(translation, quaternion)

    np.testing.assert_allclose(matrix[:3, :3], np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), atol=1e-8)
    np.testing.assert_allclose(matrix[:3, 3], translation)
    np.testing.assert_allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_apply_crop_resize_keeps_depth_nearest_neighbor_values():
    from daaam_zed.geometry import apply_crop_resize

    bgr = np.zeros((4, 8, 3), dtype=np.uint8)
    depth = np.arange(32, dtype=np.float32).reshape(4, 8)

    bgr_out, depth_out = apply_crop_resize(
        bgr=bgr,
        depth_m=depth,
        out_w=2,
        out_h=2,
        crop=(2, 0, 4, 4),
        mode="crop_resize",
    )

    assert bgr_out.shape == (2, 2, 3)
    assert depth_out.shape == (2, 2)
    assert set(depth_out.reshape(-1).tolist()).issubset(set(depth[:, 2:6].reshape(-1).tolist()))
