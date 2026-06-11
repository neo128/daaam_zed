from __future__ import annotations

import math

import cv2
import numpy as np


def compute_output_geometry(
    native_w: int,
    native_h: int,
    K_native: np.ndarray,
    out_w: int,
    out_h: int,
    mode: str,
) -> tuple[int, int, tuple[int, int, int, int], np.ndarray]:
    if mode == "none" or out_w <= 0 or out_h <= 0:
        return native_w, native_h, (0, 0, native_w, native_h), K_native.copy()

    if mode == "crop_resize":
        src_aspect = native_w / native_h
        dst_aspect = out_w / out_h
        if src_aspect > dst_aspect:
            crop_h = native_h
            crop_w = int(round(native_h * dst_aspect))
            x0 = int(round((native_w - crop_w) / 2))
            y0 = 0
        else:
            crop_w = native_w
            crop_h = int(round(native_w / dst_aspect))
            x0 = 0
            y0 = int(round((native_h - crop_h) / 2))
    elif mode == "resize":
        x0, y0, crop_w, crop_h = 0, 0, native_w, native_h
    else:
        raise ValueError(f"Unknown resize mode: {mode}")

    sx = out_w / crop_w
    sy = out_h / crop_h
    K = K_native.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - x0) * sx
    K[1, 2] = (K[1, 2] - y0) * sy
    return out_w, out_h, (x0, y0, crop_w, crop_h), K


def apply_crop_resize(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    out_w: int,
    out_h: int,
    crop: tuple[int, int, int, int],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return bgr, depth_m
    x0, y0, crop_w, crop_h = crop
    bgr_crop = bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    depth_crop = depth_m[y0 : y0 + crop_h, x0 : x0 + crop_w]
    bgr_out = cv2.resize(bgr_crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    depth_out = cv2.resize(depth_crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return bgr_out, depth_out


def sanitize_depth(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    out = depth_m.astype(np.float32, copy=True)
    invalid = ~np.isfinite(out)
    if math.isfinite(min_depth):
        invalid |= out < min_depth
    if math.isfinite(max_depth):
        invalid |= out > max_depth
    out[invalid] = 0.0
    return out


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    out = q.astype(np.float64)
    norm = np.linalg.norm(out)
    if norm > 0:
        out = out / norm
    return out


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_matrix(quaternion)
    T[:3, 3] = translation.astype(np.float64)
    return T


def matrix_to_rows(matrix: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in matrix.reshape(4, 4)]
