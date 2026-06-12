"""Depth image helpers for RGB-D hand-eye calibration."""
import os
from typing import Optional, Sequence

import cv2
import numpy as np

from .types import DepthSample


def depth_path_candidates(rgb_path: str, depth_dir: Optional[str] = None):
    directory = depth_dir if depth_dir else os.path.dirname(rgb_path)
    base = os.path.basename(rgb_path)
    root, ext = os.path.splitext(base)

    if "_Color" in base:
        prefix = base.split("_Color", 1)[0]
        depth_stem = base.replace("_Color", "_Depth")
        depth_root, _ = os.path.splitext(depth_stem)
    else:
        prefix = root.split("_", 1)[0]
        depth_root = f"{root}_Depth"

    suffix_candidates = [
        f"{depth_root}.png",
        f"{depth_root}.jpg",
        f"{depth_root}.jpeg",
        f"{depth_root}.raw",
    ]
    if prefix:
        suffix_candidates.append(f"{prefix}.raw")

    seen = set()
    candidates = []
    for name in suffix_candidates:
        path = os.path.join(directory, name)
        if path not in seen:
            seen.add(path)
            candidates.append(path)
    return candidates


def match_depth_path(rgb_path: str, depth_dir: Optional[str] = None) -> str:
    candidates = depth_path_candidates(rgb_path, depth_dir)
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_depth_image(path: str, image_shape: Optional[Sequence[int]] = None):
    if not os.path.exists(path):
        return None
    if os.path.splitext(path)[1].lower() == ".raw":
        if image_shape is None or len(image_shape) < 2:
            return None
        height, width = int(image_shape[0]), int(image_shape[1])
        depth = np.fromfile(path, dtype=np.uint16)
        if depth.size != height * width:
            return None
        return depth.reshape((height, width))

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def load_depth_png(path: str, image_shape: Optional[Sequence[int]] = None):
    return load_depth_image(path, image_shape=image_shape)


def _median_depth(depth, x, y, window_size, depth_scale, min_depth, max_depth):
    radius = max(0, int(window_size) // 2)
    h, w = depth.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float64) * depth_scale
    valid = patch[np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def sample_depth_at_corners(depth_image, corners_2d, corner_ids, camera_matrix,
                            depth_scale=0.001, min_depth=0.05, max_depth=5.0,
                            window_size=3):
    depth = np.asarray(depth_image)
    corners = np.asarray(corners_2d, dtype=np.float64)
    ids = np.asarray(corner_ids, dtype=int)
    K = np.asarray(camera_matrix, dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth.shape[:2]

    points = []
    used_ids = []
    used_corners = []
    for idx, (u, v) in enumerate(corners.T):
        x = int(round(u))
        y = int(round(v))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        z = _median_depth(depth, x, y, window_size, depth_scale, min_depth, max_depth)
        if z is None:
            continue
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        points.append([X, Y, z])
        used_ids.append(ids[idx])
        used_corners.append([u, v])

    if points:
        points_camera = np.asarray(points, dtype=np.float64).T
        used_corners_2d = np.asarray(used_corners, dtype=np.float64).T
    else:
        points_camera = np.zeros((3, 0), dtype=np.float64)
        used_corners_2d = np.zeros((2, 0), dtype=np.float64)

    return DepthSample(
        points_camera=points_camera,
        corner_ids=np.asarray(used_ids, dtype=int),
        corners_2d=used_corners_2d,
    )
