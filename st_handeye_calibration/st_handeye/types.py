"""Data classes for hand-eye calibration."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import numpy as np


class CalibrationSetup(str, Enum):
    EYE_IN_HAND = "eye-in-hand"
    EYE_TO_HAND = "eye-to-hand"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        for setup in cls:
            if setup.value == value:
                return setup
        raise ValueError(f"Unknown calibration setup: {value}")


@dataclass
class DetectionResult:
    corners_2d: np.ndarray      # (2, N) detected 2D corners
    corner_ids: np.ndarray      # (N,) corner IDs
    success: bool
    image_path: str
    num_corners: int
    num_markers: int = 0
    pnp_success: bool = False
    pnp_reprojection_error: float = float("inf")
    num_depth_corners: int = 0


@dataclass
class DepthSample:
    points_camera: np.ndarray   # (3, M) 3D points back-projected from depth
    corner_ids: np.ndarray      # (M,) board corner IDs matching points_camera
    corners_2d: np.ndarray      # (2, M) image locations used for sampling


@dataclass
class InitialGuess:
    T_C2F: np.ndarray            # Camera-to-Flange: P_F = T_C2F @ P_C
    T_O2W: np.ndarray            # Object-to-World: P_W = T_O2W @ P_O
    T_O2C_list: List[np.ndarray] # Object-to-Camera per image (derived)
    T_O2C_pnp: List[np.ndarray] = None  # from solvePnP
    setup: CalibrationSetup = CalibrationSetup.EYE_IN_HAND
    T_O2C_measured: Optional[List[np.ndarray]] = None  # depth-first measurement pose per image


@dataclass
class CalibrationResult:
    T_C2F: Optional[np.ndarray]  # Camera-to-Flange for eye-in-hand
    T_O2W: Optional[np.ndarray]  # Object-to-World for eye-in-hand
    T_O2C_opt: List[np.ndarray]  # Optimized Object-to-Camera
    T_O2C_derived: List[np.ndarray]  # Derived from constraint
    reprojection_error: Dict
    pose_error: Dict
    num_images: int
    num_images_used: int
    filtered_images: List[int]
    setup: CalibrationSetup = CalibrationSetup.EYE_IN_HAND
    T_C2W: Optional[np.ndarray] = None  # Camera-to-World for eye-to-hand
    T_O2F: Optional[np.ndarray] = None  # Object-to-Flange for eye-to-hand
    depth_used: bool = False
    frame_errors: Optional[List[Dict]] = None
    optimizer_backend: str = "quaternion_manifold"
    comparison_metrics: Optional[Dict] = None
    base_consistency_error: Optional[Dict] = None


@dataclass
class OptimizationParams:
    projection_weight: float = 1.0
    depth_weight: float = 0.01
    pose_weight: float = 100.0
    pose_trans_scale: float = 1.0
    pose_rot_scale: float = 1.0
    projection_sigma_px: Optional[float] = None
    depth_sigma_m: Optional[float] = None
    pose_trans_sigma_m: Optional[float] = None
    pose_rot_sigma_deg: Optional[float] = None
    num_iterations: int = 500
    robust_kernel: str = "huber"
    kernel_delta: float = 1.0
    optimize_frame_poses: bool = False
    optimizer_backend: str = "quaternion_manifold"
    compare_with_legacy_backend: bool = False


@dataclass
class BoardConfig:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    aruco_dict: str
