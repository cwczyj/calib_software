import csv
import os
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from st_handeye.depth import load_depth_image, match_depth_path, sample_depth_at_corners
from st_handeye.io import load_camera_params_yaml, save_calibration_yaml
from st_handeye.board import CharucoBoard
from st_handeye.optimizer import (
    GraphOptimizer, compute_initial_guess, derive_object_to_camera,
    scipy_loss_from_kernel,
)
from st_handeye.evaluation import compute_base_frame_corner_consistency, compute_reprojection_error
from st_handeye.calibrator import HandEyeCalibrator
from st_handeye.types import (
    BoardConfig, CalibrationSetup, DepthSample, DetectionResult, InitialGuess, OptimizationParams,
)


def make_transform(tx=0.0, ty=0.0, tz=0.0, rotvec=None):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    if rotvec is not None:
        T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return T


def test_charuco_board_accepts_ui_dictionary_options():
    for aruco_dict in ["DICT_5X5_50", "DICT_7X7_1000"]:
        board = CharucoBoard(BoardConfig(
            squares_x=11,
            squares_y=8,
            square_length=0.014,
            marker_length=0.010,
            aruco_dict=aruco_dict,
        ))

        assert board.num_corners == 70


def test_build_frame_errors_lists_unparticipated_images_without_metrics():
    calibrator = HandEyeCalibrator.__new__(HandEyeCalibrator)
    detection = SimpleNamespace(
        image_path="/data/001_Color.png",
        num_corners=42,
    )
    metrics = {
        "reprojection_error": {
            "per_image": [0.21],
            "per_image_stats": [{"count": 42, "mean": 0.21, "rms": 0.27, "max": 0.8}],
            "per_image_optimized": [0.18],
            "per_image_optimized_stats": [{"mean": 0.18, "rms": 0.22, "max": 0.62}],
        },
        "pose_error": {
            "per_image": [{"translation": 0.0012, "rotation": 0.3}],
        },
    }

    rows = calibrator._build_frame_errors(
        [detection],
        [0],
        metrics,
        ["/data/001_Color.png", "/data/002_Color.png"],
    )

    assert len(rows) == 2
    assert rows[0]["used"] is True
    assert rows[0]["reprojection_mean_px"] == 0.21
    assert rows[1]["index"] == 1
    assert rows[1]["image_path"] == "/data/002_Color.png"
    assert rows[1]["used"] is False
    assert rows[1]["reprojection_mean_px"] is None
    assert rows[1]["translation_error"] is None


def test_base_frame_corner_consistency_uses_measured_object_to_camera_for_eye_in_hand():
    pattern = np.array([
        [0.0, 0.08, 0.08, 0.0],
        [0.0, 0.0, 0.06, 0.06],
        [0.0, 0.0, 0.0, 0.0],
    ])
    primary = make_transform(0.04, -0.02, 0.10, [0.02, -0.01, 0.03])
    target = make_transform(0.30, 0.05, 0.12, [-0.03, 0.02, 0.04])
    poses = [
        make_transform(0.00, 0.00, 0.00, [0.00, 0.00, 0.00]),
        make_transform(0.05, 0.01, -0.02, [0.18, 0.10, -0.06]),
        make_transform(-0.04, 0.06, 0.01, [-0.16, 0.21, 0.09]),
    ]
    detections = [make_detection(pattern.shape[1]) for _ in poses]
    measured = [
        derive_object_to_camera(CalibrationSetup.EYE_IN_HAND, pose, primary, target)
        for pose in poses
    ]

    consistency = compute_base_frame_corner_consistency(
        pattern, detections, poses, primary, measured,
        setup=CalibrationSetup.EYE_IN_HAND,
    )

    assert consistency["count"] == pattern.shape[1] * len(poses)
    assert consistency["rms"] < 1e-9
    assert consistency["per_image"][0]["rms"] < 1e-9


def test_base_frame_corner_consistency_prefers_depth_points_when_available():
    pattern = np.array([
        [0.0, 0.08, 0.08, 0.0],
        [0.0, 0.0, 0.06, 0.06],
        [0.0, 0.0, 0.0, 0.0],
    ])
    primary = make_transform(0.04, -0.02, 0.10, [0.02, -0.01, 0.03])
    target = make_transform(0.30, 0.05, 0.12, [-0.03, 0.02, 0.04])
    poses = [
        make_transform(0.00, 0.00, 0.00, [0.00, 0.00, 0.00]),
        make_transform(0.05, 0.01, -0.02, [0.18, 0.10, -0.06]),
        make_transform(-0.04, 0.06, 0.01, [-0.16, 0.21, 0.09]),
    ]
    detections = [make_detection(pattern.shape[1]) for _ in poses]
    true_measured = [
        derive_object_to_camera(CalibrationSetup.EYE_IN_HAND, pose, primary, target)
        for pose in poses
    ]
    bad_measured = [T @ make_transform(0.03, -0.01, 0.02) for T in true_measured]
    depth_samples = [make_depth_sample(pattern, T) for T in true_measured]

    consistency = compute_base_frame_corner_consistency(
        pattern, detections, poses, primary, bad_measured,
        depth_samples=depth_samples,
        setup=CalibrationSetup.EYE_IN_HAND,
    )

    assert consistency["rms"] < 1e-9


def test_base_frame_corner_consistency_computes_flange_consistency_for_eye_to_hand():
    pattern = np.array([
        [0.0, 0.08, 0.08, 0.0],
        [0.0, 0.0, 0.06, 0.06],
        [0.0, 0.0, 0.0, 0.0],
    ])
    primary = make_transform(0.10, -0.03, 0.22, [0.03, 0.01, -0.02])
    target = make_transform(0.04, 0.02, 0.16, [-0.02, 0.04, 0.01])
    poses = [
        make_transform(0.00, 0.00, 0.00, [0.00, 0.00, 0.00]),
        make_transform(0.05, 0.01, -0.02, [0.18, 0.10, -0.06]),
        make_transform(-0.04, 0.06, 0.01, [-0.16, 0.21, 0.09]),
    ]
    detections = [make_detection(pattern.shape[1]) for _ in poses]
    measured = [
        derive_object_to_camera(CalibrationSetup.EYE_TO_HAND, pose, primary, target)
        for pose in poses
    ]

    consistency = compute_base_frame_corner_consistency(
        pattern, detections, poses, primary, measured,
        setup=CalibrationSetup.EYE_TO_HAND,
    )

    assert consistency["count"] == pattern.shape[1] * len(poses)
    assert consistency["rms"] < 1e-9
    assert consistency["per_image"][0]["rms"] < 1e-9


class PnpCamera:
    def __init__(self, poses):
        self.poses = list(poses)
        self.i = 0

    def solve_pnp(self, pts3d, pts2d, pnp_method="iterative", assume_undistorted=False):
        T = self.poses[self.i]
        self.i += 1
        return T.copy(), True


def make_detection(num_points):
    return DetectionResult(
        corners_2d=np.zeros((2, num_points)),
        corner_ids=np.arange(num_points),
        success=True,
        image_path="",
        num_corners=num_points,
    )


def make_depth_sample(pattern, transform, corner_ids=None):
    if corner_ids is None:
        corner_ids = np.arange(pattern.shape[1])
    corner_ids = np.asarray(corner_ids, dtype=int)
    points_object = pattern[:, corner_ids]
    points_camera = transform[:3, :3] @ points_object + transform[:3, 3:4]
    return DepthSample(
        points_camera=points_camera,
        corner_ids=corner_ids,
        corners_2d=np.zeros((2, corner_ids.shape[0]), dtype=float),
    )


def make_quality_detection(num_corners, num_markers, pnp_reprojection):
    det = make_detection(num_corners)
    det.num_markers = num_markers
    det.pnp_success = True
    det.pnp_reprojection_error = pnp_reprojection
    return det


def transform_error_norm(A, B):
    delta = np.linalg.inv(A) @ B
    return np.linalg.norm(delta[:3, 3]) + np.linalg.norm(delta[:3, :3] - np.eye(3))


def pnp_constraint_error(setup, poses, pnp_poses, primary, target):
    total = 0.0
    for pose, measured in zip(poses, pnp_poses):
        predicted = derive_object_to_camera(setup, pose, primary, target)
        total += transform_error_norm(measured, predicted)
    return total


def test_eye_in_hand_constraint_derives_object_to_camera():
    T_C2F = make_transform(0.10, 0.0, 0.0)
    T_F2W = make_transform(0.0, 0.20, 0.0)
    T_O2W = make_transform(0.0, 0.0, 0.30)

    actual = derive_object_to_camera(
        CalibrationSetup.EYE_IN_HAND,
        robot_pose_w2f=T_F2W,
        primary_transform=T_C2F,
        target_transform=T_O2W,
    )

    expected = np.linalg.inv(T_C2F) @ np.linalg.inv(T_F2W) @ T_O2W
    np.testing.assert_allclose(actual, expected)


def test_eye_to_hand_constraint_derives_object_to_camera():
    T_C2W = make_transform(0.10, 0.0, 0.0)
    T_F2W = make_transform(0.0, 0.20, 0.0)
    T_O2F = make_transform(0.0, 0.0, 0.30)

    actual = derive_object_to_camera(
        CalibrationSetup.EYE_TO_HAND,
        robot_pose_w2f=T_F2W,
        primary_transform=T_C2W,
        target_transform=T_O2F,
    )

    expected = np.linalg.inv(T_C2W) @ T_F2W @ T_O2F
    np.testing.assert_allclose(actual, expected)


def test_depth_matching_uses_same_frame_number_and_depth_suffix():
    path = match_depth_path("/tmp/session/001_Color.png")
    assert path == "/tmp/session/001_Depth.png"

    path = match_depth_path("/tmp/session/001_Color.png", depth_dir="/tmp/depth")
    assert path == "/tmp/depth/001_Depth.png"


def test_depth_matching_falls_back_to_same_frame_raw_file():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "001.raw")
        open(raw_path, "wb").close()

        path = match_depth_path(os.path.join(tmp, "001_Color.png"))

        assert path == raw_path


def test_load_depth_image_reads_raw_using_rgb_shape():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "001.raw")
        np.array([[0, 1000], [1500, 2000]], dtype=np.uint16).tofile(raw_path)

        depth = load_depth_image(raw_path, image_shape=(2, 2))

        assert depth.shape == (2, 2)
        assert depth.dtype == np.uint16
        np.testing.assert_array_equal(depth, np.array([[0, 1000], [1500, 2000]], dtype=np.uint16))


def test_sample_depth_at_corners_filters_invalid_depth_and_backprojects_mm_to_m():
    depth = np.zeros((4, 5), dtype=np.uint16)
    depth[1, 2] = 1000
    depth[2, 3] = 500
    K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    corners = np.array([[2.0, 1.0, 3.0], [1.0, 1.0, 2.0]])
    ids = np.array([4, 5, 6])

    sampled = sample_depth_at_corners(
        depth, corners, ids, K, depth_scale=0.001, min_depth=0.1, max_depth=2.0,
        window_size=1
    )

    assert sampled.corner_ids.tolist() == [4, 6]
    np.testing.assert_allclose(
        sampled.points_camera,
        np.array([[0.01, 0.01], [0.0, 0.005], [1.0, 0.5]]),
    )


def test_sample_depth_uses_local_median_window_to_reject_bad_center_pixel():
    depth = np.zeros((5, 5), dtype=np.uint16)
    depth[1:4, 1:4] = np.array([
        [1000, 1000, 1000],
        [1000, 9000, 1000],
        [1000, 1000, 1000],
    ], dtype=np.uint16)
    K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
    corners = np.array([[2.0], [2.0]])
    ids = np.array([8])

    sampled = sample_depth_at_corners(
        depth, corners, ids, K,
        depth_scale=0.001, min_depth=0.1, max_depth=10.0, window_size=3
    )

    assert sampled.corner_ids.tolist() == [8]
    np.testing.assert_allclose(sampled.points_camera[:, 0], np.array([0.0, 0.0, 1.0]))


def test_load_poses_csv_allows_units_rotation_order_and_direction():
    from st_handeye.io import load_poses_csv

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poses.csv")
        with open(path, "w") as f:
            f.write("1.0,2.0,3.0,10.0,20.0,30.0\n")

        poses = load_poses_csv(
            path,
            trans_unit=1.0,
            rot_order="zyx",
            rot_deg=True,
            invert=False,
        )

    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    expected[:3, :3] = Rotation.from_euler("zyx", [10.0, 20.0, 30.0], degrees=True).as_matrix()
    np.testing.assert_allclose(poses[0], expected)


def test_load_poses_csv_defaults_to_flange_to_world_without_inversion():
    from st_handeye.io import load_poses_csv

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poses.csv")
        with open(path, "w") as f:
            f.write("1.0,2.0,3.0,10.0,20.0,30.0\n")

        poses = load_poses_csv(
            path,
            trans_unit=1.0,
            rot_order="zyx",
            rot_deg=True,
        )

    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    expected[:3, :3] = Rotation.from_euler("zyx", [10.0, 20.0, 30.0], degrees=True).as_matrix()
    np.testing.assert_allclose(poses[0], expected)


def test_depth_weight_default_is_conservative_for_meter_scale_residuals():
    assert OptimizationParams().depth_weight == 0.01


def test_robust_kernel_mapping_supports_expected_scipy_losses():
    assert scipy_loss_from_kernel("NONE") == "linear"
    assert scipy_loss_from_kernel("huber") == "huber"
    assert scipy_loss_from_kernel("soft_l1") == "soft_l1"
    assert scipy_loss_from_kernel("cauchy") == "cauchy"


def test_robust_kernel_defaults_to_huber():
    assert OptimizationParams().robust_kernel == "huber"


def test_quality_filter_rejects_low_detection_quality_and_bad_pnp_fit():
    calibrator = HandEyeCalibrator.__new__(HandEyeCalibrator)
    detections = [
        make_quality_detection(50, 35, 0.25),
        make_quality_detection(39, 35, 0.25),
        make_quality_detection(50, 29, 0.25),
        make_quality_detection(50, 35, 1.25),
    ]

    kept = calibrator._quality_filter(
        detections,
        min_corners=40,
        min_markers=30,
        max_pnp_reprojection=1.0,
    )

    assert kept == [0]


def test_constraint_filter_iteratively_rejects_pose_outlier():
    calibrator = HandEyeCalibrator.__new__(HandEyeCalibrator)
    primary = make_transform(0.02, -0.01, 0.08, [0.02, -0.01, 0.03])
    target = make_transform(0.3, -0.08, 0.01, [-0.02, 0.01, -0.04])
    poses = [
        make_transform(0.00, 0.00, 0.00, [0.00, 0.00, 0.00]),
        make_transform(0.05, 0.00, 0.01, [0.03, -0.01, 0.02]),
        make_transform(0.00, 0.06, 0.00, [-0.02, 0.04, 0.01]),
        make_transform(0.04, -0.02, 0.02, [0.01, 0.03, -0.02]),
        make_transform(-0.03, 0.04, 0.01, [-0.04, 0.01, 0.03]),
        make_transform(0.02, 0.03, -0.01, [0.02, -0.03, 0.04]),
    ]
    pnp_poses = [
        derive_object_to_camera(CalibrationSetup.EYE_IN_HAND, pose, primary, target)
        for pose in poses
    ]
    pnp_poses[-1] = pnp_poses[-1] @ make_transform(
        0.12, 0.02, -0.01, [0.0, 0.0, np.deg2rad(35.0)]
    )
    init = InitialGuess(primary, target, pnp_poses, pnp_poses, CalibrationSetup.EYE_IN_HAND)

    kept = calibrator._filter(
        init, poses, rot_thresh=2.0, trans_thresh=0.020,
        setup=CalibrationSetup.EYE_IN_HAND, max_iterations=3, mad_scale=3.5,
    )

    assert kept == [0, 1, 2, 3, 4]


def test_filter_diagnostics_csv_records_filter_reason_and_metrics():
    calibrator = HandEyeCalibrator.__new__(HandEyeCalibrator)
    detections = [
        make_quality_detection(50, 35, 0.25),
        make_quality_detection(39, 35, 0.25),
    ]
    diagnostics = calibrator._build_filter_diagnostics(detections)
    diagnostics[1]["filter_reason"] = "low_corners"
    diagnostics[1]["constraint_translation_mm"] = 12.5
    diagnostics[1]["constraint_rotation_deg"] = 0.6

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "diagnostics.csv")
        calibrator._save_filter_diagnostics(out, diagnostics)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))

    assert rows[1]["filter_reason"] == "low_corners"
    assert rows[1]["num_corners"] == "39"
    assert rows[1]["constraint_translation_mm"] == "12.5"


def test_sigma_parameters_override_legacy_optimizer_weights():
    params = OptimizationParams(
        projection_weight=100.0,
        pose_weight=100.0,
        projection_sigma_px=2.0,
        pose_trans_sigma_m=0.5,
        pose_rot_sigma_deg=10.0,
        depth_sigma_m=0.25,
    )

    weights = GraphOptimizer()._residual_weights(params)

    assert weights["projection"] == 0.5
    assert weights["pose_trans"] == 2.0
    assert np.isclose(weights["pose_rot"], 1.0 / np.deg2rad(10.0))
    assert weights["depth"] == 4.0


def test_optimizer_defaults_use_bounded_interactive_iterations():
    assert OptimizationParams().num_iterations == 500
    assert OptimizationParams().optimizer_backend == "quaternion_manifold"


def test_optimizer_supports_quaternion_manifold_backend_round_trip():
    pose = make_transform(0.03, -0.02, 0.12, [0.08, -0.04, 0.03])
    optimizer = GraphOptimizer()

    state = optimizer._T2manifold(pose)
    restored = optimizer._manifold_to_T(state)

    np.testing.assert_allclose(restored, pose, atol=1e-9)
    assert state.shape == (7,)
    assert state[0] >= 0.0


def test_optimizer_can_apply_lie_increment_on_quaternion_manifold():
    base = make_transform(0.03, -0.02, 0.12, [0.08, -0.04, 0.03])
    delta = np.array([0.02, -0.01, 0.03, 0.01, 0.02, -0.04])
    optimizer = GraphOptimizer()

    updated = optimizer._apply_manifold_delta(base, delta)
    recovered = optimizer._manifold_pose_delta(base, updated)

    np.testing.assert_allclose(recovered, delta, atol=1e-8)


def test_graph_optimizer_legacy_backend_keeps_current_global_pose_reference_behavior():
    pattern = np.array([
        [-0.04, 0.04, 0.04, -0.04, 0.0],
        [-0.03, -0.03, 0.03, 0.03, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    primary = make_transform(0.04, -0.02, 0.08, [0.03, -0.02, 0.01])
    target = make_transform(0.2, 0.03, 0.1, [-0.02, 0.01, 0.04])
    poses = [
        make_transform(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_transform(0.05, 0.0, 0.02, [0.02, 0.01, 0.0]),
        make_transform(0.0, 0.04, 0.01, [-0.01, 0.02, 0.03]),
    ]
    detections = []
    pnp_poses = []
    for pose in poses:
        T_O2C = derive_object_to_camera(CalibrationSetup.EYE_IN_HAND, pose, primary, target)
        pts_cam = T_O2C[:3, :3] @ pattern + T_O2C[:3, 3:4]
        uv = K @ pts_cam
        detections.append(DetectionResult(
            corners_2d=uv[:2] / uv[2:],
            corner_ids=np.arange(pattern.shape[1]),
            success=True,
            image_path="",
            num_corners=pattern.shape[1],
        ))
        pnp_poses.append(T_O2C @ make_transform(0.03, -0.02, 0.01, [0.04, 0.0, -0.02]))

    init = InitialGuess(primary, target, pnp_poses, pnp_poses, CalibrationSetup.EYE_IN_HAND)
    params = OptimizationParams(
        num_iterations=20,
        projection_weight=1.0,
        pose_weight=100.0,
        optimizer_backend="rotvec_scipy",
    )

    _, _, frame_refs, derived = GraphOptimizer().optimize(
        pattern, detections, poses, init, K, params, setup=CalibrationSetup.EYE_IN_HAND
    )

    for ref, pnp, der in zip(frame_refs, pnp_poses, derived):
        np.testing.assert_allclose(ref, pnp)
        assert transform_error_norm(ref, der) > 1e-3


def test_projection_residuals_match_camera_projection_order():
    pattern = np.array([
        [0.0, 0.05, 0.10],
        [0.0, 0.02, 0.04],
        [0.0, 0.0, 0.0],
    ])
    K = np.array([[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]])
    T_O2C = make_transform(0.0, 0.0, 1.0)
    det = DetectionResult(
        corners_2d=np.array([[51.0, 54.0], [39.0, 44.0]]),
        corner_ids=np.array([0, 2]),
        success=True,
        image_path="",
        num_corners=2,
    )

    residuals = []
    GraphOptimizer()._append_projection_residuals(residuals, pattern, det, T_O2C, K, 2.0)

    pts_cam = T_O2C[:3, :3] @ pattern[:, det.corner_ids] + T_O2C[:3, 3:4]
    uv = K @ pts_cam
    proj = uv[:2] / uv[2:]
    expected = ((det.corners_2d - proj) * 2.0).T.reshape(-1)
    np.testing.assert_allclose(residuals, expected)


def test_reprojection_error_reports_mean_rms_max_and_count():
    class LinearCamera:
        def project(self, points_3d, object2eye=None):
            return points_3d[:2]

    points = np.array([
        [0.0, 2.0, 5.0],
        [0.0, 1.0, 4.0],
        [1.0, 1.0, 1.0],
    ])
    observed = np.array([
        [3.0, 2.0, 9.0],
        [4.0, 1.0, 7.0],
    ])

    result = compute_reprojection_error(points, observed, np.eye(4), LinearCamera())

    assert result["count"] == 3
    np.testing.assert_allclose(result["all"], np.array([5.0, 0.0, 5.0]))
    assert result["mean"] == np.mean([5.0, 0.0, 5.0])
    assert result["rms"] == np.sqrt(np.mean([25.0, 0.0, 25.0]))
    assert result["max"] == 5.0


def test_frame_error_summary_labels_global_and_reference_errors(capsys):
    calibrator = HandEyeCalibrator.__new__(HandEyeCalibrator)
    calibrator._print_frame_error_summary([
        {
            "index": 0,
            "corner_count": 42,
            "reprojection_mean_px": 1.2,
            "reprojection_rms_px": 1.4,
            "reprojection_max_px": 3.5,
            "reference_reprojection_mean_px": 0.6,
            "translation_error": 0.002,
            "rotation_error_deg": 0.7,
        }
    ])

    out = capsys.readouterr().out
    assert "global mean/rms/max(px)" in out
    assert "pnp mean(px)" in out


def test_frame_error_summary_handles_unparticipated_frames(capsys):
    calibrator = HandEyeCalibrator.__new__(HandEyeCalibrator)
    calibrator._print_frame_error_summary([
        {
            "index": 23,
            "corner_count": None,
            "used": False,
            "reprojection_mean_px": None,
            "reprojection_rms_px": None,
            "reprojection_max_px": None,
            "reference_reprojection_mean_px": None,
            "translation_error": None,
            "rotation_error_deg": None,
        }
    ])

    out = capsys.readouterr().out
    assert "  23      --" in out
    assert "pose trans(mm)" in out
    assert "--/--/       --" in out


def test_global_optimizer_keeps_frame_poses_as_visual_references():
    pattern = np.array([
        [-0.04, 0.04, 0.04, -0.04, 0.0],
        [-0.03, -0.03, 0.03, 0.03, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    primary = make_transform(0.04, -0.02, 0.08, [0.03, -0.02, 0.01])
    target = make_transform(0.2, 0.03, 0.1, [-0.02, 0.01, 0.04])
    poses = [
        make_transform(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_transform(0.05, 0.0, 0.02, [0.02, 0.01, 0.0]),
        make_transform(0.0, 0.04, 0.01, [-0.01, 0.02, 0.03]),
    ]
    detections = []
    pnp_poses = []
    for pose in poses:
        T_O2C = derive_object_to_camera(CalibrationSetup.EYE_IN_HAND, pose, primary, target)
        pts_cam = T_O2C[:3, :3] @ pattern + T_O2C[:3, 3:4]
        uv = K @ pts_cam
        detections.append(DetectionResult(
            corners_2d=uv[:2] / uv[2:],
            corner_ids=np.arange(pattern.shape[1]),
            success=True,
            image_path="",
            num_corners=pattern.shape[1],
        ))
        pnp_poses.append(T_O2C @ make_transform(0.03, -0.02, 0.01, [0.04, 0.0, -0.02]))

    init = InitialGuess(primary, target, pnp_poses, pnp_poses, CalibrationSetup.EYE_IN_HAND)
    params = OptimizationParams(num_iterations=20, projection_weight=1.0, pose_weight=100.0)

    _, _, frame_refs, derived = GraphOptimizer().optimize(
        pattern, detections, poses, init, K, params, setup=CalibrationSetup.EYE_IN_HAND
    )

    for ref, pnp, der in zip(frame_refs, pnp_poses, derived):
        np.testing.assert_allclose(ref, pnp)
        assert transform_error_norm(ref, der) > 1e-3


def test_solve_pnp_can_ignore_distortion_for_undistorted_points(monkeypatch):
    from st_handeye.camera import CameraModel
    import cv2

    captured = {}

    def fake_solve_pnp(points_3d, points_2d, camera_matrix, dist_coeffs, flags):
        captured["dist_coeffs"] = dist_coeffs
        return True, np.zeros((3, 1)), np.zeros((3, 1))

    monkeypatch.setattr(cv2, "solvePnP", fake_solve_pnp)
    camera = CameraModel(np.eye(3), np.array([0.1, -0.2, 0.01, 0.02, 0.03]))

    _, ok = camera.solve_pnp(
        np.zeros((3, 4)),
        np.zeros((2, 4)),
        assume_undistorted=True,
    )

    assert ok is True
    np.testing.assert_allclose(captured["dist_coeffs"], np.zeros_like(camera.D))


def test_save_calibration_yaml_writes_new_schema_for_eye_to_hand():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "calibration.yaml")
        save_calibration_yaml(
            out,
            setup=CalibrationSetup.EYE_TO_HAND,
            transforms={"T_C2W": np.eye(4), "T_O2F": make_transform(0.0, 0.0, 0.2)},
            metrics={"reprojection_error": {"mean": 1.25}, "pose_error": {"translation_mean": 0.002}},
            num_images=10,
            num_images_used=8,
            filtered_images=[3, 7],
            depth_used=True,
        )

        with open(out, "r") as f:
            data = yaml.safe_load(f)

    assert data["setup"] == "eye-to-hand"
    assert set(data["transforms"].keys()) == {"T_C2W", "T_O2F"}
    assert data["num_images"] == 10
    assert data["num_images_used"] == 8
    assert data["filtered_images"] == [3, 7]
    assert data["depth_used"] is True
    assert data["metrics"]["reprojection_error"]["mean"] == 1.25


def test_load_camera_params_yaml_accepts_ros_matrix_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "camera.yaml")
        with open(path, "w") as f:
            f.write(
                "camera_matrix:\n"
                "  rows: 3\n"
                "  cols: 3\n"
                "  data: [1, 0, 2, 0, 3, 4, 0, 0, 1]\n"
                "distortion_coefficients:\n"
                "  rows: 1\n"
                "  cols: 5\n"
                "  data: [0.1, 0.2, 0.3, 0.4, 0.5]\n"
            )

        K, D = load_camera_params_yaml(path)

    np.testing.assert_allclose(K, np.array([[1, 0, 2], [0, 3, 4], [0, 0, 1]]))
    np.testing.assert_allclose(D, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))


def test_load_camera_params_yaml_accepts_opencv_matrix_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "camera.yaml")
        with open(path, "w") as f:
            f.write(
                "%YAML:1.0\n"
                "---\n"
                "camera_matrix: !!opencv-matrix\n"
                "  rows: 3\n"
                "  cols: 3\n"
                "  dt: d\n"
                "  data: [1, 0, 2, 0, 3, 4, 0, 0, 1]\n"
                "distortion_coefficients: !!opencv-matrix\n"
                "  rows: 5\n"
                "  cols: 1\n"
                "  dt: d\n"
                "  data: [0.1, 0.2, 0.3, 0.4, 0.5]\n"
            )

        K, D = load_camera_params_yaml(path)

    np.testing.assert_allclose(K, np.array([[1, 0, 2], [0, 3, 4], [0, 0, 1]]))
    np.testing.assert_allclose(D, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))


def test_load_camera_params_yaml_accepts_flat_k_d_keys():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "camera.yaml")
        with open(path, "w") as f:
            f.write(
                "K: [1, 0, 2, 0, 3, 4, 0, 0, 1]\n"
                "D: [0.1, 0.2, 0.3, 0.4, 0.5]\n"
            )

        K, D = load_camera_params_yaml(path)

    np.testing.assert_allclose(K, np.array([[1, 0, 2], [0, 3, 4], [0, 0, 1]]))
    np.testing.assert_allclose(D, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))


def test_eye_to_hand_initial_guess_uses_multiple_frames_to_reduce_pnp_pose_error():
    pattern = np.array([
        [0.0, 0.08, 0.08, 0.0],
        [0.0, 0.0, 0.06, 0.06],
        [0.0, 0.0, 0.0, 0.0],
    ])
    T_C2W_true = make_transform(0.2, -0.1, 0.7, [0.08, -0.04, 0.03])
    T_O2F_true = make_transform(0.03, 0.02, 0.08, [-0.05, 0.02, 0.04])
    poses = [
        make_transform(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_transform(0.12, 0.0, 0.0, [0.0, 0.08, 0.02]),
        make_transform(0.0, 0.10, 0.0, [-0.06, 0.02, 0.05]),
        make_transform(0.05, -0.08, 0.0, [0.04, -0.03, 0.07]),
    ]
    pnp_poses = [
        derive_object_to_camera(CalibrationSetup.EYE_TO_HAND, p, T_C2W_true, T_O2F_true)
        for p in poses
    ]
    pnp_poses[0] = pnp_poses[0] @ make_transform(0.04, -0.03, 0.02)
    detections = [make_detection(pattern.shape[1]) for _ in poses]

    init = compute_initial_guess(
        pattern, detections, poses, PnpCamera(pnp_poses), CalibrationSetup.EYE_TO_HAND
    )

    first_frame_primary = poses[0] @ np.linalg.inv(pnp_poses[0])
    first_frame_error = pnp_constraint_error(
        CalibrationSetup.EYE_TO_HAND, poses, pnp_poses, first_frame_primary, np.eye(4)
    )
    optimized_error = pnp_constraint_error(
        CalibrationSetup.EYE_TO_HAND, poses, pnp_poses, init.T_C2F, init.T_O2W
    )
    assert optimized_error < first_frame_error * 0.75


def test_initial_guess_prefers_depth_3d_correspondences_over_bad_pnp_when_available():
    pattern = np.array([
        [0.0, 0.08, 0.08, 0.0],
        [0.0, 0.0, 0.06, 0.06],
        [0.0, 0.0, 0.0, 0.0],
    ])
    primary_true = make_transform(0.05, -0.02, 0.12, [0.04, -0.03, 0.02])
    target_true = make_transform(0.3, 0.04, 0.08, [-0.02, 0.01, 0.05])
    poses = [
        make_transform(0.00, 0.00, 0.00, [0.00, 0.00, 0.00]),
        make_transform(0.05, 0.01, -0.02, [0.18, 0.10, -0.06]),
        make_transform(-0.04, 0.06, 0.01, [-0.16, 0.21, 0.09]),
        make_transform(0.03, -0.05, 0.02, [0.12, -0.18, 0.15]),
    ]
    true_object_to_camera = [
        derive_object_to_camera(CalibrationSetup.EYE_IN_HAND, pose, primary_true, target_true)
        for pose in poses
    ]
    bad_pnp_poses = [
        T @ make_transform(0.03, -0.02, 0.015, [0.12, -0.08, 0.05])
        for T in true_object_to_camera
    ]
    detections = [make_detection(pattern.shape[1]) for _ in poses]
    depth_samples = [
        make_depth_sample(pattern, T)
        for T in true_object_to_camera
    ]

    init = compute_initial_guess(
        pattern,
        detections,
        poses,
        PnpCamera(bad_pnp_poses),
        CalibrationSetup.EYE_IN_HAND,
        depth_samples=depth_samples,
    )

    depth_error = pnp_constraint_error(
        CalibrationSetup.EYE_IN_HAND, poses, true_object_to_camera, init.T_C2F, init.T_O2W
    )
    pnp_only_init = compute_initial_guess(
        pattern,
        detections,
        poses,
        PnpCamera(bad_pnp_poses),
        CalibrationSetup.EYE_IN_HAND,
    )
    pnp_only_error = pnp_constraint_error(
        CalibrationSetup.EYE_IN_HAND, poses, true_object_to_camera, pnp_only_init.T_C2F, pnp_only_init.T_O2W
    )

    assert depth_error < 1e-6
    assert depth_error < pnp_only_error * 0.1


def test_eye_to_hand_initial_guess_uses_tsai_handeye_with_flange_to_world_robot_poses(monkeypatch):
    pattern = np.array([
        [0.0, 0.08, 0.08, 0.0],
        [0.0, 0.0, 0.06, 0.06],
        [0.0, 0.0, 0.0, 0.0],
    ])
    primary = make_transform(0.2, -0.1, 0.7, [0.08, -0.04, 0.03])
    target = make_transform(0.03, 0.02, 0.08, [-0.05, 0.02, 0.04])
    poses = [
        make_transform(0.02, -0.01, 0.03, [0.02, -0.03, 0.04]),
        make_transform(0.12, 0.0, 0.0, [0.0, 0.18, 0.02]),
        make_transform(0.0, 0.10, 0.0, [-0.16, 0.02, 0.05]),
    ]
    pnp_poses = [
        derive_object_to_camera(CalibrationSetup.EYE_TO_HAND, pose, primary, target)
        for pose in poses
    ]
    detections = [make_detection(pattern.shape[1]) for _ in poses]
    captured = {}

    def fake_calibrate_handeye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method):
        captured["R_gripper2base"] = R_gripper2base
        captured["t_gripper2base"] = t_gripper2base
        captured["R_target2cam"] = R_target2cam
        captured["t_target2cam"] = t_target2cam
        captured["method"] = method
        return primary[:3, :3], primary[:3, 3].reshape(3, 1)

    monkeypatch.setattr("st_handeye.optimizer.cv2.calibrateHandEye", fake_calibrate_handeye)

    init = compute_initial_guess(
        pattern, detections, poses, PnpCamera(pnp_poses), CalibrationSetup.EYE_TO_HAND
    )

    expected_robot = poses[0]
    np.testing.assert_allclose(captured["R_gripper2base"][0], expected_robot[:3, :3])
    np.testing.assert_allclose(captured["t_gripper2base"][0], expected_robot[:3, 3])
    np.testing.assert_allclose(captured["R_target2cam"][0], pnp_poses[0][:3, :3])
    np.testing.assert_allclose(captured["t_target2cam"][0], pnp_poses[0][:3, 3])
    assert captured["method"] == cv2.CALIB_HAND_EYE_TSAI
    np.testing.assert_allclose(init.T_C2F, primary)


def test_eye_to_hand_initial_guess_estimates_target_from_all_frames_after_tsai(monkeypatch):
    pattern = np.array([
        [0.0, 0.08, 0.08, 0.0],
        [0.0, 0.0, 0.06, 0.06],
        [0.0, 0.0, 0.0, 0.0],
    ])
    primary = make_transform(0.2, -0.1, 0.7, [0.08, -0.04, 0.03])
    target = make_transform(0.03, 0.02, 0.08, [-0.05, 0.02, 0.04])
    poses = [
        make_transform(0.02, -0.01, 0.03, [0.02, -0.03, 0.04]),
        make_transform(0.12, 0.0, 0.0, [0.0, 0.18, 0.02]),
        make_transform(0.0, 0.10, 0.0, [-0.16, 0.02, 0.05]),
        make_transform(0.05, -0.08, 0.0, [0.04, -0.03, 0.17]),
    ]
    pnp_poses = [
        derive_object_to_camera(CalibrationSetup.EYE_TO_HAND, pose, primary, target)
        for pose in poses
    ]
    pnp_poses[0] = pnp_poses[0] @ make_transform(0.01, -0.005, 0.006, [0.02, -0.01, 0.0])
    first_frame_target = np.linalg.inv(poses[0]) @ primary @ pnp_poses[0]
    detections = [make_detection(pattern.shape[1]) for _ in poses]

    def fake_calibrate_handeye(*args, **kwargs):
        return primary[:3, :3], primary[:3, 3].reshape(3, 1)

    def fail_if_fallback_used(*args, **kwargs):
        raise AssertionError("eye-to-hand initial target should be estimated after Tsai, not fallback-fit")

    monkeypatch.setattr("st_handeye.optimizer.cv2.calibrateHandEye", fake_calibrate_handeye)
    monkeypatch.setattr("st_handeye.optimizer._fit_global_transforms_from_object_to_camera", fail_if_fallback_used)

    init = compute_initial_guess(
        pattern, detections, poses, PnpCamera(pnp_poses), CalibrationSetup.EYE_TO_HAND
    )

    assert transform_error_norm(target, init.T_O2W) < transform_error_norm(target, first_frame_target)


def test_graph_optimizer_recovers_synthetic_eye_to_hand_solution():
    pattern = np.array([
        [-0.04, 0.04, 0.04, -0.04, 0.0],
        [-0.03, -0.03, 0.03, 0.03, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    T_C2W_true = make_transform(0.1, -0.05, 0.6, [0.08, -0.04, 0.03])
    T_O2F_true = make_transform(0.02, 0.01, 0.12, [-0.05, 0.02, 0.04])
    poses = [
        make_transform(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_transform(0.06, 0.0, 0.0, [0.0, 0.08, 0.02]),
        make_transform(0.0, 0.05, 0.0, [-0.06, 0.02, 0.05]),
        make_transform(0.04, -0.03, 0.0, [0.04, -0.03, 0.07]),
    ]
    detections = []
    pnp_poses = []
    for pose in poses:
        T_O2C = derive_object_to_camera(CalibrationSetup.EYE_TO_HAND, pose, T_C2W_true, T_O2F_true)
        pnp_poses.append(T_O2C)
        pts_cam = T_O2C[:3, :3] @ pattern + T_O2C[:3, 3:4]
        uv = K @ pts_cam
        detections.append(DetectionResult(
            corners_2d=uv[:2] / uv[2:],
            corner_ids=np.arange(pattern.shape[1]),
            success=True,
            image_path="",
            num_corners=pattern.shape[1],
        ))

    init = compute_initial_guess(
        pattern, detections, poses, PnpCamera(pnp_poses), CalibrationSetup.EYE_TO_HAND
    )
    params = OptimizationParams(num_iterations=100, projection_weight=1.0, pose_weight=100.0)
    primary, target, _, _ = GraphOptimizer().optimize(
        pattern, detections, poses, init, K, params, setup=CalibrationSetup.EYE_TO_HAND
    )

    assert transform_error_norm(T_C2W_true, primary) < 1e-5
    assert transform_error_norm(T_O2F_true, target) < 1e-5
