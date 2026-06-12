"""Factor-graph style hand-eye calibration with SciPy least_squares.

Coordinate systems: F=Flange, C=Camera, O=Object(board), W=World
Transform naming: T_A2B means P_B = T_A2B @ P_A

Default optimization variables are two global SE(3) transforms:
    - eye-in-hand: T_C2F and T_O2W
    - eye-to-hand: T_C2W and T_O2F

The legacy diagnostic mode can also optimize one T_O2C_i per image.

Residuals:
    1. Pinhole reprojection error from derived T_O2C_i
    2. Optional RGB-D 3D point error from depth back-projection
    3. Legacy diagnostic mode only: hand-eye kinematic constraint
       tying optimized T_O2C_i to robot poses

This is not a g2o binding; it solves the same nonlinear least-squares
factor graph directly with scipy.optimize.least_squares.
"""
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from typing import List, Tuple
import cv2
import nanomanifold.SE3 as se3

from .types import CalibrationSetup, InitialGuess, OptimizationParams


def scipy_loss_from_kernel(kernel: str) -> str:
    name = (kernel or "NONE").lower()
    if name == "none":
        return "linear"
    if name in ("huber", "soft_l1", "cauchy"):
        return name
    raise ValueError(f"Unsupported robust kernel: {kernel}")


def derive_object_to_camera(setup, robot_pose_w2f, primary_transform, target_transform):
    setup = CalibrationSetup.parse(setup)
    if setup == CalibrationSetup.EYE_IN_HAND:
        return np.linalg.inv(primary_transform) @ np.linalg.inv(robot_pose_w2f) @ target_transform
    if setup == CalibrationSetup.EYE_TO_HAND:
        return np.linalg.inv(primary_transform) @ robot_pose_w2f @ target_transform
    raise ValueError(f"Unsupported calibration setup: {setup}")


def _primary_from_first_measurement(setup, pose, object_to_camera, target_transform):
    setup = CalibrationSetup.parse(setup)
    if setup == CalibrationSetup.EYE_IN_HAND:
        return np.linalg.inv(pose) @ target_transform @ np.linalg.inv(object_to_camera)
    if setup == CalibrationSetup.EYE_TO_HAND:
        return pose @ target_transform @ np.linalg.inv(object_to_camera)
    raise ValueError(f"Unsupported calibration setup: {setup}")


def _target_from_pose_primary_measurement(setup, pose, primary_transform, object_to_camera):
    setup = CalibrationSetup.parse(setup)
    if setup == CalibrationSetup.EYE_IN_HAND:
        return pose @ primary_transform @ object_to_camera
    if setup == CalibrationSetup.EYE_TO_HAND:
        return np.linalg.inv(pose) @ primary_transform @ object_to_camera
    raise ValueError(f"Unsupported calibration setup: {setup}")


def _target_candidates_from_primary(setup, poses, object_to_camera_list, primary):
    return [
        _target_from_pose_primary_measurement(setup, pose, primary, object_to_camera)
        for pose, object_to_camera in zip(poses, object_to_camera_list)
    ]


class GraphOptimizer:
    OPTIMIZER_TOLERANCE = 1e-8
    
    def optimize(self, pattern_3d, detections, poses, initial_guess, 
                 camera_matrix, params, setup=CalibrationSetup.EYE_IN_HAND,
                 depth_samples=None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        setup = CalibrationSetup.parse(setup)
        backend = getattr(params, "optimizer_backend", "quaternion_manifold").lower()
        if backend == "rotvec_scipy":
            if params.optimize_frame_poses:
                return self._optimize_with_frame_poses(
                    pattern_3d, detections, poses, initial_guess,
                    camera_matrix, params, setup, depth_samples
                )
            return self._optimize_global(
                pattern_3d, detections, poses, initial_guess,
                camera_matrix, params, setup, depth_samples
            )
        if backend != "quaternion_manifold":
            raise ValueError(f"Unsupported optimizer backend: {backend}")
        if params.optimize_frame_poses:
            return self._optimize_with_frame_poses_manifold(
                pattern_3d, detections, poses, initial_guess,
                camera_matrix, params, setup, depth_samples
            )
        return self._optimize_global_manifold(
            pattern_3d, detections, poses, initial_guess,
            camera_matrix, params, setup, depth_samples
        )

    def _optimize_global(self, pattern_3d, detections, poses, initial_guess,
                         camera_matrix, params, setup, depth_samples=None):
        n = len(poses)
        depth_samples = depth_samples or [None] * n

        x0 = np.zeros(12)
        x0[0:6] = self._T2v6(initial_guess.T_C2F)
        x0[6:12] = self._T2v6(initial_guess.T_O2W)

        loss = scipy_loss_from_kernel(params.robust_kernel)
        method = 'lm' if loss == 'linear' else 'trf'
        start = time.perf_counter()
        result = least_squares(
            self._global_residuals, x0,
            args=(pattern_3d, detections, poses, camera_matrix, params, setup, depth_samples),
            method=method, max_nfev=params.num_iterations,
            ftol=self.OPTIMIZER_TOLERANCE,
            xtol=self.OPTIMIZER_TOLERANCE,
            gtol=self.OPTIMIZER_TOLERANCE,
            loss=loss,
            f_scale=params.kernel_delta
        )
        elapsed = time.perf_counter() - start
        print(f'iterations={result.nfev}, chi2={np.sum(result.fun**2):.6f}, time={elapsed:.2f}s')

        primary = self._v62T(result.x[0:6])
        target = self._v62T(result.x[6:12])
        T_O2C_derived = [
            derive_object_to_camera(setup, poses[i], primary, target) for i in range(n)
        ]
        frame_references = (
            initial_guess.T_O2C_measured
            if initial_guess.T_O2C_measured is not None
            else initial_guess.T_O2C_pnp
            if initial_guess.T_O2C_pnp is not None
            else T_O2C_derived
        )

        return primary, target, frame_references, T_O2C_derived

    def _global_residuals(self, x, p3d, dets, poses, K, p, setup, depth_samples):
        res = []

        primary = self._v62T(x[0:6])
        target = self._v62T(x[6:12])

        weights = self._residual_weights(p)
        wpj = weights["projection"]
        wdepth = weights["depth"]

        for i, (pose, det) in enumerate(zip(poses, dets)):
            T_O2C = derive_object_to_camera(setup, pose, primary, target)
            self._append_projection_residuals(res, p3d, det, T_O2C, K, wpj)
            self._append_depth_residuals(res, p3d, depth_samples, i, T_O2C, wdepth)

        return np.array(res)

    def _optimize_global_manifold(self, pattern_3d, detections, poses, initial_guess,
                                  camera_matrix, params, setup, depth_samples=None):
        n = len(poses)
        depth_samples = depth_samples or [None] * n

        primary_base = self._T2manifold(initial_guess.T_C2F)
        target_base = self._T2manifold(initial_guess.T_O2W)
        x0 = np.zeros(12)

        loss = scipy_loss_from_kernel(params.robust_kernel)
        method = 'lm' if loss == 'linear' else 'trf'
        start = time.perf_counter()
        result = least_squares(
            self._global_residuals_manifold, x0,
            args=(
                pattern_3d, detections, poses, camera_matrix, params, setup,
                depth_samples, primary_base, target_base,
            ),
            method=method, max_nfev=params.num_iterations,
            ftol=self.OPTIMIZER_TOLERANCE,
            xtol=self.OPTIMIZER_TOLERANCE,
            gtol=self.OPTIMIZER_TOLERANCE,
            loss=loss,
            f_scale=params.kernel_delta
        )
        elapsed = time.perf_counter() - start
        print(f'iterations={result.nfev}, chi2={np.sum(result.fun**2):.6f}, time={elapsed:.2f}s')

        primary = self._apply_manifold_delta(initial_guess.T_C2F, result.x[0:6])
        target = self._apply_manifold_delta(initial_guess.T_O2W, result.x[6:12])
        T_O2C_derived = [
            derive_object_to_camera(setup, poses[i], primary, target) for i in range(n)
        ]
        frame_references = (
            initial_guess.T_O2C_measured
            if initial_guess.T_O2C_measured is not None
            else initial_guess.T_O2C_pnp
            if initial_guess.T_O2C_pnp is not None
            else T_O2C_derived
        )

        return primary, target, frame_references, T_O2C_derived

    def _global_residuals_manifold(self, x, p3d, dets, poses, K, p, setup, depth_samples,
                                   primary_base, target_base):
        res = []

        primary = self._manifold_to_T(self._apply_manifold_delta_to_state(primary_base, x[0:6]))
        target = self._manifold_to_T(self._apply_manifold_delta_to_state(target_base, x[6:12]))

        weights = self._residual_weights(p)
        wpj = weights["projection"]
        wdepth = weights["depth"]

        for i, (pose, det) in enumerate(zip(poses, dets)):
            T_O2C = derive_object_to_camera(setup, pose, primary, target)
            self._append_projection_residuals(res, p3d, det, T_O2C, K, wpj)
            self._append_depth_residuals(res, p3d, depth_samples, i, T_O2C, wdepth)

        return np.array(res)

    def _optimize_with_frame_poses(self, pattern_3d, detections, poses, initial_guess,
                                   camera_matrix, params, setup, depth_samples=None):
        setup = CalibrationSetup.parse(setup)
        n = len(poses)
        depth_samples = depth_samples or [None] * n
        
        x0 = np.zeros(12 + n * 6)
        x0[0:6] = self._T2v6(initial_guess.T_C2F)
        x0[6:12] = self._T2v6(initial_guess.T_O2W)
        
        for i in range(n):
            T_O2C_init = derive_object_to_camera(
                setup, poses[i], initial_guess.T_C2F, initial_guess.T_O2W
            )
            x0[12 + i*6 : 12 + (i+1)*6] = self._T2v6(T_O2C_init)
        
        loss = scipy_loss_from_kernel(params.robust_kernel)
        method = 'lm' if loss == 'linear' else 'trf'
        start = time.perf_counter()
        result = least_squares(
            self._residuals, x0,
            args=(pattern_3d, detections, poses, camera_matrix, params, setup, depth_samples),
            method=method, max_nfev=params.num_iterations,
            ftol=self.OPTIMIZER_TOLERANCE,
            xtol=self.OPTIMIZER_TOLERANCE,
            gtol=self.OPTIMIZER_TOLERANCE,
            loss=loss,
            f_scale=params.kernel_delta
        )
        elapsed = time.perf_counter() - start
        print(f'iterations={result.nfev}, chi2={np.sum(result.fun**2):.6f}, time={elapsed:.2f}s')
        
        primary = self._v62T(result.x[0:6])
        target = self._v62T(result.x[6:12])
        T_O2C_opt = [self._v62T(result.x[12 + i*6 : 12 + (i+1)*6]) for i in range(n)]
        
        T_O2C_derived = [
            derive_object_to_camera(setup, poses[i], primary, target) for i in range(n)
        ]
        
        return primary, target, T_O2C_opt, T_O2C_derived

    def _optimize_with_frame_poses_manifold(self, pattern_3d, detections, poses, initial_guess,
                                            camera_matrix, params, setup, depth_samples=None):
        setup = CalibrationSetup.parse(setup)
        n = len(poses)
        depth_samples = depth_samples or [None] * n

        primary_base = self._T2manifold(initial_guess.T_C2F)
        target_base = self._T2manifold(initial_guess.T_O2W)
        frame_bases = []
        for i in range(n):
            T_O2C_init = derive_object_to_camera(
                setup, poses[i], initial_guess.T_C2F, initial_guess.T_O2W
            )
            frame_bases.append(self._T2manifold(T_O2C_init))

        x0 = np.zeros(12 + n * 6)
        loss = scipy_loss_from_kernel(params.robust_kernel)
        method = 'lm' if loss == 'linear' else 'trf'
        start = time.perf_counter()
        result = least_squares(
            self._residuals_manifold, x0,
            args=(
                pattern_3d, detections, poses, camera_matrix, params, setup,
                depth_samples, primary_base, target_base, frame_bases,
            ),
            method=method, max_nfev=params.num_iterations,
            ftol=self.OPTIMIZER_TOLERANCE,
            xtol=self.OPTIMIZER_TOLERANCE,
            gtol=self.OPTIMIZER_TOLERANCE,
            loss=loss,
            f_scale=params.kernel_delta
        )
        elapsed = time.perf_counter() - start
        print(f'iterations={result.nfev}, chi2={np.sum(result.fun**2):.6f}, time={elapsed:.2f}s')

        primary = self._manifold_to_T(self._apply_manifold_delta_to_state(primary_base, result.x[0:6]))
        target = self._manifold_to_T(self._apply_manifold_delta_to_state(target_base, result.x[6:12]))
        T_O2C_opt = [
            self._manifold_to_T(
                self._apply_manifold_delta_to_state(
                    frame_bases[i], result.x[12 + i * 6: 12 + (i + 1) * 6]
                )
            )
            for i in range(n)
        ]
        T_O2C_derived = [
            derive_object_to_camera(setup, poses[i], primary, target) for i in range(n)
        ]

        return primary, target, T_O2C_opt, T_O2C_derived
    
    def _residuals(self, x, p3d, dets, poses, K, p, setup, depth_samples):
        res = []
        
        primary = self._v62T(x[0:6])
        target = self._v62T(x[6:12])
        
        weights = self._residual_weights(p)
        wpj = weights["projection"]
        wdepth = weights["depth"]
        wpose_trans = weights["pose_trans"]
        wpose_rot = weights["pose_rot"]
        
        for i, (pose, det) in enumerate(zip(poses, dets)):
            T_O2C = self._v62T(x[12 + i*6 : 12 + (i+1)*6])
            self._append_projection_residuals(res, p3d, det, T_O2C, K, wpj)
            self._append_depth_residuals(res, p3d, depth_samples, i, T_O2C, wdepth)
            
            T_O2C_derived = derive_object_to_camera(setup, pose, primary, target)
            delta = np.linalg.inv(T_O2C) @ T_O2C_derived
            
            trans_err = delta[:3, 3] * wpose_trans
            res.extend(trans_err)
            
            rot_err = Rotation.from_matrix(delta[:3,:3]).as_rotvec() * wpose_rot
            res.extend(rot_err)
        
        return np.array(res)

    def _residuals_manifold(self, x, p3d, dets, poses, K, p, setup, depth_samples,
                            primary_base, target_base, frame_bases):
        res = []

        primary = self._manifold_to_T(self._apply_manifold_delta_to_state(primary_base, x[0:6]))
        target = self._manifold_to_T(self._apply_manifold_delta_to_state(target_base, x[6:12]))

        weights = self._residual_weights(p)
        wpj = weights["projection"]
        wdepth = weights["depth"]
        wpose_trans = weights["pose_trans"]
        wpose_rot = weights["pose_rot"]

        for i, (pose, det) in enumerate(zip(poses, dets)):
            T_O2C = self._manifold_to_T(
                self._apply_manifold_delta_to_state(
                    frame_bases[i], x[12 + i * 6: 12 + (i + 1) * 6]
                )
            )
            self._append_projection_residuals(res, p3d, det, T_O2C, K, wpj)
            self._append_depth_residuals(res, p3d, depth_samples, i, T_O2C, wdepth)

            T_O2C_derived = derive_object_to_camera(setup, pose, primary, target)
            pose_err = self._manifold_pose_delta(T_O2C, T_O2C_derived)
            res.extend(pose_err[:3] * wpose_trans)
            res.extend(pose_err[3:6] * wpose_rot)

        return np.array(res)

    def _append_projection_residuals(self, res, p3d, det, T_O2C, K, weight):
        p3 = p3d[:, det.corner_ids] if det.corner_ids.shape[0] < p3d.shape[1] else p3d
        pcam = T_O2C[:3, :3] @ p3 + T_O2C[:3, 3:4]
        uv = K @ pcam
        proj = uv[:2] / uv[2:]
        proj_err = (det.corners_2d - proj) * weight
        res.extend(proj_err.T.reshape(-1))

    def _append_depth_residuals(self, res, p3d, depth_samples, index, T_O2C, weight):
        depth = depth_samples[index] if index < len(depth_samples) else None
        if depth is not None and depth.points_camera.shape[1] > 0:
            p3_depth = p3d[:, depth.corner_ids]
            pred = T_O2C[:3, :3] @ p3_depth + T_O2C[:3, 3:4]
            depth_err = (depth.points_camera - pred) * weight
            res.extend(depth_err.ravel(order="F"))
    
    def _T2v6(self, T):
        return np.concatenate([T[:3,3], Rotation.from_matrix(T[:3,:3]).as_rotvec()])
    
    def _v62T(self, v):
        T = np.eye(4)
        T[:3,3] = v[:3]
        T[:3,:3] = Rotation.from_rotvec(v[3:6]).as_matrix()
        return T

    def _T2manifold(self, T):
        state = np.asarray(se3.from_matrix(np.asarray(T, dtype=float)), dtype=float)
        if state[0] < 0.0:
            state[:4] *= -1.0
        return state

    def _manifold_to_T(self, state):
        return np.asarray(se3.to_matrix(np.asarray(state, dtype=float)), dtype=float)

    def _v62xi(self, v):
        v = np.asarray(v, dtype=float)
        return np.concatenate([v[3:6], v[0:3]])

    def _xi2v6(self, xi):
        xi = np.asarray(xi, dtype=float)
        return np.concatenate([xi[3:6], xi[0:3]])

    def _apply_manifold_delta_to_state(self, state, delta):
        updated = np.asarray(
            se3.multiply(se3.exp(self._v62xi(delta)), np.asarray(state, dtype=float)),
            dtype=float,
        )
        if updated[0] < 0.0:
            updated[:4] *= -1.0
        return updated

    def _apply_manifold_delta(self, T, delta):
        return self._manifold_to_T(self._apply_manifold_delta_to_state(self._T2manifold(T), delta))

    def _manifold_pose_delta(self, base_T, target_T):
        base_state = self._T2manifold(base_T)
        target_state = self._T2manifold(target_T)
        delta_state = np.asarray(se3.multiply(target_state, se3.inverse(base_state)), dtype=float)
        return self._xi2v6(np.asarray(se3.log(delta_state), dtype=float))

    def _residual_weights(self, params):
        def sigma_weight(value):
            return None if value is None else 1.0 / float(value)

        projection = sigma_weight(params.projection_sigma_px)
        depth = sigma_weight(params.depth_sigma_m)
        pose_trans = sigma_weight(params.pose_trans_sigma_m)
        pose_rot = (
            None if params.pose_rot_sigma_deg is None
            else 1.0 / np.deg2rad(float(params.pose_rot_sigma_deg))
        )
        return {
            "projection": projection if projection is not None else np.sqrt(params.projection_weight),
            "depth": depth if depth is not None else np.sqrt(params.depth_weight),
            "pose_trans": (
                pose_trans if pose_trans is not None
                else np.sqrt(params.pose_weight * params.pose_trans_scale)
            ),
            "pose_rot": (
                pose_rot if pose_rot is not None
                else np.sqrt(params.pose_weight * params.pose_rot_scale)
            ),
        }


def _fit_global_transforms_from_object_to_camera(setup, poses, object_to_camera_list,
                                                 primary_init=None, target_init=None,
                                                 max_nfev=200):
    setup = CalibrationSetup.parse(setup)
    if target_init is None:
        target_init = np.eye(4)
    if primary_init is None:
        primary_init = _primary_from_first_measurement(
            setup, poses[0], object_to_camera_list[0], target_init
        )

    helper = GraphOptimizer()
    x0 = np.zeros(12)
    x0[:6] = helper._T2v6(primary_init)
    x0[6:12] = helper._T2v6(target_init)

    def residual(x):
        primary = helper._v62T(x[:6])
        target = helper._v62T(x[6:12])
        res = []
        for pose, measured in zip(poses, object_to_camera_list):
            predicted = derive_object_to_camera(setup, pose, primary, target)
            delta = np.linalg.inv(measured) @ predicted
            res.extend(delta[:3, 3])
            res.extend(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
        return np.asarray(res)

    result = least_squares(
        residual, x0, method="trf", max_nfev=max_nfev,
        ftol=1e-12, xtol=1e-12, gtol=1e-12
    )
    return helper._v62T(result.x[:6]), helper._v62T(result.x[6:12])


def _make_transform(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def _tsai_handeye_primary(robot_transforms, target_to_camera_list):
    R_robot, t_robot, R_target, t_target = [], [], [], []
    for robot, target_to_camera in zip(robot_transforms, target_to_camera_list):
        R_robot.append(robot[:3, :3].astype(np.float64))
        t_robot.append(robot[:3, 3].astype(np.float64))
        R_target.append(target_to_camera[:3, :3].astype(np.float64))
        t_target.append(target_to_camera[:3, 3].astype(np.float64))

    R_primary, t_primary = cv2.calibrateHandEye(
        R_robot, t_robot, R_target, t_target, cv2.CALIB_HAND_EYE_TSAI
    )
    return _make_transform(R_primary, t_primary)


def _estimate_target_from_primary(setup, poses, object_to_camera_list, primary):
    setup = CalibrationSetup.parse(setup)
    candidates = _target_candidates_from_primary(setup, poses, object_to_camera_list, primary)

    target = np.eye(4)
    target[:3, :3] = Rotation.from_matrix([c[:3, :3] for c in candidates]).mean().as_matrix()
    target[:3, 3] = np.mean([c[:3, 3] for c in candidates], axis=0)
    return target


def compute_initial_guess(pattern_3d, detections, poses, camera,
                          setup=CalibrationSetup.EYE_IN_HAND,
                          pnp_method="iterative",
                          depth_samples=None) -> InitialGuess:
    setup = CalibrationSetup.parse(setup)
    T_O2C_pnp = []
    for det in detections:
        p3 = pattern_3d[:, det.corner_ids]
        T, ok = camera.solve_pnp(
            p3, det.corners_2d, pnp_method=pnp_method,
            assume_undistorted=True
        )
        T_O2C_pnp.append(T if ok else np.eye(4))
    T_O2C_measured = _depth_object_to_camera_measurements(pattern_3d, depth_samples, T_O2C_pnp)
    
    if setup == CalibrationSetup.EYE_IN_HAND:
        try:
            primary = _tsai_handeye_primary(poses, T_O2C_measured)
        except cv2.error:
            primary = np.eye(4)
        target = _estimate_target_from_primary(setup, poses, T_O2C_measured, primary)
    else:
        target0 = np.eye(4)
        primary0 = _primary_from_first_measurement(
            setup, poses[0], T_O2C_measured[0], target0
        )
        try:
            primary = _tsai_handeye_primary(poses, T_O2C_measured)
            target = _estimate_target_from_primary(setup, poses, T_O2C_measured, primary)
            if pnp_constraint_error(setup, poses, T_O2C_measured, primary, target) >= pnp_constraint_error(
                setup, poses, T_O2C_measured, primary0, target0
            ):
                primary, target = _fit_global_transforms_from_object_to_camera(
                    setup, poses, T_O2C_measured, primary0, target0
                )
        except cv2.error:
            primary, target = _fit_global_transforms_from_object_to_camera(
                setup, poses, T_O2C_measured, primary0, target0
            )

    T_O2C_list = [
        derive_object_to_camera(setup, poses[i], primary, target)
        for i in range(len(poses))
    ]
    
    return InitialGuess(primary, target, T_O2C_list, T_O2C_pnp, setup, T_O2C_measured)


def pnp_constraint_error(setup, poses, object_to_camera_list, primary, target):
    total = 0.0
    for pose, measured in zip(poses, object_to_camera_list):
        predicted = derive_object_to_camera(setup, pose, primary, target)
        delta = np.linalg.inv(measured) @ predicted
        total += np.linalg.norm(delta[:3, 3])
        total += Rotation.from_matrix(delta[:3, :3]).magnitude()
    return total


def _kabsch_object_to_camera(points_object, points_camera):
    points_object = np.asarray(points_object, dtype=np.float64)
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_object.shape[0] != 3 or points_camera.shape[0] != 3:
        raise ValueError("3D point correspondences must have shape (3, N)")
    if points_object.shape[1] < 3 or points_camera.shape[1] < 3:
        raise ValueError("At least 3 point correspondences are required")

    centered_object = points_object - np.mean(points_object, axis=1, keepdims=True)
    centered_camera = points_camera - np.mean(points_camera, axis=1, keepdims=True)
    if np.linalg.matrix_rank(centered_object) < 2:
        raise ValueError("Depth correspondences are degenerate")

    covariance = centered_camera @ centered_object.T
    U, _, Vt = np.linalg.svd(covariance)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    t = np.mean(points_camera, axis=1) - R @ np.mean(points_object, axis=1)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _depth_object_to_camera_measurements(pattern_3d, depth_samples, fallback_poses):
    measurements = []
    for fallback, depth in zip(fallback_poses, depth_samples or []):
        if depth is None or depth.points_camera.shape[1] < 3:
            measurements.append(fallback)
            continue
        try:
            measurements.append(
                _kabsch_object_to_camera(pattern_3d[:, depth.corner_ids], depth.points_camera)
            )
        except (ValueError, np.linalg.LinAlgError):
            measurements.append(fallback)
    if len(measurements) < len(fallback_poses):
        measurements.extend(fallback_poses[len(measurements):])
    return measurements
