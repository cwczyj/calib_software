import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation
from typing import Dict

from .camera import CameraModel
from .optimizer import derive_object_to_camera
from .types import CalibrationSetup


def _reference_transform_for_consistency(setup, pose, primary_transform):
    setup = CalibrationSetup.parse(setup)
    pose = np.asarray(pose, dtype=np.float64)
    primary_transform = np.asarray(primary_transform, dtype=np.float64)
    if setup == CalibrationSetup.EYE_IN_HAND:
        return pose @ primary_transform
    return np.linalg.inv(pose) @ primary_transform


def _camera_points_for_detection(pattern_3d, det, measured_pose, depth_sample):
    if depth_sample is not None and depth_sample.points_camera.shape[1] > 0:
        return (
            np.asarray(depth_sample.corner_ids, dtype=int),
            np.asarray(depth_sample.points_camera, dtype=np.float64),
        )
    if measured_pose is None:
        return None, None
    corner_ids = np.asarray(det.corner_ids, dtype=int)
    points_object = pattern_3d[:, corner_ids]
    T_O2C = np.asarray(measured_pose, dtype=np.float64)
    points_camera = T_O2C[:3, :3] @ points_object + T_O2C[:3, 3:4]
    return corner_ids, points_camera


def compute_reprojection_error(pattern_3d, corners_2d, T_O2C, camera) -> Dict:
    proj = camera.project(pattern_3d, T_O2C)
    err = np.linalg.norm(corners_2d - proj, axis=0)
    return {
        'count': int(err.shape[0]),
        'mean': np.mean(err),
        'rms': np.sqrt(np.mean(err ** 2)),
        'max': np.max(err),
        'std': np.std(err),
        'all': err
    }


def compute_pose_error(setup, primary_transform, target_transform, T_F2W, T_O2C) -> Dict:
    T_O2C_derived = derive_object_to_camera(setup, T_F2W, primary_transform, target_transform)
    delta = np.linalg.inv(T_O2C) @ T_O2C_derived
    return {
        'translation': np.linalg.norm(delta[:3,3]),
        'rotation': Rotation.from_matrix(delta[:3,:3]).magnitude() * 180 / np.pi
    }


def compute_reference_frame_corner_consistency(pattern_3d, detections, poses, primary_transform,
                                               T_O2C_measured, depth_samples=None,
                                               setup=CalibrationSetup.EYE_IN_HAND) -> Dict:
    setup = CalibrationSetup.parse(setup)

    depth_samples = depth_samples or [None] * len(detections)
    T_O2C_measured = T_O2C_measured or []
    observations_by_corner = {}
    per_image_observations = []

    for i, det in enumerate(detections):
        depth = depth_samples[i] if i < len(depth_samples) else None
        measured_pose = T_O2C_measured[i] if i < len(T_O2C_measured) else None
        corner_ids, points_camera = _camera_points_for_detection(pattern_3d, det, measured_pose, depth)
        if corner_ids is None:
            per_image_observations.append([])
            continue

        T_C2Reference = _reference_transform_for_consistency(setup, poses[i], primary_transform)
        points_reference = T_C2Reference[:3, :3] @ points_camera + T_C2Reference[:3, 3:4]
        image_observations = []
        for column, corner_id in enumerate(corner_ids):
            point = points_reference[:, column]
            observations_by_corner.setdefault(int(corner_id), []).append((i, point))
            image_observations.append((int(corner_id), point))
        per_image_observations.append(image_observations)

    corner_means = {
        corner_id: np.mean([point for _, point in observations], axis=0)
        for corner_id, observations in observations_by_corner.items()
        if len(observations) >= 2
    }
    if not corner_means:
        return None

    all_errors = []
    per_image = []
    for image_observations in per_image_observations:
        image_errors = [
            np.linalg.norm(point - corner_means[corner_id])
            for corner_id, point in image_observations
            if corner_id in corner_means
        ]
        all_errors.extend(image_errors)
        per_image.append(_distance_stats(image_errors))

    if not all_errors:
        return None

    return {
        **_distance_stats(all_errors),
        'per_image': per_image,
        'per_corner': {
            int(corner_id): _distance_stats([
                np.linalg.norm(point - corner_means[corner_id])
                for _, point in observations
            ])
            for corner_id, observations in observations_by_corner.items()
            if corner_id in corner_means
        },
    }


def _distance_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {'count': 0, 'mean': None, 'rms': None, 'max': None, 'std': None}
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'rms': float(np.sqrt(np.mean(values ** 2))),
        'max': float(np.max(values)),
        'std': float(np.std(values)),
    }


def evaluate_calibration(pattern_3d, detections, poses, primary_transform, target_transform,
                          T_O2C_opt, camera, T_O2C_derived=None,
                          setup=CalibrationSetup.EYE_IN_HAND,
                          T_O2C_measured=None, depth_samples=None) -> Dict:
    setup = CalibrationSetup.parse(setup)
    n = len(poses)

    if T_O2C_derived is None:
        T_O2C_derived = [
            derive_object_to_camera(setup, poses[i], primary_transform, target_transform)
            for i in range(n)
        ]

    reproj_derived = []
    reproj_derived_stats = []
    all_err_derived = []
    for i, det in enumerate(detections):
        p3 = pattern_3d[:, det.corner_ids]
        r = compute_reprojection_error(p3, det.corners_2d, T_O2C_derived[i], camera)
        reproj_derived.append(r['mean'])
        reproj_derived_stats.append({
            'count': r['count'],
            'mean': r['mean'],
            'rms': r['rms'],
            'max': r['max'],
            'std': r['std'],
        })
        all_err_derived.extend(r['all'])

    reproj_opt = []
    reproj_opt_stats = []
    all_err_opt = []
    for i, det in enumerate(detections):
        p3 = pattern_3d[:, det.corner_ids]
        r = compute_reprojection_error(p3, det.corners_2d, T_O2C_opt[i], camera)
        reproj_opt.append(r['mean'])
        reproj_opt_stats.append({
            'count': r['count'],
            'mean': r['mean'],
            'rms': r['rms'],
            'max': r['max'],
            'std': r['std'],
        })
        all_err_opt.extend(r['all'])

    pose_errors = [compute_pose_error(setup, primary_transform, target_transform, poses[i], T_O2C_opt[i])
                   for i in range(n)]
    reference_consistency = compute_reference_frame_corner_consistency(
        pattern_3d, detections, poses, primary_transform,
        T_O2C_measured if T_O2C_measured is not None else T_O2C_opt,
        depth_samples=depth_samples,
        setup=setup,
    )

    return {
        'reprojection_error': {
            'mean': np.mean(all_err_derived), 'std': np.std(all_err_derived), 'max': np.max(all_err_derived),
            'rms': np.sqrt(np.mean(np.asarray(all_err_derived) ** 2)),
            'per_image': reproj_derived,
            'per_image_stats': reproj_derived_stats,
            'mean_optimized': np.mean(all_err_opt),
            'rms_optimized': np.sqrt(np.mean(np.asarray(all_err_opt) ** 2)),
            'max_optimized': np.max(all_err_opt),
            'std_optimized': np.std(all_err_opt),
            'per_image_optimized_stats': reproj_opt_stats,
            'per_image_optimized': reproj_opt
        },
        'pose_error': {
            'translation_mean': np.mean([p['translation'] for p in pose_errors]),
            'rotation_mean': np.mean([p['rotation'] for p in pose_errors]),
            'per_image': pose_errors
        },
        'base_consistency': reference_consistency
    }


def compute_base_frame_corner_consistency(pattern_3d, detections, poses, primary_transform,
                                          T_O2C_measured, depth_samples=None,
                                          setup=CalibrationSetup.EYE_IN_HAND) -> Dict:
    return compute_reference_frame_corner_consistency(
        pattern_3d=pattern_3d,
        detections=detections,
        poses=poses,
        primary_transform=primary_transform,
        T_O2C_measured=T_O2C_measured,
        depth_samples=depth_samples,
        setup=setup,
    )


def generate_visualization(image_paths, pattern_3d, detections, T_O2C_list,
                           camera, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (img_path, det, T_O2C) in enumerate(zip(image_paths, detections, T_O2C_list)):
        if not det.success:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        undist = camera.undistort(img)
        p3 = pattern_3d[:, det.corner_ids]
        proj = camera.project(p3, T_O2C)

        for j, (pt, cid) in enumerate(zip(det.corners_2d.T, det.corner_ids)):
            cv2.circle(undist, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
            cv2.putText(undist, str(cid), (int(pt[0])+5, int(pt[1])-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        for pt in proj.T:
            cv2.drawMarker(undist, (int(pt[0]), int(pt[1])), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)
        
        for d, r in zip(det.corners_2d.T, proj.T):
            cv2.line(undist, (int(d[0]),int(d[1])), (int(r[0]),int(r[1])), (0, 255, 255), 1)

        err = np.linalg.norm(det.corners_2d.T - proj.T, axis=1)
        cv2.putText(undist, f'img{i}: mean={np.mean(err):.2f} max={np.max(err):.2f} corners={len(det.corner_ids)}',
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        rvec = Rotation.from_matrix(T_O2C[:3,:3]).as_rotvec().reshape(3,1).astype(np.float32)
        tvec = T_O2C[:3,3].reshape(3,1).astype(np.float32)
        cv2.drawFrameAxes(undist, camera.K.astype(np.float32), camera.D.astype(np.float32),
                          rvec, tvec, 0.028)
        
        cv2.imwrite(os.path.join(output_dir, f'reproj_{i:03d}.png'), undist)
    print(f'Visualization saved to {output_dir}/')
