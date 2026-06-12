#!/usr/bin/env python3
"""Analyze per-frame hand position error in hand-eye calibration.

Computes the deviation between:
1. Measured hand2world (from poses.csv)
2. Estimated hand2world (from calibration: H2W = H2E @ inv(O2E) @ O2W)

Uses:
- H2E, O2W from calibration_result.yaml
- O2E from solvePnP (object2eye for each frame)
- H2W_meas from poses.csv (original hand pose in base frame)

This helps identify frames with large pose measurement errors that should be filtered.
"""
import numpy as np
import os
import yaml
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Dict

from st_handeye.io import read_matrix_csv, read_ros_camera_params, load_poses_csv
from st_handeye.board import CharucoBoard
from st_handeye.camera import CameraModel
from st_handeye.types import BoardConfig


def load_calibration_result(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load hand2eye and object2world from calibration_result.yaml."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    if 'transforms' in data:
        setup = data.get('setup', 'eye-in-hand')
        if setup != 'eye-in-hand':
            raise ValueError('analyze_frame_error.py currently supports only eye-in-hand results')
        h2e_data = data['transforms']['T_C2F']['data']
        o2w_data = data['transforms']['T_O2W']['data']
        return np.array(h2e_data).reshape(4, 4), np.array(o2w_data).reshape(4, 4)
    
    h2e_data = data['hand2eye']['data']
    h2e = np.array(h2e_data).reshape(4, 4)
    
    o2w_data = data['object2world']['data']
    o2w = np.array(o2w_data).reshape(4, 4)
    
    return h2e, o2w


def parse_poses_raw(filepath: str, trans_unit: float = 0.001, rot_deg: bool = True) -> List[np.ndarray]:
    """Parse poses.csv to get hand2world (flange pose in base frame).
    
    Returns list of 4x4 hand2world matrices (NOT inverted to world2hand).
    """
    poses_h2w = []
    data = read_matrix_csv(filepath)
    for row in data:
        t = row[:3] * trans_unit  # mm -> m
        r = row[3:6]  # degrees
        pose = np.eye(4)
        pose[:3, 3] = t
        pose[:3, :3] = Rotation.from_euler('xyz', r, degrees=rot_deg).as_matrix()
        poses_h2w.append(pose)
    return poses_h2w


def compute_o2e_from_pnp(pattern_3d, corners_2d, corner_ids, camera: CameraModel) -> np.ndarray:
    """Compute object2eye using solvePnP."""
    p3 = pattern_3d[:, corner_ids]
    T, ok = camera.solve_pnp(p3, corners_2d, assume_undistorted=True)
    return T if ok else None


def compute_pose_error_per_frame(hand2eye: np.ndarray, object2world: np.ndarray,
                                  hand2world_meas: np.ndarray, object2eye: np.ndarray) -> Dict:
    """Per-frame pose error from hand-eye constraint.
    
    Constraint: O2E = H2E @ W2H @ O2W = H2E @ inv(H2W) @ O2W
    Derive: H2W = O2W @ inv(O2E) @ H2E
    
    Error: delta = inv(H2W_meas) @ H2W_est
    """
    h2w_est = object2world @ np.linalg.inv(object2eye) @ hand2eye
    delta = np.linalg.inv(hand2world_meas) @ h2w_est
    trans_error = np.linalg.norm(delta[:3, 3])
    rot_error = Rotation.from_matrix(delta[:3, :3]).magnitude() * 180 / np.pi
    return {'translation': trans_error, 'rotation': rot_error, 'h2w_est': h2w_est}


def analyze_frame_errors(image_dir: str, poses_file: str, 
                          calibration_file: str, camera_params_file: str,
                          board_config: BoardConfig) -> Dict:
    """Analyze per-frame errors and identify problematic frames.
    
    Returns:
        - per_frame_errors: List of dicts with frame index, trans/rot error
        - statistics: Mean/std/max errors
        - outlier_frames: Frames with error > threshold
    """
    # Load calibration result
    hand2eye, object2world = load_calibration_result(calibration_file)
    
    # Load camera
    camera = CameraModel.from_yaml(camera_params_file)
    
    # Load board
    board = CharucoBoard(board_config)
    
    # Load poses (hand2world)
    poses_h2w = parse_poses_raw(poses_file)
    
    # Find images
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith('.png') and '_Color.' in f
    ])
    
    import cv2
    errors = []
    
    print(f"Analyzing {len(image_files)} frames...")
    print(f"{'Frame':>6} {'Corners':>8} {'Trans(mm)':>10} {'Rot(deg)':>10} {'Status':>10}")
    print("-" * 50)
    
    for idx, fname in enumerate(image_files):
        if idx >= len(poses_h2w):
            break
        
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Detect corners
        det = board.detect(img, camera)
        if not det.success:
            print(f"{idx:>6} {'--':>8} {'--':>10} {'--':>10} {'NO DETECT':>10}")
            continue
        
        # Compute O2E from solvePnP
        p3 = board.pattern_3d[:, det.corner_ids]
        o2e, ok = camera.solve_pnp(p3, det.corners_2d, assume_undistorted=True)
        if not ok:
            print(f"{idx:>6} {det.num_corners:>8} {'--':>10} {'--':>10} {'PNP FAIL':>10}")
            continue
        
        # Compute pose error
        h2w_meas = poses_h2w[idx]
        err = compute_pose_error_per_frame(hand2eye, object2world, h2w_meas, o2e)
        
        # Status based on error magnitude
        status = 'OK' if err['translation'] < 0.020 and err['rotation'] < 2.0 else 'BAD'
        
        print(f"{idx:>6} {det.num_corners:>8} {err['translation']*1000:>10.2f} {err['rotation']:>10.2f} {status:>10}")
        
        errors.append({
            'frame': idx,
            'image': fname,
            'corners': det.num_corners,
            'trans_error_mm': err['translation'] * 1000,
            'rot_error_deg': err['rotation'],
            'status': status,
            'h2w_meas_t': h2w_meas[:3, 3],
            'h2w_est_t': err['h2w_est'][:3, 3],
        })
    
    # Statistics
    if len(errors) > 0:
        trans_errors = [e['trans_error_mm'] for e in errors]
        rot_errors = [e['rot_error_deg'] for e in errors]
        
        stats = {
            'trans_mean': np.mean(trans_errors),
            'trans_std': np.std(trans_errors),
            'trans_max': np.max(trans_errors),
            'rot_mean': np.mean(rot_errors),
            'rot_std': np.std(rot_errors),
            'rot_max': np.max(rot_errors),
        }
        
        print("\n" + "=" * 50)
        print(f"Statistics:")
        print(f"  Translation error: mean={stats['trans_mean']:.2f}mm, std={stats['trans_std']:.2f}mm, max={stats['trans_max']:.2f}mm")
        print(f"  Rotation error:    mean={stats['rot_mean']:.2f}deg, std={stats['rot_std']:.2f}deg, max={stats['rot_max']:.2f}deg")
        
        # Identify outliers (e.g., > 2 std from mean or > 20mm)
        outliers = []
        for e in errors:
            if e['trans_error_mm'] > stats['trans_mean'] + 2 * stats['trans_std'] or \
               e['trans_error_mm'] > 20.0:
                outliers.append(e['frame'])
        
        print(f"\nOutlier frames (trans > mean+2std or >20mm): {outliers}")
        
        return {
            'per_frame_errors': errors,
            'statistics': stats,
            'outlier_frames': outliers,
        }
    
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze per-frame pose errors')
    parser.add_argument('image_dir', help='Directory with calibration images')
    parser.add_argument('poses_file', help='CSV file with robot poses')
    parser.add_argument('calibration_file', help='YAML file with calibration result')
    parser.add_argument('-c', '--camera_params', help='Camera params YAML file')
    parser.add_argument('--squares_x', type=int, default=11)
    parser.add_argument('--squares_y', type=int, default=8)
    parser.add_argument('--square_length', type=float, default=0.014)
    parser.add_argument('--marker_length', type=float, default=0.010)
    parser.add_argument('--aruco_dict', default='DICT_5X5_100')
    args = parser.parse_args()
    
    if args.camera_params is None:
        args.camera_params = os.path.join(args.image_dir, 'camera_params.yaml')
    
    board_config = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict=args.aruco_dict
    )
    
    result = analyze_frame_errors(
        args.image_dir, args.poses_file, args.calibration_file,
        args.camera_params, board_config
    )
    
    if result:
        print("\nRecommended: Exclude outlier frames and re-run calibration")
        print("Use: calibrate.py ... --filter_inconsistent")


if __name__ == '__main__':
    main()
