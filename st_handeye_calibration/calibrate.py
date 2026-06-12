#!/usr/bin/env python3
import argparse
import os
import sys

from st_handeye import HandEyeCalibrator, BoardConfig, CalibrationSetup, OptimizationParams


def main():
    parser = argparse.ArgumentParser(description='Hand-eye calibration')
    parser.add_argument('image_dir', help='Directory with calibration images')
    parser.add_argument('poses_file', help='CSV file with robot poses')
    parser.add_argument('-c', '--camera_params', help='Camera params YAML file')
    parser.add_argument('--squares_x', type=int, default=14, help='Number of squares in X (default: 14)')
    parser.add_argument('--squares_y', type=int, default=9, help='Number of squares in Y (default: 9)')
    parser.add_argument('--square_length', type=float, default=0.020, help='Square length in meters (default: 0.020)')
    parser.add_argument('--marker_length', type=float, default=0.015, help='Marker length in meters (default: 0.015)')
    parser.add_argument('--aruco_dict', default='DICT_5X5_50', help='ArUco dictionary (default: DICT_5X5_50)')
    parser.add_argument('--output', default='calibration_result.yaml', help='Output YAML file')
    parser.add_argument('--setup', choices=[s.value for s in CalibrationSetup],
                        default=CalibrationSetup.EYE_IN_HAND.value,
                        help='Calibration setup: eye-in-hand or eye-to-hand')
    parser.add_argument('--visualize', action='store_true', help='Generate reprojection visualization')
    parser.add_argument('--save_detection', action='store_true', help='Save detection visualization images')
    parser.add_argument('--filter_inconsistent', action='store_true', help='Filter inconsistent poses')
    parser.add_argument('--rot_threshold', type=float, default=2.0, help='Rotation threshold in degrees (default: 2.0)')
    parser.add_argument('--trans_threshold', type=float, default=0.020, help='Translation threshold in meters (default: 0.020)')
    parser.add_argument('--min_filter_corners', type=int, default=40,
                        help='Minimum ChArUco corners kept by --filter_inconsistent')
    parser.add_argument('--min_filter_markers', type=int, default=30,
                        help='Minimum ArUco markers kept by --filter_inconsistent')
    parser.add_argument('--max_filter_pnp_reprojection', type=float, default=1.0,
                        help='Maximum per-frame PnP reprojection error kept by --filter_inconsistent, in pixels')
    parser.add_argument('--filter_iterations', type=int, default=3,
                        help='Maximum robust constraint-filter iterations')
    parser.add_argument('--filter_mad_scale', type=float, default=3.5,
                        help='MAD scale for adaptive robust constraint filtering')
    parser.add_argument('--pose_trans_unit', type=float, default=0.001,
                        help='Multiplier converting pose translation values to meters')
    parser.add_argument('--pose_rot_order', default='xyz',
                        help='Euler rotation order for pose CSV rows')
    parser.add_argument('--pose_rot_unit', choices=['deg', 'rad'], default='deg',
                        help='Rotation unit for pose CSV rows')
    parser.add_argument('--pose_direction', choices=['invert', 'as-is'], default='as-is',
                        help='Use invert only when CSV pose must be inverted before calibration')
    parser.add_argument('--pnp_method', choices=['iterative', 'epnp', 'sqpnp', 'ippe'],
                        default='iterative', help='PnP method used for board pose estimation')
    parser.add_argument('--diagnostics_output',
                        help='Optional CSV path for per-frame detection/filter diagnostics')
    parser.add_argument('--use_depth', choices=['off', 'optional', 'required'], default='off',
                        help='Use aligned 16-bit depth PNGs in optimization')
    parser.add_argument('--depth_dir', help='Directory containing {NNN}_Depth.png files')
    parser.add_argument('--depth_scale', type=float, default=0.001,
                        help='Scale from depth pixel units to meters (default: 0.001 for mm)')
    parser.add_argument('--min_depth', type=float, default=0.05, help='Minimum valid depth in meters')
    parser.add_argument('--max_depth', type=float, default=5.0, help='Maximum valid depth in meters')
    parser.add_argument('--depth_window', type=int, default=3,
                        help='Odd pixel window size for median depth sampling at each corner')
    parser.add_argument('--projection_weight', type=float, default=1.0)
    parser.add_argument('--depth_weight', type=float, default=0.01)
    parser.add_argument('--pose_weight', type=float, default=100.0)
    parser.add_argument('--optimize_frame_poses', action='store_true',
                        help='Also optimize one object-to-camera pose per frame for diagnostics')
    parser.add_argument('--projection_sigma_px', type=float,
                        help='Projection residual sigma in pixels; overrides projection_weight')
    parser.add_argument('--depth_sigma_m', type=float,
                        help='Depth residual sigma in meters; overrides depth_weight')
    parser.add_argument('--pose_trans_sigma_m', type=float,
                        help='Pose translation residual sigma in meters; overrides pose_weight translation scale')
    parser.add_argument('--pose_rot_sigma_deg', type=float,
                        help='Pose rotation residual sigma in degrees; overrides pose_weight rotation scale')
    parser.add_argument('--robust_kernel', choices=['NONE', 'huber', 'soft_l1', 'cauchy'], default='huber')
    parser.add_argument('--kernel_delta', type=float, default=1.0)
    parser.add_argument('--num_iterations', type=int, default=500)
    parser.add_argument('--optimizer_backend', choices=['quaternion_manifold', 'rotvec_scipy'],
                        default='quaternion_manifold')
    parser.add_argument('--compare_with_legacy_backend', action='store_true')
    args = parser.parse_args()
    
    if args.camera_params is None:
        args.camera_params = os.path.join(args.image_dir, 'camera_params.yaml')
    
    config = BoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict=args.aruco_dict
    )
    
    opt_params = OptimizationParams(
        projection_weight=args.projection_weight,
        depth_weight=args.depth_weight,
        pose_weight=args.pose_weight,
        projection_sigma_px=args.projection_sigma_px,
        depth_sigma_m=args.depth_sigma_m,
        pose_trans_sigma_m=args.pose_trans_sigma_m,
        pose_rot_sigma_deg=args.pose_rot_sigma_deg,
        num_iterations=args.num_iterations,
        robust_kernel=args.robust_kernel,
        kernel_delta=args.kernel_delta,
        optimize_frame_poses=args.optimize_frame_poses,
        optimizer_backend=args.optimizer_backend,
        compare_with_legacy_backend=args.compare_with_legacy_backend,
    )
    
    calibrator = HandEyeCalibrator(config, args.camera_params)
    result = calibrator.calibrate(args.image_dir, args.poses_file, args.filter_inconsistent,
                                   args.rot_threshold, args.trans_threshold,
                                   opt_params=opt_params,
                                   setup=args.setup,
                                   depth_dir=args.depth_dir,
                                   use_depth=args.use_depth,
                                   depth_scale=args.depth_scale,
                                   min_depth=args.min_depth,
                                   max_depth=args.max_depth,
                                   depth_window=args.depth_window,
                                   min_filter_corners=args.min_filter_corners,
                                   min_filter_markers=args.min_filter_markers,
                                   max_filter_pnp_reprojection=args.max_filter_pnp_reprojection,
                                   filter_iterations=args.filter_iterations,
                                   filter_mad_scale=args.filter_mad_scale,
                                   pose_trans_unit=args.pose_trans_unit,
                                   pose_rot_order=args.pose_rot_order,
                                   pose_rot_degrees=args.pose_rot_unit == 'deg',
                                   pose_invert=args.pose_direction == 'invert',
                                   pnp_method=args.pnp_method,
                                   diagnostics_output=args.diagnostics_output)
    calibrator.save(result, args.output)
    
    if args.save_detection:
        detection_dir = os.path.join(args.image_dir, 'detection')
        calibrator.save_detections(detection_dir)
    
    if args.visualize:
        output_dir = os.path.join(args.image_dir, 'reprojection')
        calibrator.visualize(result, output_dir)


if __name__ == '__main__':
    main()
