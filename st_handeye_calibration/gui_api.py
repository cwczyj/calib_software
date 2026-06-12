#!/usr/bin/env python3
import argparse
import json
import os
import sys

import cv2
import numpy as np

from result_payloads import build_preview_payload, calibration_result_payload
from st_handeye import BoardConfig, HandEyeCalibrator, OptimizationParams
from st_handeye.board import CharucoBoard
from st_handeye.camera import CameraModel
from st_handeye.depth import load_depth_image, sample_depth_at_corners


def detect_charuco_image(
    image_path,
    camera_params=None,
    output_dir=None,
    squares_x=11,
    squares_y=8,
    square_length=0.014,
    marker_length=0.010,
    aruco_dict="DICT_5X5_100",
    depth_path=None,
    camera_intrinsics=None,
):
    if camera_params is None and camera_intrinsics is None:
        camera_params = os.path.join(os.path.dirname(image_path), "camera_params.yaml")

    config = BoardConfig(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        aruco_dict=aruco_dict,
    )
    board = CharucoBoard(config)
    camera = make_camera_model(camera_params, camera_intrinsics)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    detection = board.detect(image, camera)
    detection.image_path = image_path

    output_dir = output_dir or os.path.join(os.path.dirname(image_path), "detection")
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0].replace("_Color", "")
    output_path = os.path.join(output_dir, f"detection_{stem}.png")

    overlay = camera.undistort(image)
    if detection.success:
        for point, corner_id in zip(detection.corners_2d.T, detection.corner_ids):
            x, y = int(point[0]), int(point[1])
            cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                str(int(corner_id)),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )
        axes_drawn = draw_charuco_board_axes(overlay, board, detection, camera, square_length)
        message = "ok"
    else:
        axes_drawn = False
        message = "ChArUco detection failed"

    camera_points_by_id = {}
    if detection.success and depth_path:
        depth_image = load_depth_image(depth_path, image_shape=image.shape[:2])
        if depth_image is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        depth_sample = sample_depth_at_corners(
            depth_image,
            detection.corners_2d,
            detection.corner_ids,
            camera.K,
        )
        camera_points_by_id = {
            int(corner_id): depth_sample.points_camera[:, index]
            for index, corner_id in enumerate(depth_sample.corner_ids)
        }

    corner_rows = []
    if detection.success:
        for point, corner_id in zip(detection.corners_2d.T, detection.corner_ids):
            camera_point = camera_points_by_id.get(int(corner_id))
            corner_rows.append({
                "id": int(corner_id),
                "imagePoint": [float(point[0]), float(point[1])],
                "cameraPoint": None if camera_point is None else [
                    float(camera_point[0]),
                    float(camera_point[1]),
                    float(camera_point[2]),
                ],
            })

    status = f"Corners: {detection.num_corners} | Markers: {detection.num_markers} | Axes: {'drawn' if axes_drawn else 'skipped'}"
    cv2.putText(
        overlay,
        status,
        (10, max(24, overlay.shape[0] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(output_path, overlay)

    return {
        "imagePath": image_path,
        "outputPath": output_path,
        "success": bool(detection.success),
        "numCorners": int(detection.num_corners),
        "numMarkers": int(detection.num_markers),
        "message": message,
        "axesDrawn": axes_drawn,
        "cornerRows": corner_rows,
    }


def draw_charuco_board_axes(overlay, board, detection, camera, square_length):
    try:
        if detection.num_corners < 4:
            return False
        pattern = np.asarray(board.pattern_3d, dtype=np.float64)[:, detection.corner_ids]
        transform, ok = camera.solve_pnp(
            pattern,
            detection.corners_2d,
            pnp_method="iterative",
            assume_undistorted=True,
        )
        if not ok:
            return False
        rvec, _ = cv2.Rodrigues(transform[:3, :3])
        tvec = transform[:3, 3].reshape(3, 1)
        zero_distortion = np.zeros_like(camera.D, dtype=np.float64)
        axis_length = max(float(square_length) * 3.0, 0.02)
        cv2.drawFrameAxes(
            overlay,
            camera.K.astype(np.float32),
            zero_distortion.astype(np.float32),
            rvec.astype(np.float32),
            tvec.astype(np.float32),
            axis_length,
        )
        draw_projected_axes_overlay(overlay, camera, transform, axis_length)
        return True
    except (AttributeError, cv2.error, ValueError, IndexError):
        return False


def draw_projected_axes_overlay(overlay, camera, transform, axis_length):
    points = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
    ], dtype=np.float64).T
    projected = camera.project(points, transform).T
    if not np.all(np.isfinite(projected)):
        return
    origin = tuple(np.round(projected[0]).astype(int))
    axes = [
        ("X", tuple(np.round(projected[1]).astype(int)), (0, 0, 255)),
        ("Y", tuple(np.round(projected[2]).astype(int)), (0, 210, 0)),
        ("Z", tuple(np.round(projected[3]).astype(int)), (255, 80, 0)),
    ]
    cv2.circle(overlay, origin, 6, (255, 255, 255), -1)
    cv2.circle(overlay, origin, 6, (20, 20, 20), 1)
    for label, end, color in axes:
        cv2.line(overlay, origin, end, color, 4, cv2.LINE_AA)
        cv2.circle(overlay, end, 5, color, -1)
        cv2.putText(
            overlay,
            label,
            (end[0] + 7, end[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def make_camera_model(camera_params=None, camera_intrinsics=None):
    if camera_intrinsics is not None:
        K = np.array([
            [float(camera_intrinsics["fx"]), 0.0, float(camera_intrinsics["cx"])],
            [0.0, float(camera_intrinsics["fy"]), float(camera_intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ])
        distortion = None
        for key in ("distortionCoefficients", "distortion_coefficients", "distortion"):
            if key in camera_intrinsics and camera_intrinsics[key] is not None:
                distortion = camera_intrinsics[key]
                break
        if distortion is None:
            distortion = np.zeros(5)
        return CameraModel(K, np.asarray(distortion, dtype=np.float64).reshape(-1))
    return CameraModel.from_yaml(camera_params)


def run_calibration(payload):
    image_dir = payload["imageDir"]
    poses_file = payload["posesFile"]
    setup = payload.get("setup", "eye-in-hand")
    pose_format = payload.get("poseFormat", "sxyz")
    camera_params = payload.get("cameraParams")
    camera_intrinsics = payload.get("cameraIntrinsics")
    camera_model = None
    output_path = payload.get("outputPath") or os.path.join(image_dir, "calibration_result.yaml")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if camera_params is None:
        if camera_intrinsics is not None:
            camera_model = make_camera_model(camera_intrinsics=camera_intrinsics)
        else:
            camera_params = os.path.join(image_dir, "camera_params.yaml")

    config = BoardConfig(
        squares_x=payload.get("squaresX", 11),
        squares_y=payload.get("squaresY", 8),
        square_length=payload.get("squareLength", 0.014),
        marker_length=payload.get("markerLength", 0.010),
        aruco_dict=payload.get("arucoDict", "DICT_5X5_100"),
    )
    opt_params = OptimizationParams(
        num_iterations=payload.get("numIterations", 500),
        optimizer_backend=payload.get("optimizerBackend", "quaternion_manifold"),
        compare_with_legacy_backend=payload.get("compareWithLegacyBackend", False),
    )
    pose_rot_order = scipy_euler_order(pose_format)
    filter_inconsistent = payload.get("filterInconsistent")
    if filter_inconsistent is None:
        filter_inconsistent = False if "excludedImageIndices" in payload else True

    calibrator = HandEyeCalibrator(config, camera_params_file=camera_params, camera=camera_model)
    result = calibrator.calibrate(
        image_dir,
        poses_file,
        filter_inconsistent=filter_inconsistent,
        opt_params=opt_params,
        setup=setup,
        use_depth=payload.get("useDepth", "off"),
        excluded_image_indices=payload.get("excludedImageIndices", []),
        pose_trans_unit=payload.get("poseTransUnit", 0.001),
        pose_rot_order=pose_rot_order,
        pose_rot_degrees=payload.get("poseRotUnit", "deg") == "deg",
        pose_invert=payload.get("poseDirection", "as-is") == "invert",
        pnp_method=payload.get("pnpMethod", "iterative"),
    )
    calibrator.save(result, output_path)
    calibrator.save_detections(os.path.join(image_dir, "detection"))
    return calibration_result_payload(
        result,
        output_path,
        image_dir=image_dir,
        poses_file=poses_file,
        pose_format=pose_format,
    )


def scipy_euler_order(pose_format):
    pose_format = pose_format or "sxyz"
    if len(pose_format) == 4 and pose_format[0] in ("s", "r"):
        axes = pose_format[1:]
        return axes.lower() if pose_format[0] == "s" else axes.upper()
    return pose_format


def detect_charuco(args):
    return detect_charuco_image(
        args.image_path,
        camera_params=args.camera_params,
        output_dir=args.output_dir,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict=args.aruco_dict,
        depth_path=args.depth_path,
        camera_intrinsics=args.camera_intrinsics,
    )


def run_calibration_cli(args):
    camera_intrinsics = None
    if all(value is not None for value in [args.cx, args.cy, args.fx, args.fy]):
        camera_intrinsics = {"cx": args.cx, "cy": args.cy, "fx": args.fx, "fy": args.fy}
        if args.distortion_coefficients:
            camera_intrinsics["distortionCoefficients"] = parse_float_list(args.distortion_coefficients)
    return run_calibration({
        "imageDir": args.image_dir,
        "posesFile": args.poses_file,
        "cameraParams": args.camera_params,
        "cameraIntrinsics": camera_intrinsics,
        "setup": args.setup,
        "poseFormat": args.pose_format,
        "useDepth": args.use_depth,
        "squaresX": args.squares_x,
        "squaresY": args.squares_y,
        "squareLength": args.square_length,
        "markerLength": args.marker_length,
        "arucoDict": args.aruco_dict,
        "excludedImageIndices": parse_index_list(args.excluded_image_indices),
        "outputPath": args.output,
    })


def build_preview_cli(args):
    return build_preview_payload(
        image_dir=args.image_dir,
        poses_file=args.poses_file,
        setup=args.setup,
        pose_format=args.pose_format,
        primary_transform_name=args.primary_transform_name,
        primary_matrix_rows=args.primary_matrix,
        secondary_transform_name=args.secondary_transform_name,
        secondary_matrix_rows=args.secondary_matrix,
    )


def parse_index_list(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="GUI helpers for hand-eye calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect-charuco")
    detect_parser.add_argument("image_path")
    detect_parser.add_argument("-c", "--camera_params")
    detect_parser.add_argument("--output_dir")
    detect_parser.add_argument("--squares_x", type=int, default=11)
    detect_parser.add_argument("--squares_y", type=int, default=8)
    detect_parser.add_argument("--square_length", type=float, default=0.014)
    detect_parser.add_argument("--marker_length", type=float, default=0.010)
    detect_parser.add_argument("--aruco_dict", default="DICT_5X5_100")
    detect_parser.add_argument("--depth_path")
    detect_parser.add_argument("--cx", type=float)
    detect_parser.add_argument("--cy", type=float)
    detect_parser.add_argument("--fx", type=float)
    detect_parser.add_argument("--fy", type=float)
    detect_parser.add_argument("--distortion_coefficients")

    calibration_parser = subparsers.add_parser("run-calibration")
    calibration_parser.add_argument("image_dir")
    calibration_parser.add_argument("poses_file")
    calibration_parser.add_argument("-c", "--camera_params")
    calibration_parser.add_argument("--setup", default="eye-in-hand")
    calibration_parser.add_argument("--pose_format", default="sxyz")
    calibration_parser.add_argument("--use_depth", default="off")
    calibration_parser.add_argument("--output")
    calibration_parser.add_argument("--cx", type=float)
    calibration_parser.add_argument("--cy", type=float)
    calibration_parser.add_argument("--fx", type=float)
    calibration_parser.add_argument("--fy", type=float)
    calibration_parser.add_argument("--distortion_coefficients")
    calibration_parser.add_argument("--squares_x", type=int, default=11)
    calibration_parser.add_argument("--squares_y", type=int, default=8)
    calibration_parser.add_argument("--square_length", type=float, default=0.014)
    calibration_parser.add_argument("--marker_length", type=float, default=0.010)
    calibration_parser.add_argument("--aruco_dict", default="DICT_5X5_100")
    calibration_parser.add_argument("--excluded_image_indices", default="")

    preview_parser = subparsers.add_parser("build-preview")
    preview_parser.add_argument("image_dir")
    preview_parser.add_argument("poses_file")
    preview_parser.add_argument("--setup", default="eye-in-hand")
    preview_parser.add_argument("--pose_format", default="sxyz")
    preview_parser.add_argument("--primary_transform_name", required=True)
    preview_parser.add_argument("--secondary_transform_name", required=True)
    preview_parser.add_argument("--primary_matrix", required=True)
    preview_parser.add_argument("--secondary_matrix", required=True)

    args = parser.parse_args()
    if args.command == "detect-charuco" and all(value is not None for value in [args.cx, args.cy, args.fx, args.fy]):
        args.camera_intrinsics = {"cx": args.cx, "cy": args.cy, "fx": args.fx, "fy": args.fy}
        if args.distortion_coefficients:
            args.camera_intrinsics["distortionCoefficients"] = parse_float_list(args.distortion_coefficients)
    else:
        args.camera_intrinsics = None
    if args.command == "detect-charuco":
        result = detect_charuco(args)
    elif args.command == "run-calibration":
        result = run_calibration_cli(args)
    elif args.command == "build-preview":
        result = build_preview_cli(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
