import numpy as np

from st_handeye.io import find_image_files, load_poses_csv


def calibration_result_payload(result, output_path, image_dir=None, poses_file=None, pose_format="sxyz"):
    setup, primary_transform_name, primary_matrix, secondary_transform_name, secondary_matrix = (
        calibration_result_transforms(result)
    )

    pose_error = result.pose_error or {}
    reprojection_error = result.reprojection_error or {}
    reference_consistency = getattr(result, "base_consistency_error", None) or {}
    average_error_mm = float(pose_error.get("translation_mean", 0.0)) * 1000.0
    rotation_error_deg = float(pose_error.get("rotation_mean", 0.0))
    reprojection_px = float(reprojection_error.get("mean", 0.0))
    primary_matrix_rows = format_matrix_rows(primary_matrix)
    secondary_matrix_rows = format_matrix_rows(secondary_matrix)
    frame_errors = frame_error_payloads(result)

    preview_payload = calibration_preview_payload(
        result=result,
        image_dir=image_dir,
        poses_file=poses_file,
        pose_format=pose_format,
        setup=setup,
        primary_transform_name=primary_transform_name,
        primary_matrix=primary_matrix,
        primary_matrix_rows=primary_matrix_rows,
        secondary_transform_name=secondary_transform_name,
        secondary_matrix=secondary_matrix,
        secondary_matrix_rows=secondary_matrix_rows,
        frame_errors=frame_errors,
    )

    return {
        "outputPath": output_path,
        "setup": setup,
        "primaryTransformName": primary_transform_name,
        "primaryMatrixRows": primary_matrix_rows,
        "secondaryTransformName": secondary_transform_name,
        "secondaryMatrixRows": secondary_matrix_rows,
        "matrixRows": primary_matrix_rows,
        "averageErrorMm": average_error_mm,
        "rotationErrorDeg": rotation_error_deg,
        "reprojectionErrorPx": reprojection_px,
        "reprojectionRmsPx": optional_float(reprojection_error.get("rms")),
        "baseConsistencyMeanMm": metric_m_to_mm(reference_consistency.get("mean")),
        "baseConsistencyRmsMm": metric_m_to_mm(reference_consistency.get("rms")),
        "baseConsistencyMaxMm": metric_m_to_mm(reference_consistency.get("max")),
        "baseConsistencyCount": optional_int(reference_consistency.get("count")),
        "numImages": int(result.num_images),
        "numImagesUsed": int(result.num_images_used),
        "filteredImages": [int(index) for index in result.filtered_images],
        "frameErrors": frame_errors,
        **preview_payload,
        "depthUsed": bool(result.depth_used),
        "optimizerBackend": getattr(result, "optimizer_backend", "quaternion_manifold"),
        "comparisonMetrics": getattr(result, "comparison_metrics", None),
        "message": (
            f"{primary_transform_name} 标定完成；有效数据 {result.num_images_used}/{result.num_images}；"
            f"平均平移误差 {average_error_mm:.3f} mm"
        ),
    }


def calibration_result_transforms(result):
    setup = result.setup.value if hasattr(result.setup, "value") else str(result.setup)
    if setup == "eye-in-hand":
        return setup, "T_C2F", result.T_C2F, "T_O2W", result.T_O2W
    return setup, "T_C2W", result.T_C2W, "T_O2F", result.T_O2F


def result_object_to_camera_matrices(result):
    matrices = getattr(result, "T_O2C_opt", None)
    if matrices is None or len(matrices) == 0:
        return getattr(result, "T_O2C_derived", None)
    return matrices


def calibration_preview_payload(
    result,
    image_dir,
    poses_file,
    pose_format,
    setup,
    primary_transform_name,
    primary_matrix,
    primary_matrix_rows,
    secondary_transform_name,
    secondary_matrix,
    secondary_matrix_rows,
    frame_errors,
):
    if not (image_dir and poses_file and primary_matrix is not None and secondary_matrix is not None):
        return {"previewFrames": []}
    try:
        return build_preview_payload(
            image_dir=image_dir,
            poses_file=poses_file,
            setup=setup,
            pose_format=pose_format,
            primary_transform_name=primary_transform_name,
            primary_matrix_rows=primary_matrix_rows,
            secondary_transform_name=secondary_transform_name,
            secondary_matrix_rows=secondary_matrix_rows,
            frame_errors=getattr(result, "frame_errors", None) or frame_errors,
            object_to_camera_matrices=result_object_to_camera_matrices(result),
        )
    except (FileNotFoundError, ValueError):
        return {"previewFrames": []}


def build_preview_payload(
    image_dir,
    poses_file,
    setup,
    pose_format,
    primary_transform_name,
    primary_matrix_rows,
    secondary_transform_name,
    secondary_matrix_rows,
    frame_errors=None,
    object_to_camera_matrices=None,
):
    del primary_transform_name, secondary_transform_name
    setup = str(setup)
    pose_rot_order = scipy_euler_order(pose_format)
    image_paths = find_image_files(image_dir)
    poses = load_poses_csv(
        poses_file,
        trans_unit=0.001,
        rot_deg=True,
        rot_order=pose_rot_order,
        invert=False,
    )
    primary = parse_transform_matrix(primary_matrix_rows)
    secondary = parse_transform_matrix(secondary_matrix_rows)
    object_to_camera_list = [] if object_to_camera_matrices is None else list(object_to_camera_matrices)
    frame_error_map = {
        int(row.get("index", 0)): row
        for row in (frame_errors or [])
        if isinstance(row, dict)
    }
    original_to_local = {}
    local_count = 0
    for row in (frame_errors or []):
        if isinstance(row, dict) and row.get("used", True):
            original_to_local[int(row.get("index", 0))] = local_count
            local_count += 1
    preview_frames = []
    for index in range(min(len(image_paths), len(poses))):
        preview_frames.append(
            preview_frame_payload(
                index=index,
                pose=poses[index],
                image_path=image_paths[index],
                frame_error=frame_error_map.get(index, {}),
                setup=setup,
                primary=primary,
                secondary=secondary,
                object_to_camera=object_to_camera_for_frame(index, original_to_local, object_to_camera_list),
            )
        )
    return {"previewFrames": preview_frames}


def object_to_camera_for_frame(index, original_to_local, object_to_camera_list):
    local_idx = original_to_local.get(index)
    if local_idx is None or local_idx >= len(object_to_camera_list):
        return None
    return object_to_camera_list[local_idx]


def preview_frame_payload(index, pose, image_path, frame_error, setup, primary, secondary, object_to_camera):
    camera_in_base, board_in_base, board_in_focus = preview_frame_transforms(
        setup, pose, primary, secondary, object_to_camera
    )
    resolved_image_path = str(frame_error.get("image_path") or frame_error.get("imagePath") or image_path)
    return {
        "index": int(index),
        "imagePath": resolved_image_path,
        "used": bool(frame_error.get("used", True)),
        "cameraInBase": matrix_to_rows(camera_in_base),
        "boardInBase": matrix_to_rows(board_in_base),
        "boardInFocus": matrix_to_rows(board_in_focus),
    }


def preview_frame_transforms(setup, pose, primary, secondary, object_to_camera):
    pose = np.asarray(pose, dtype=np.float64)
    primary = np.asarray(primary, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    measured = None if object_to_camera is None else np.asarray(object_to_camera, dtype=np.float64)

    if str(setup) == "eye-in-hand":
        camera_in_base = pose @ primary
        board_in_base = camera_in_base @ measured if measured is not None else secondary
        return camera_in_base, board_in_base, board_in_base

    camera_in_base = primary
    board_in_base = pose @ secondary
    board_in_focus = np.linalg.inv(pose) @ primary @ measured if measured is not None else secondary
    return camera_in_base, board_in_base, board_in_focus


def format_matrix_rows(matrix):
    if matrix is None:
        return []
    array = np.asarray(matrix, dtype=float)
    return [", ".join(f"{value:.7f}" for value in row) for row in array]


def matrix_to_rows(matrix):
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=float)]


def parse_transform_matrix(matrix_rows):
    if matrix_rows is None:
        raise ValueError("transform matrix is required")
    if isinstance(matrix_rows, str):
        values = parse_float_list(matrix_rows)
        if len(values) != 16:
            raise ValueError(f"expected 16 matrix values, got {len(values)}")
        return np.asarray(values, dtype=np.float64).reshape(4, 4)
    rows = []
    for row in matrix_rows:
        if isinstance(row, str):
            rows.append(parse_float_list(row))
        else:
            rows.append([float(value) for value in row])
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform matrix must be 4x4, got {matrix.shape}")
    return matrix


def parse_float_list(value):
    if isinstance(value, str):
        parts = value.replace("\n", ",").split(",")
        return [float(part.strip()) for part in parts if part.strip()]
    return [float(item) for item in value]


def scipy_euler_order(pose_format):
    pose_format = pose_format or "sxyz"
    if len(pose_format) == 4 and pose_format[0] in ("s", "r"):
        axes = pose_format[1:]
        return axes.lower() if pose_format[0] == "s" else axes.upper()
    return pose_format


def frame_error_payloads(result):
    setup = getattr(result, "setup", "eye-in-hand")
    rows = getattr(result, "frame_errors", None)
    reference_rows = reference_consistency_rows(result)
    if rows is not None:
        return [
            frame_error_payload(
                frame_error_row(result_row=row, reference_row=reference_rows[index] if index < len(reference_rows) else None),
                setup=setup,
            )
            for index, row in enumerate(rows)
        ]

    reprojection_error = result.reprojection_error or {}
    pose_error = result.pose_error or {}
    derived = reprojection_error.get("per_image", [])
    optimized = reprojection_error.get("per_image_optimized", [])
    poses = pose_error.get("per_image", [])
    payloads = []
    for index in range(max(len(derived), len(optimized), len(poses), len(reference_rows))):
        pose_row = poses[index] if index < len(poses) and isinstance(poses[index], dict) else {}
        reference_row = reference_rows[index] if index < len(reference_rows) else None
        payloads.append(frame_error_payload(
            frame_error_row(
                result_row={
                    "index": index,
                    "image_path": "",
                    "used": True,
                    "reprojection_error_px": derived[index] if index < len(derived) else None,
                    "optimized_reprojection_error_px": optimized[index] if index < len(optimized) else None,
                    "translation_error": pose_row.get("translation"),
                    "rotation_error_deg": pose_row.get("rotation"),
                },
                reference_row=reference_row,
            ),
            setup=setup,
        ))
    return payloads


def reference_consistency_rows(result):
    reference_consistency = getattr(result, "base_consistency_error", None)
    if not isinstance(reference_consistency, dict):
        return []
    per_image = reference_consistency.get("per_image", [])
    return per_image if isinstance(per_image, list) else []


def merge_reference_consistency(row, reference_row):
    if not isinstance(reference_row, dict):
        return row
    merged = dict(row)
    merged.setdefault("base_consistency_mean", reference_row.get("mean"))
    merged.setdefault("base_consistency_rms", reference_row.get("rms"))
    merged.setdefault("base_consistency_max", reference_row.get("max"))
    merged.setdefault("base_consistency_count", reference_row.get("count"))
    return merged


def frame_error_row(result_row, reference_row):
    return merge_reference_consistency(result_row, reference_row)


def base_consistency_rows(result):
    return reference_consistency_rows(result)


def merge_base_consistency(row, base_row):
    return merge_reference_consistency(row, base_row)


def frame_error_payload(row, setup="eye-in-hand"):
    translation_error = row.get("translation_error")
    reprojection_mean = row.get("reprojection_mean_px", row.get("reprojection_error_px"))
    reference_mean = row.get("reference_reprojection_mean_px", row.get("optimized_reprojection_error_px"))
    return {
        "index": int(row.get("index", 0)),
        "imagePath": str(row.get("image_path") or ""),
        "used": bool(row.get("used", True)),
        "cornerCount": optional_int(row.get("corner_count")),
        "reprojectionMeanPx": optional_float(reprojection_mean),
        "reprojectionRmsPx": optional_float(row.get("reprojection_rms_px")),
        "reprojectionMaxPx": optional_float(row.get("reprojection_max_px")),
        "referenceReprojectionMeanPx": optional_float(reference_mean),
        "referenceReprojectionRmsPx": optional_float(row.get("reference_reprojection_rms_px")),
        "referenceReprojectionMaxPx": optional_float(row.get("reference_reprojection_max_px")),
        "reprojectionErrorPx": optional_float(reprojection_mean),
        "optimizedReprojectionErrorPx": optional_float(reference_mean),
        "translationErrorMm": None if translation_error is None else float(translation_error) * 1000.0,
        "rotationErrorDeg": optional_float(row.get("rotation_error_deg")),
        "baseConsistencyMeanMm": metric_m_to_mm(row.get("base_consistency_mean")),
        "baseConsistencyRmsMm": metric_m_to_mm(row.get("base_consistency_rms")),
        "baseConsistencyMaxMm": metric_m_to_mm(row.get("base_consistency_max")),
        "baseConsistencyCount": optional_int(row.get("base_consistency_count")),
        "errorMeanings": frame_error_meanings(setup),
    }


def optional_float(value):
    return None if value is None else float(value)


def optional_int(value):
    return None if value is None else int(value)


def metric_m_to_mm(value):
    return None if value is None else float(value) * 1000.0


def frame_error_meanings(setup="eye-in-hand"):
    setup_value = setup.value if hasattr(setup, "value") else str(setup)
    consistency_frame = "法兰末端坐标系" if setup_value == "eye-to-hand" else "机械臂底座坐标系"
    return {
        "cornerCount": {
            "unit": "count",
            "description": "该帧参与误差统计的 ChArUco 角点数量。",
        },
        "reprojectionMeanPx": {
            "unit": "px",
            "description": "全局手眼结果推导出的 T_O2C 对该帧所有角点的平均二维重投影距离。",
        },
        "reprojectionRmsPx": {
            "unit": "px",
            "description": "全局手眼结果推导投影的角点重投影 RMS，较 mean 更强调大误差角点。",
        },
        "reprojectionMaxPx": {
            "unit": "px",
            "description": "全局手眼结果推导投影下该帧单个角点的最大重投影距离。",
        },
        "referenceReprojectionMeanPx": {
            "unit": "px",
            "description": "每帧参考位姿对该帧角点的平均重投影距离，用于判断视觉检测/参考位姿本身质量。",
        },
        "referenceReprojectionRmsPx": {
            "unit": "px",
            "description": "每帧参考位姿的角点重投影 RMS。",
        },
        "referenceReprojectionMaxPx": {
            "unit": "px",
            "description": "每帧参考位姿下单个角点的最大重投影距离。",
        },
        "translationErrorMm": {
            "unit": "mm",
            "description": "参考位姿与全局手眼链路推导位姿之间的平移残差。",
        },
        "rotationErrorDeg": {
            "unit": "deg",
            "description": "参考位姿与全局手眼链路推导位姿之间的最小旋转角残差。",
        },
        "baseConsistencyRmsMm": {
            "unit": "mm",
            "description": f"该帧角点转换到{consistency_frame}后，与同一角点跨帧均值之间的 RMS 距离，作为三维偏差结果。",
        },
    }
