import json
import socket
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np
import pytest

from gui_api import detect_charuco_image, run_calibration
from api_server import app


def test_detect_charuco_image_uses_st_handeye_and_writes_overlay(tmp_path):
    import gui_api

    image_path = tmp_path / "001_Color.png"
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    class FakeDetection:
        success = True
        num_corners = 24
        num_markers = 12
        corners_2d = np.vstack([
            np.linspace(1.0, 8.0, 24),
            np.linspace(1.0, 6.0, 24),
        ])
        corner_ids = np.arange(24)

    class FakeBoard:
        def __init__(self, config):
            self.config = config

        def detect(self, image, camera):
            return FakeDetection()

    class FakeCamera:
        def __init__(self):
            self.K = np.eye(3)

        def undistort(self, image):
            return image.copy()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gui_api, "CharucoBoard", FakeBoard)
    monkeypatch.setattr(gui_api, "make_camera_model", lambda *args, **kwargs: FakeCamera())
    try:
        result = detect_charuco_image(
            str(image_path),
            output_dir=str(tmp_path),
            camera_intrinsics={"cx": 1.0, "cy": 1.0, "fx": 100.0, "fy": 100.0},
        )
    finally:
        monkeypatch.undo()

    assert result["success"] is True
    assert result["numCorners"] > 20
    assert result["numMarkers"] > 0
    assert result["axesDrawn"] is False
    assert len(result["cornerRows"]) == result["numCorners"]
    assert set(result["cornerRows"][0]) == {"id", "imagePoint", "cameraPoint"}
    assert len(result["cornerRows"][0]["imagePoint"]) == 2
    assert result["cornerRows"][0]["cameraPoint"] is None
    assert result["outputPath"].endswith("detection_001.png")
    assert (tmp_path / "detection_001.png").exists()


def test_detect_charuco_image_draws_charuco_board_axes(tmp_path, monkeypatch):
    import gui_api

    image_path = tmp_path / "001_Color.png"
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    calls = {"axes": 0}

    class FakeDetection:
        success = True
        num_corners = 6
        num_markers = 4
        corners_2d = np.array([
            [20.0, 40.0, 60.0, 20.0, 40.0, 60.0],
            [20.0, 20.0, 20.0, 40.0, 40.0, 40.0],
        ])
        corner_ids = np.arange(6)

    class FakeBoard:
        def __init__(self, config):
            self.config = config
            self.pattern_3d = np.array([
                [0.02, 0.04, 0.06, 0.02, 0.04, 0.06],
                [0.02, 0.02, 0.02, 0.04, 0.04, 0.04],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])

        def detect(self, image, camera):
            return FakeDetection()

    class FakeCamera:
        K = np.eye(3, dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)

        def undistort(self, image):
            return image.copy()

        def solve_pnp(self, pts3d, pts2d, pnp_method="iterative", assume_undistorted=False):
            transform = np.eye(4, dtype=np.float64)
            transform[2, 3] = 0.5
            return transform, True

        def project(self, points_3d, object2eye=None):
            pts = np.asarray(points_3d, dtype=np.float64)
            if object2eye is not None:
                pts = object2eye[:3, :3] @ pts + object2eye[:3, 3:4]
            return pts[:2] / pts[2:]

    def fake_draw_frame_axes(*args, **kwargs):
        calls["axes"] += 1

    monkeypatch.setattr(gui_api, "CharucoBoard", FakeBoard)
    monkeypatch.setattr(gui_api, "make_camera_model", lambda *args, **kwargs: FakeCamera())
    monkeypatch.setattr(gui_api.cv2, "drawFrameAxes", fake_draw_frame_axes)

    result = detect_charuco_image(
        str(image_path),
        output_dir=str(tmp_path),
        camera_intrinsics={"cx": 1.0, "cy": 1.0, "fx": 100.0, "fy": 100.0},
    )

    assert result["success"] is True
    assert result["axesDrawn"] is True
    assert calls["axes"] == 1


def test_api_server_exposes_fastapi_app():
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)


def test_api_server_run_calibration_endpoint_returns_structured_result(monkeypatch):
    import api_server

    def fake_run_calibration(payload):
        assert payload["imageDir"] == "/data/session"
        return {
            "outputPath": "/data/session/calibration_result.yaml",
            "setup": "eye-in-hand",
            "primaryTransformName": "T_C2F",
            "matrixRows": ["1.0000000, 0.0000000, 0.0000000, 0.0000000"],
            "averageErrorMm": 1.2,
            "rotationErrorDeg": 0.4,
            "reprojectionErrorPx": 0.3,
            "numImages": 6,
            "numImagesUsed": 5,
            "filteredImages": [2],
            "depthUsed": False,
            "message": "done",
        }

    monkeypatch.setattr(api_server, "run_calibration", fake_run_calibration)
    result = api_server.run_calibration_endpoint({
        "imageDir": "/data/session",
        "posesFile": "/data/session/pose.txt",
    })

    assert result["primaryTransformName"] == "T_C2F"
    assert result["averageErrorMm"] == 1.2


def test_api_server_detect_charuco_endpoint_returns_overlay(tmp_path):
    import api_server

    expected = {
        "imagePath": str(tmp_path / "001_Color.png"),
        "outputPath": str(tmp_path / "detection_001.png"),
        "success": True,
        "numCorners": 24,
        "numMarkers": 12,
        "message": "ok",
        "cornerRows": [{"id": 7, "imagePoint": [2.0, 1.0], "cameraPoint": [0.0, 0.0, 1.0]}],
    }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(api_server, "detect_charuco_image", lambda *args, **kwargs: expected)
    try:
        result = api_server.detect_charuco({
            "imagePath": str(tmp_path / "001_Color.png"),
            "outputDir": str(tmp_path),
            "depthPath": str(tmp_path / "001_Depth.png"),
            "cameraIntrinsics": {"cx": 640.0, "cy": 360.0, "fx": 600.0, "fy": 600.0},
        })
    finally:
        monkeypatch.undo()

    assert result == expected


def test_detect_charuco_image_uses_aligned_depth_for_camera_points(tmp_path):
    import gui_api

    image_path = tmp_path / "001_Color.png"
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    depth_path = tmp_path / "001_Depth.png"
    cv2.imwrite(str(depth_path), np.full(image.shape[:2], 1000, dtype=np.uint16))

    class FakeDetection:
        success = True
        num_corners = 1
        num_markers = 1
        corners_2d = np.array([[2.0], [1.0]])
        corner_ids = np.array([7])

    class FakeBoard:
        def __init__(self, config):
            self.config = config

        def detect(self, image, camera):
            return FakeDetection()

    class FakeCamera:
        def __init__(self):
            self.K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])

        def undistort(self, image):
            return image.copy()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gui_api, "CharucoBoard", FakeBoard)
    monkeypatch.setattr(gui_api, "make_camera_model", lambda *args, **kwargs: FakeCamera())
    try:
        result = detect_charuco_image(
            str(image_path),
            output_dir=str(tmp_path),
            depth_path=str(depth_path),
            camera_intrinsics={"cx": 1.0, "cy": 1.0, "fx": 100.0, "fy": 100.0},
        )
    finally:
        monkeypatch.undo()

    assert result["success"] is True
    assert result["cornerRows"]
    assert all("cameraPoint" in row for row in result["cornerRows"])
    assert any(row["cameraPoint"] is not None for row in result["cornerRows"])
    point = next(row["cameraPoint"] for row in result["cornerRows"] if row["cameraPoint"] is not None)
    assert len(point) == 3
    assert point[2] == 1.0


def test_detect_charuco_image_accepts_raw_depth_with_rgb_shape(tmp_path):
    import gui_api

    image_path = tmp_path / "001_Color.png"
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    raw_path = tmp_path / "001.raw"
    np.full(image.shape[:2], 1000, dtype=np.uint16).tofile(raw_path)

    class FakeDetection:
        success = True
        num_corners = 1
        num_markers = 1
        corners_2d = np.array([[2.0], [1.0]])
        corner_ids = np.array([7])

    class FakeBoard:
        def __init__(self, config):
            self.config = config

        def detect(self, image, camera):
            return FakeDetection()

    class FakeCamera:
        def __init__(self):
            self.K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])

        def undistort(self, image):
            return image.copy()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gui_api, "CharucoBoard", FakeBoard)
    monkeypatch.setattr(gui_api, "make_camera_model", lambda *args, **kwargs: FakeCamera())
    try:
        result = detect_charuco_image(
            str(image_path),
            output_dir=str(tmp_path),
            depth_path=str(raw_path),
            camera_intrinsics={"cx": 1.0, "cy": 1.0, "fx": 100.0, "fy": 100.0},
        )
    finally:
        monkeypatch.undo()

    assert result["success"] is True
    assert any(row["cameraPoint"] is not None for row in result["cornerRows"])


def test_run_calibration_returns_structured_result(monkeypatch, tmp_path):
    calls = {}

    class FakeCalibrator:
        def __init__(self, config, camera_params_file=None, camera=None):
            calls["config"] = config
            calls["camera_params_file"] = camera_params_file
            calls["camera"] = camera

        def calibrate(self, image_dir, poses_file, **kwargs):
            calls["image_dir"] = image_dir
            calls["poses_file"] = poses_file
            calls["kwargs"] = kwargs
            return SimpleCalibrationResult()

        def save(self, result, output_path):
            calls["saved"] = (result, output_path)
            with open(output_path, "w") as f:
                f.write("saved: true\n")

        def save_detections(self, output_dir):
            calls["detection_dir"] = output_dir

    monkeypatch.setattr("gui_api.HandEyeCalibrator", FakeCalibrator)
    output_path = tmp_path / "calibration_result.yaml"

    result = run_calibration({
        "imageDir": "/data/session",
        "posesFile": "/data/session/pose.txt",
        "cameraIntrinsics": {"cx": 640.0, "cy": 360.0, "fx": 600.0, "fy": 600.0},
        "setup": "eye-to-hand",
        "marker": "charuco",
        "poseFormat": "sxyz",
        "excludedImageIndices": [1, 4],
        "outputPath": str(output_path),
    })

    assert result["outputPath"] == str(output_path)
    assert result["setup"] == "eye-to-hand"
    assert result["primaryTransformName"] == "T_C2W"
    assert result["matrixRows"][0] == "1.0000000, 0.0000000, 0.0000000, 0.1000000"
    assert result["averageErrorMm"] == pytest.approx(2.5)
    assert result["numImages"] == 5
    assert result["numImagesUsed"] == 4
    assert result["filteredImages"] == [3]
    assert len(result["frameErrors"]) == 2
    assert result["frameErrors"][0]["index"] == 0
    assert result["frameErrors"][0]["imagePath"] == "001_Color.png"
    assert result["frameErrors"][0]["reprojectionErrorPx"] == 0.21
    assert result["frameErrors"][0]["optimizedReprojectionErrorPx"] == 0.18
    assert result["frameErrors"][0]["translationErrorMm"] == 1.2
    assert result["frameErrors"][0]["rotationErrorDeg"] == 0.3
    assert result["frameErrors"][1]["index"] == 1
    assert result["frameErrors"][1]["imagePath"] == "002_Color.png"
    assert result["frameErrors"][1]["reprojectionErrorPx"] == 0.63
    assert result["frameErrors"][1]["optimizedReprojectionErrorPx"] == 0.44
    assert result["frameErrors"][1]["translationErrorMm"] == 3.8
    assert result["frameErrors"][1]["rotationErrorDeg"] == 1.1
    assert output_path.exists()
    assert calls["kwargs"]["setup"] == "eye-to-hand"
    assert calls["kwargs"]["pose_rot_order"] == "xyz"
    assert calls["kwargs"]["pose_invert"] is False
    assert calls["kwargs"]["use_depth"] == "off"
    assert calls["kwargs"]["excluded_image_indices"] == [1, 4]
    assert calls["camera_params_file"] is None
    assert calls["camera"] is not None
    assert calls["detection_dir"] == "/data/session/detection"
    assert calls["kwargs"]["opt_params"].optimizer_backend == "quaternion_manifold"


def test_run_calibration_can_forward_compare_mode(monkeypatch, tmp_path):
    calls = {}

    class FakeCalibrator:
        def __init__(self, config, camera_params_file=None, camera=None):
            pass

        def calibrate(self, image_dir, poses_file, **kwargs):
            calls["opt_params"] = kwargs["opt_params"]
            return SimpleCalibrationResult()

        def save(self, result, output_path):
            with open(output_path, "w") as f:
                f.write("saved: true\n")

        def save_detections(self, output_dir):
            pass

    monkeypatch.setattr("gui_api.HandEyeCalibrator", FakeCalibrator)

    run_calibration({
        "imageDir": "/data/session",
        "posesFile": "/data/session/pose.txt",
        "cameraIntrinsics": {"cx": 640.0, "cy": 360.0, "fx": 600.0, "fy": 600.0},
        "outputPath": str(tmp_path / "calibration_result.yaml"),
        "compareWithLegacyBackend": True,
    })

    assert calls["opt_params"].compare_with_legacy_backend is True


def test_run_calibration_disables_default_filtering_when_excluding_selected_images(monkeypatch, tmp_path):
    calls = {}

    class FakeCalibrator:
        def __init__(self, config, camera_params_file=None, camera=None):
            pass

        def calibrate(self, image_dir, poses_file, **kwargs):
            calls["kwargs"] = kwargs
            return SimpleCalibrationResult()

        def save(self, result, output_path):
            with open(output_path, "w") as f:
                f.write("saved: true\n")

        def save_detections(self, output_dir):
            pass

    monkeypatch.setattr("gui_api.HandEyeCalibrator", FakeCalibrator)
    output_path = tmp_path / "calibration_result.yaml"

    run_calibration({
        "imageDir": "/data/session",
        "posesFile": "/data/session/pose.txt",
        "cameraIntrinsics": {"cx": 640.0, "cy": 360.0, "fx": 600.0, "fy": 600.0},
        "excludedImageIndices": [1, 4],
        "outputPath": str(output_path),
    })

    assert calls["kwargs"]["filter_inconsistent"] is False


def test_run_calibration_uses_inline_intrinsics_without_saving_camera_params(monkeypatch, tmp_path):
    calls = {}

    class FakeCalibrator:
        def __init__(self, config, camera_params_file=None, camera=None):
            calls["camera_params_file"] = camera_params_file
            calls["camera"] = camera

        def calibrate(self, image_dir, poses_file, **kwargs):
            return SimpleCalibrationResult()

        def save(self, result, output_path):
            with open(output_path, "w") as f:
                f.write("saved: true\n")

        def save_detections(self, output_dir):
            calls["detection_dir"] = output_dir

    monkeypatch.setattr("gui_api.HandEyeCalibrator", FakeCalibrator)
    image_dir = tmp_path / "session"
    image_dir.mkdir()
    output_path = image_dir / "calibration_result.yaml"

    run_calibration({
        "imageDir": str(image_dir),
        "posesFile": str(image_dir / "poses.csv"),
        "cameraIntrinsics": {
            "cx": 640.0,
            "cy": 360.0,
            "fx": 600.0,
            "fy": 610.0,
            "distortionCoefficients": [0.1, -0.2, 0.01, 0.02, 0.03],
        },
        "outputPath": str(output_path),
    })

    assert not (image_dir / "camera_params.yaml").exists()
    assert calls["camera_params_file"] is None
    assert calls["camera"] is not None
    np.testing.assert_allclose(
        calls["camera"].K,
        np.array([[600.0, 0.0, 640.0], [0.0, 610.0, 360.0], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(calls["camera"].D, np.array([0.1, -0.2, 0.01, 0.02, 0.03]))


def test_make_camera_model_uses_inline_distortion_coefficients():
    from gui_api import make_camera_model

    camera = make_camera_model(camera_intrinsics={
        "cx": 640.0,
        "cy": 360.0,
        "fx": 600.0,
        "fy": 610.0,
        "distortionCoefficients": [0.1, -0.2, 0.01, 0.02, 0.03],
    })

    np.testing.assert_allclose(
        camera.K,
        np.array([[600.0, 0.0, 640.0], [0.0, 610.0, 360.0], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(camera.D, np.array([0.1, -0.2, 0.01, 0.02, 0.03]))


def test_build_preview_payload_eye_in_hand_returns_per_frame_camera_and_board_in_base(tmp_path):
    from gui_api import build_preview_payload

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "001_Color.png").write_bytes(b"")
    (image_dir / "002_Color.png").write_bytes(b"")
    poses_file = tmp_path / "poses.csv"
    poses_file.write_text("0,0,0,0,0,0\n0,0,0,0,0,0\n", encoding="utf-8")

    payload = build_preview_payload(
        image_dir=str(image_dir),
        poses_file=str(poses_file),
        setup="eye-in-hand",
        pose_format="sxyz",
        primary_transform_name="T_C2F",
        primary_matrix_rows=[
            "1, 0, 0, 0.1",
            "0, 1, 0, 0.2",
            "0, 0, 1, 0.3",
            "0, 0, 0, 1",
        ],
        secondary_transform_name="T_O2W",
        secondary_matrix_rows=[
            "1, 0, 0, 1.0",
            "0, 1, 0, 2.0",
            "0, 0, 1, 3.0",
            "0, 0, 0, 1",
        ],
        frame_errors=[
            {"index": 0, "image_path": str(image_dir / "001_Color.png"), "used": True},
            {"index": 1, "image_path": str(image_dir / "002_Color.png"), "used": False},
        ],
        object_to_camera_matrices=[
            [
                [1, 0, 0, 0.01],
                [0, 1, 0, 0.02],
                [0, 0, 1, 0.03],
                [0, 0, 0, 1],
            ],
            [
                [1, 0, 0, 0.04],
                [0, 1, 0, 0.05],
                [0, 0, 1, 0.06],
                [0, 0, 0, 1],
            ],
        ],
    )

    assert payload["previewFrames"][0]["used"] is True
    assert payload["previewFrames"][1]["used"] is False
    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][0]["cameraInBase"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][0]["boardInBase"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 0.11],
            [0.0, 1.0, 0.0, 0.22],
            [0.0, 0.0, 1.0, 0.33],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][1]["boardInBase"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][0]["boardInFocus"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 0.11],
            [0.0, 1.0, 0.0, 0.22],
            [0.0, 0.0, 1.0, 0.33],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )


def test_build_preview_payload_eye_to_hand_returns_constant_camera_and_per_frame_board(tmp_path):
    from gui_api import build_preview_payload

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "001_Color.png").write_bytes(b"")
    poses_file = tmp_path / "poses.csv"
    poses_file.write_text("1000,2000,3000,0,0,0\n", encoding="utf-8")

    payload = build_preview_payload(
        image_dir=str(image_dir),
        poses_file=str(poses_file),
        setup="eye-to-hand",
        pose_format="sxyz",
        primary_transform_name="T_C2W",
        primary_matrix_rows=[
            "1, 0, 0, 4.0",
            "0, 1, 0, 5.0",
            "0, 0, 1, 6.0",
            "0, 0, 0, 1",
        ],
        secondary_transform_name="T_O2F",
        secondary_matrix_rows=[
            "1, 0, 0, 0.4",
            "0, 1, 0, 0.5",
            "0, 0, 1, 0.6",
            "0, 0, 0, 1",
        ],
        frame_errors=[
            {"index": 0, "image_path": str(image_dir / "001_Color.png"), "used": True},
        ],
    )

    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][0]["cameraInBase"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][0]["boardInBase"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 1.4],
            [0.0, 1.0, 0.0, 2.5],
            [0.0, 0.0, 1.0, 3.6],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][0]["boardInFocus"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.6],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )


def test_run_calibration_auto_uses_camera_params_in_image_dir(monkeypatch, tmp_path):
    calls = {}

    class FakeCalibrator:
        def __init__(self, config, camera_params_file=None, camera=None):
            calls["camera_params_file"] = camera_params_file
            calls["camera"] = camera

        def calibrate(self, image_dir, poses_file, **kwargs):
            return SimpleCalibrationResult()

        def save(self, result, output_path):
            with open(output_path, "w") as f:
                f.write("saved: true\n")

        def save_detections(self, output_dir):
            pass

    monkeypatch.setattr("gui_api.HandEyeCalibrator", FakeCalibrator)
    image_dir = tmp_path / "session"
    image_dir.mkdir()
    camera_params = image_dir / "camera_params.yaml"
    camera_params.write_text(
        "camera_matrix:\n"
        "  rows: 3\n"
        "  cols: 3\n"
        "  data: [1, 0, 2, 0, 3, 4, 0, 0, 1]\n"
        "distortion_coefficients:\n"
        "  rows: 1\n"
        "  cols: 5\n"
        "  data: [0, 0, 0, 0, 0]\n"
    )

    run_calibration({
        "imageDir": str(image_dir),
        "posesFile": str(image_dir / "poses.csv"),
    })

    assert calls["camera_params_file"] == str(camera_params)
    assert calls["camera"] is None


class SimpleCalibrationResult:
    setup = "eye-to-hand"
    T_C2F = None
    T_O2W = None
    T_C2W = np.array([
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ])
    T_O2F = np.eye(4)
    T_O2C_opt = []
    T_O2C_derived = []
    reprojection_error = {"mean": 0.42, "mean_optimized": 0.31}
    pose_error = {"translation_mean": 0.0025, "rotation_mean": 0.8}
    num_images = 5
    num_images_used = 4
    filtered_images = [3]
    depth_used = False
    optimizer_backend = "quaternion_manifold"
    comparison_metrics = None
    base_consistency_error = None
    frame_errors = [
        {
            "index": 0,
            "image_path": "001_Color.png",
            "used": True,
            "corner_count": 42,
            "reprojection_error_px": 0.21,
            "reprojection_rms_px": 0.27,
            "reprojection_max_px": 0.80,
            "optimized_reprojection_error_px": 0.18,
            "reference_reprojection_rms_px": 0.22,
            "reference_reprojection_max_px": 0.62,
            "translation_error": 0.0012,
            "rotation_error_deg": 0.3,
        },
        {
            "index": 1,
            "image_path": "002_Color.png",
            "used": True,
            "corner_count": 38,
            "reprojection_error_px": 0.63,
            "reprojection_rms_px": 0.71,
            "reprojection_max_px": 1.40,
            "optimized_reprojection_error_px": 0.44,
            "reference_reprojection_rms_px": 0.49,
            "reference_reprojection_max_px": 1.10,
            "translation_error": 0.0038,
            "rotation_error_deg": 1.1,
        },
    ]


def test_calibration_result_payload_describes_per_frame_errors():
    from gui_api import calibration_result_payload

    payload = calibration_result_payload(SimpleCalibrationResult(), "/tmp/result.yaml")
    frame = payload["frameErrors"][0]

    assert frame["cornerCount"] == 42
    assert frame["reprojectionMeanPx"] == 0.21
    assert frame["reprojectionRmsPx"] == 0.27
    assert frame["reprojectionMaxPx"] == 0.80
    assert frame["referenceReprojectionMeanPx"] == 0.18
    assert frame["referenceReprojectionRmsPx"] == 0.22
    assert frame["referenceReprojectionMaxPx"] == 0.62
    assert frame["translationErrorMm"] == 1.2
    assert frame["rotationErrorDeg"] == 0.3
    assert frame["errorMeanings"]["reprojectionMeanPx"]["unit"] == "px"
    assert "全局手眼" in frame["errorMeanings"]["reprojectionMeanPx"]["description"]
    assert "参考位姿" in frame["errorMeanings"]["referenceReprojectionMeanPx"]["description"]
    assert frame["errorMeanings"]["baseConsistencyRmsMm"]["unit"] == "mm"
    assert "法兰末端坐标系" in frame["errorMeanings"]["baseConsistencyRmsMm"]["description"]


def test_calibration_result_payload_includes_base_consistency_summary():
    from gui_api import calibration_result_payload

    class EyeInHandResult(SimpleCalibrationResult):
        setup = "eye-in-hand"
        T_C2F = np.eye(4)
        T_O2W = np.eye(4)
        T_C2W = None
        T_O2F = None
        base_consistency_error = {
            "count": 80,
            "mean": 0.0011,
            "rms": 0.0015,
            "max": 0.0023,
            "per_image": [
                {"rms": 0.0012, "mean": 0.0010, "max": 0.0020, "count": 42},
                {"rms": 0.0018, "mean": 0.0014, "max": 0.0023, "count": 38},
            ],
        }

    result = EyeInHandResult()

    payload = calibration_result_payload(result, "/tmp/result.yaml")

    assert payload["baseConsistencyMeanMm"] == pytest.approx(1.1)
    assert payload["baseConsistencyRmsMm"] == pytest.approx(1.5)
    assert payload["baseConsistencyMaxMm"] == pytest.approx(2.3)
    assert payload["baseConsistencyCount"] == 80
    assert payload["frameErrors"][0]["baseConsistencyRmsMm"] == pytest.approx(1.2)
    assert payload["frameErrors"][0]["baseConsistencyCount"] == 42


def test_calibration_result_payload_includes_eye_in_hand_per_frame_board_preview(tmp_path):
    from gui_api import calibration_result_payload

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "001_Color.png").write_bytes(b"")
    (image_dir / "002_Color.png").write_bytes(b"")
    poses_file = tmp_path / "poses.csv"
    poses_file.write_text("0,0,0,0,0,0\n0,0,0,0,0,0\n", encoding="utf-8")

    class EyeInHandResult(SimpleCalibrationResult):
        setup = "eye-in-hand"
        T_C2F = np.array([
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ])
        T_O2W = np.eye(4)
        T_C2W = None
        T_O2F = None
        T_O2C_opt = [
            np.array([
                [1.0, 0.0, 0.0, 0.01],
                [0.0, 1.0, 0.0, 0.02],
                [0.0, 0.0, 1.0, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            np.array([
                [1.0, 0.0, 0.0, 0.04],
                [0.0, 1.0, 0.0, 0.05],
                [0.0, 0.0, 1.0, 0.06],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        ]

    payload = calibration_result_payload(
        EyeInHandResult(),
        "/tmp/result.yaml",
        image_dir=str(image_dir),
        poses_file=str(poses_file),
    )

    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][0]["boardInBase"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 0.11],
            [0.0, 1.0, 0.0, 0.22],
            [0.0, 0.0, 1.0, 0.33],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
    np.testing.assert_allclose(
        np.asarray(payload["previewFrames"][1]["boardInBase"], dtype=float),
        np.array([
            [1.0, 0.0, 0.0, 0.14],
            [0.0, 1.0, 0.0, 0.25],
            [0.0, 0.0, 1.0, 0.36],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )


def test_calibration_result_payload_includes_backend_metadata():
    from gui_api import calibration_result_payload

    result = SimpleCalibrationResult()
    result.comparison_metrics = {
        "selected_backend": "quaternion_manifold",
        "legacy": {"reprojection_error_px": 0.52},
        "manifold": {"reprojection_error_px": 0.42},
    }

    payload = calibration_result_payload(result, "/tmp/result.yaml")

    assert payload["optimizerBackend"] == "quaternion_manifold"
    assert payload["comparisonMetrics"]["legacy"]["reprojection_error_px"] == 0.52


def wait_for_health(port):
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("API server did not become healthy")


def find_free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
