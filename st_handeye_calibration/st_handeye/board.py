import numpy as np
import cv2
from typing import Tuple, Optional, List
from .types import DetectionResult, BoardConfig
from .camera import CameraModel


class CharucoBoard:
    ARUCO_DICTS = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
        'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
        'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
        'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
        'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
        'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(self, config: BoardConfig):
        self.squares_x = config.squares_x
        self.squares_y = config.squares_y
        self.square_length = config.square_length
        self.marker_length = config.marker_length

        if config.aruco_dict not in self.ARUCO_DICTS:
            raise ValueError(f"Unknown dict: {config.aruco_dict}")
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICTS[config.aruco_dict])
        self._board = self._create_board()
        self.pattern_3d = self._board.getChessboardCorners().T[:3]
        self.num_corners = (self.squares_x - 1) * (self.squares_y - 1)

    def _create_board(self) -> cv2.aruco.CharucoBoard:
        try:
            board = cv2.aruco.CharucoBoard(
                size=(self.squares_x, self.squares_y),
                squareLength=self.square_length,
                markerLength=self.marker_length,
                dictionary=self._aruco_dict
            )
            board.setLegacyPattern(True)
            return board
        except TypeError:
            board = cv2.aruco.CharucoBoard_create(
                self.squares_x, self.squares_y,
                self.square_length, self.marker_length, self._aruco_dict
            )
            return board

    def detect(self, image: np.ndarray, camera: CameraModel, min_corners: int = 20) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        undist = camera.undistort(gray)
        zero_distortion = np.zeros_like(camera.D)

        corners, ids, num_markers = self._detect_charuco(undist, camera.K, zero_distortion)
        if corners is None or len(corners) < min_corners:
            return DetectionResult(np.zeros((2,0)), np.zeros(0,int), False, '', 0, num_markers)

        corners_2d = np.full((2, self.num_corners), np.nan)
        ids_list = []
        for i, cid in enumerate(ids.flatten()):
            if cid < self.num_corners:
                corners_2d[0, cid] = corners[i][0][0]
                corners_2d[1, cid] = corners[i][0][1]
                ids_list.append(cid)

        mask = ~np.isnan(corners_2d[0,:])
        return DetectionResult(
            corners_2d[:,mask], np.array(ids_list,int), True, '', len(ids_list), num_markers
        )

    def _detect_charuco(self, undist, K=None, D=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        cb_size = (self.squares_x - 1, self.squares_y - 1)
        marker_corners = None
        marker_ids = None
        
        # Try CharucoDetector API (OpenCV 4.7+) with camera intrinsics
        try:
            params = cv2.aruco.CharucoParameters()
            if K is not None and D is not None:
                params.cameraMatrix = K.astype(np.float64)
                params.distCoeffs = D.astype(np.float64)
            detector = cv2.aruco.CharucoDetector(self._board, params, cv2.aruco.DetectorParameters())
            corners, ids, marker_corners, marker_ids = detector.detectBoard(undist)
            if corners is not None and len(corners) >= 20:
                n_markers = 0 if marker_ids is None else len(marker_ids)
                return corners, ids, n_markers
        except (AttributeError, TypeError):
            pass

        try:
            detector_params = cv2.aruco.DetectorParameters()
            aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, detector_params)
            marker_corners, marker_ids, _ = aruco_detector.detectMarkers(undist)
            if marker_ids is not None and len(marker_ids) > 0:
                try:
                    cv2.aruco.refineDetectedMarkers(
                        undist, self._board, marker_corners, marker_ids, None, None
                    )
                except cv2.error:
                    pass
                ret, corners, ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, undist, self._board, K, D
                )
                if ret is not None and ret >= 20 and corners is not None:
                    cv2.cornerSubPix(
                        undist, corners, (5, 5), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    return corners, ids, len(marker_ids)
        except (AttributeError, TypeError, cv2.error):
            pass
        
        # Fallback to traditional chessboard detection
        ret, cb = cv2.findChessboardCorners(undist, cb_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            n_markers = 0 if marker_ids is None else len(marker_ids)
            return None, None, n_markers
        cb = cv2.cornerSubPix(undist, cb, (5,5), (-1,-1),
                              (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        n_markers = 0 if marker_ids is None else len(marker_ids)
        return cb, np.arange(len(cb)).reshape(-1,1), n_markers
