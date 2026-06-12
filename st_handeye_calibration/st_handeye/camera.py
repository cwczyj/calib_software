import numpy as np
import cv2
from typing import Tuple


class CameraModel:
    def __init__(self, camera_matrix: np.ndarray, distortion: np.ndarray):
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.D = np.asarray(distortion, dtype=np.float64)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'CameraModel':
        from .io import load_camera_params_yaml
        K, D = load_camera_params_yaml(filepath)
        return cls(K, D)
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image, self.K, self.D)
    
    def project(self, points_3d: np.ndarray, object2eye: np.ndarray = None) -> np.ndarray:
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.shape[0] == 4:
            pts = pts[:3]
        if object2eye is not None:
            T = np.asarray(object2eye, dtype=np.float64)
            pts = T[:3, :3] @ pts + T[:3, 3:4]
        uv = self.K @ pts
        return uv[:2] / uv[2:]
    
    def solve_pnp(self, pts3d: np.ndarray, pts2d: np.ndarray,
                  pnp_method: str = "iterative",
                  assume_undistorted: bool = False) -> Tuple[np.ndarray, bool]:
        p3 = np.asarray(pts3d, dtype=np.float64)
        p2 = np.asarray(pts2d, dtype=np.float64)
        if p3.shape[0] == 3:
            p3 = p3.T
        if p2.shape[0] == 2:
            p2 = p2.T
        T = np.eye(4)
        distortion = np.zeros_like(self.D) if assume_undistorted else self.D

        method = (pnp_method or "iterative").lower()
        if method in ("ippe", "ippe_generic"):
            try:
                ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                    p3, p2, self.K, distortion, flags=cv2.SOLVEPNP_IPPE
                )
            except cv2.error:
                ok, rvecs, tvecs, reproj = False, [], [], None
            if not ok or len(rvecs) == 0:
                return T, False
            best = self._choose_pnp_solution(p3, p2, rvecs, tvecs, reproj)
            if best is None:
                return T, False
            rvec, tvec = best
        else:
            flags = {
                "iterative": cv2.SOLVEPNP_ITERATIVE,
                "epnp": cv2.SOLVEPNP_EPNP,
                "sqpnp": getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP),
            }
            if method not in flags:
                raise ValueError(f"Unsupported PnP method: {pnp_method}")
            ok, rvec, tvec = cv2.solvePnP(p3, p2, self.K, distortion, flags=flags[method])
            if not ok:
                return T, False

        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = np.asarray(tvec).reshape(3)
        return T, True

    def _choose_pnp_solution(self, pts3d, pts2d, rvecs, tvecs, reproj):
        best = None
        best_err = np.inf
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R = cv2.Rodrigues(rvec)[0]
            pts_cam = R @ pts3d.T + np.asarray(tvec).reshape(3, 1)
            if np.any(pts_cam[2] <= 0):
                continue
            if reproj is not None:
                err = float(np.asarray(reproj).reshape(-1)[i])
            else:
                uv = self.K @ pts_cam
                proj = (uv[:2] / uv[2:]).T
                err = float(np.mean(np.linalg.norm(pts2d - proj, axis=1)))
            if err < best_err:
                best = (rvec, tvec)
                best_err = err
        return best
