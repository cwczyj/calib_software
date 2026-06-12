import os
import csv
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Optional
from dataclasses import replace

from .types import BoardConfig, DetectionResult, CalibrationResult, CalibrationSetup, OptimizationParams
from .board import CharucoBoard
from .camera import CameraModel
from .depth import load_depth_image, match_depth_path, sample_depth_at_corners
from .optimizer import GraphOptimizer, compute_initial_guess
from .evaluation import evaluate_calibration, generate_visualization
from .io import load_poses_csv, find_image_files, save_calibration_yaml


class HandEyeCalibrator:
    def __init__(
        self,
        board_config: BoardConfig,
        camera_params_file: Optional[str] = None,
        camera: Optional[CameraModel] = None,
    ):
        self.board = CharucoBoard(board_config)
        if camera is not None:
            self.camera = camera
        elif camera_params_file is not None:
            self.camera = CameraModel.from_yaml(camera_params_file)
        else:
            raise ValueError('camera_params_file or camera is required')
        self._detections = None
        self._image_paths = None

    def calibrate(self, image_dir: str, poses_file: str,
                  filter_inconsistent: bool = True,
                  rot_threshold: float = 2.0,
                  trans_threshold: float = 0.020,
                  opt_params: Optional[OptimizationParams] = None,
                  setup=CalibrationSetup.EYE_IN_HAND,
                  depth_dir: Optional[str] = None,
                  use_depth=False,
                  depth_scale: float = 0.001,
                  min_depth: float = 0.05,
                  max_depth: float = 5.0,
                  depth_window: int = 3,
                  min_filter_corners: int = 40,
                  min_filter_markers: int = 30,
                  max_filter_pnp_reprojection: float = 1.0,
                  filter_iterations: int = 3,
                  filter_mad_scale: float = 3.5,
                  pose_trans_unit: float = 0.001,
                  pose_rot_order: str = "xyz",
                  pose_rot_degrees: bool = True,
                  pose_invert: bool = False,
                  pnp_method: str = "iterative",
                  excluded_image_indices: Optional[List[int]] = None,
                  diagnostics_output: Optional[str] = None) -> CalibrationResult:
        if opt_params is None:
            opt_params = OptimizationParams()
        setup = CalibrationSetup.parse(setup)
        depth_mode = "off"
        if isinstance(use_depth, str):
            depth_mode = use_depth.lower()
        elif use_depth:
            depth_mode = "optional"
        depth_required = depth_mode == "required"
        depth_enabled = depth_mode in ("optional", "required", "true", "1", "yes")

        print(f'Loading from {image_dir}...')
        image_paths = find_image_files(image_dir)
        poses = load_poses_csv(
            poses_file,
            trans_unit=pose_trans_unit,
            rot_deg=pose_rot_degrees,
            rot_order=pose_rot_order,
            invert=pose_invert,
        )
        print(f'Found {len(image_paths)} images, {len(poses)} poses')
        all_frame_paths = list(image_paths)

        print('Detecting corners...')
        detections, valid_idx, depth_samples = [], [], []
        for i, path in enumerate(image_paths[:len(poses)]):
            img = cv2.imread(path)
            if img is None:
                continue
            det = self.board.detect(img, self.camera)
            det.image_path = path
            det.frame_index = i
            if det.success:
                p3 = self.board.pattern_3d[:, det.corner_ids]
                T_pnp, det.pnp_success = self.camera.solve_pnp(
                    p3, det.corners_2d, pnp_method=pnp_method,
                    assume_undistorted=True
                )
                if det.pnp_success:
                    proj = self.camera.project(p3, T_pnp)
                    det.pnp_reprojection_error = float(
                        np.mean(np.linalg.norm(det.corners_2d - proj, axis=0))
                    )
                depth_sample = None
                if depth_enabled:
                    depth_path = match_depth_path(path, depth_dir)
                    depth_img = load_depth_image(depth_path, image_shape=img.shape[:2])
                    if depth_img is None:
                        if depth_required:
                            raise FileNotFoundError(f'Depth image not found: {depth_path}')
                    else:
                        depth_sample = sample_depth_at_corners(
                            depth_img, det.corners_2d, det.corner_ids, self.camera.K,
                            depth_scale=depth_scale, min_depth=min_depth, max_depth=max_depth,
                            window_size=depth_window
                        )
                        det.num_depth_corners = int(depth_sample.corner_ids.shape[0])
                detections.append(det)
                depth_samples.append(depth_sample)
                valid_idx.append(i)
                depth_msg = f', depth={det.num_depth_corners}' if depth_enabled else ''
                print(f'  Image {i}: {det.num_corners} corners, markers={det.num_markers}{depth_msg}')

        poses = [poses[i] for i in valid_idx]
        print(f'Detected: {len(detections)} images')

        if len(detections) < 3:
            raise ValueError('Not enough detections')

        diagnostics = self._build_filter_diagnostics(detections)
        filtered = []
        active_indices = [int(getattr(det, "frame_index", index)) for index, det in enumerate(detections)]
        manual_excluded = {int(index) for index in (excluded_image_indices or [])}
        if manual_excluded:
            keep = [i for i, index in enumerate(active_indices) if index not in manual_excluded]
            rejected = [index for index in active_indices if index in manual_excluded]
            for index in rejected:
                if index in diagnostics:
                    diagnostics[index]["filter_reason"] = "manual_excluded"
            if len(keep) < 3:
                raise ValueError('Not enough detections after excluding selected frames')
            detections = [detections[i] for i in keep]
            poses = [poses[i] for i in keep]
            depth_samples = [depth_samples[i] for i in keep]
            active_indices = [active_indices[i] for i in keep]
            filtered.extend(rejected)
            print(f'  Manually excluded: {rejected}')
        if filter_inconsistent:
            print('Filtering by detection quality...')
            keep = self._quality_filter(
                detections,
                min_filter_corners,
                min_filter_markers,
                max_filter_pnp_reprojection,
            )
            rejected = [active_indices[i] for i in range(len(active_indices)) if i not in keep]
            keep_set = set(keep)
            for i, det in enumerate(detections):
                if i not in keep_set:
                    diagnostics[active_indices[i]]["filter_reason"] = self._quality_filter_reason(
                        det, min_filter_corners, min_filter_markers, max_filter_pnp_reprojection
                    )
            if len(keep) >= 3:
                detections = [detections[i] for i in keep]
                poses = [poses[i] for i in keep]
                depth_samples = [depth_samples[i] for i in keep]
                active_indices = [active_indices[i] for i in keep]
                filtered.extend(rejected)
            print(f'  Quality kept: {len(keep)}, Filtered: {rejected}')

        print('Computing initial guess...')
        init = compute_initial_guess(
            self.board.pattern_3d, detections, poses, self.camera, setup,
            pnp_method=pnp_method, depth_samples=depth_samples
        )

        if filter_inconsistent:
            print('Filtering inconsistent...')
            consistent = self._filter(
                init, poses, rot_threshold, trans_threshold, setup,
                max_iterations=filter_iterations, mad_scale=filter_mad_scale,
                diagnostics=diagnostics, active_indices=active_indices
            )
            rejected = [active_indices[i] for i in range(len(active_indices)) if i not in consistent]
            for idx in rejected:
                if diagnostics[idx]["filter_reason"] == "kept":
                    diagnostics[idx]["filter_reason"] = "constraint_outlier"
            if len(consistent) >= 3:
                detections = [detections[i] for i in consistent]
                poses = [poses[i] for i in consistent]
                depth_samples = [depth_samples[i] for i in consistent]
                active_indices = [active_indices[i] for i in consistent]
                filtered.extend(rejected)
                init = compute_initial_guess(
                    self.board.pattern_3d, detections, poses, self.camera, setup,
                    pnp_method=pnp_method, depth_samples=depth_samples
                )
            filtered = sorted(set(filtered))
            print(f'  Consistent: {len(consistent)}, Filtered: {filtered}')

        print('Optimizing...')
        opt = GraphOptimizer()
        primary, target, T_O2C_opt, T_O2C_derived = opt.optimize(
            self.board.pattern_3d, detections, poses, init,
            self.camera.K, opt_params, setup=setup, depth_samples=depth_samples
        )

        print('Evaluating...')
        metrics = evaluate_calibration(
            self.board.pattern_3d, detections, poses,
            primary, target, T_O2C_opt, self.camera, T_O2C_derived,
            setup=setup, T_O2C_measured=init.T_O2C_measured,
            depth_samples=depth_samples,
        )

        self._detections = detections
        self._image_paths = [det.image_path for det in detections]
        depth_used = any(ds is not None and ds.points_camera.shape[1] > 0 for ds in depth_samples)

        if setup == CalibrationSetup.EYE_IN_HAND:
            T_C2F, T_O2W = primary, target
            T_C2W, T_O2F = None, None
        else:
            T_C2F, T_O2W = None, None
            T_C2W, T_O2F = primary, target

        frame_errors = self._build_frame_errors(detections, active_indices, metrics, all_frame_paths)
        comparison_metrics = None
        backend_name = getattr(opt_params, "optimizer_backend", "quaternion_manifold")
        if getattr(opt_params, "compare_with_legacy_backend", False) and backend_name != "rotvec_scipy":
            legacy_params = replace(
                opt_params,
                optimizer_backend="rotvec_scipy",
                compare_with_legacy_backend=False,
            )
            legacy_primary, legacy_target, legacy_T_O2C_opt, legacy_T_O2C_derived = opt.optimize(
                self.board.pattern_3d, detections, poses, init,
                self.camera.K, legacy_params, setup=setup, depth_samples=depth_samples
            )
            legacy_metrics = evaluate_calibration(
                self.board.pattern_3d, detections, poses,
                legacy_primary, legacy_target, legacy_T_O2C_opt,
                self.camera, legacy_T_O2C_derived, setup=setup,
                T_O2C_measured=init.T_O2C_measured, depth_samples=depth_samples
            )
            comparison_metrics = {
                "selected_backend": backend_name,
                "legacy": self._summarize_metrics(legacy_metrics),
                "manifold": self._summarize_metrics(metrics),
            }

        result = CalibrationResult(
            T_C2F=T_C2F, T_O2W=T_O2W, T_O2C_opt=T_O2C_opt,
            T_O2C_derived=T_O2C_derived,
            reprojection_error=metrics['reprojection_error'],
            pose_error=metrics['pose_error'],
            num_images=len(image_paths), num_images_used=len(detections),
            filtered_images=filtered, setup=setup, T_C2W=T_C2W, T_O2F=T_O2F,
            depth_used=depth_used, frame_errors=frame_errors,
            optimizer_backend=backend_name, comparison_metrics=comparison_metrics,
            base_consistency_error=metrics.get("base_consistency")
        )

        if setup == CalibrationSetup.EYE_IN_HAND:
            print(f'\nT_C2F (camera pose in flange frame):\n{T_C2F}')
            print(f'T_O2W:\n{T_O2W}')
        else:
            print(f'\nT_C2W (camera pose in world frame):\n{T_C2W}')
            print(f'T_O2F:\n{T_O2F}')
        print()
        print('=== Evaluation Results ===')
        print(f'Reprojection (derived): {metrics["reprojection_error"]["mean"]:.2f} px')
        print(f'Reprojection (opt): {metrics["reprojection_error"]["mean_optimized"]:.2f} px')
        print(f'Pose error: {metrics["pose_error"]["translation_mean"]*1000:.2f} mm, '
              f'{metrics["pose_error"]["rotation_mean"]:.2f} deg')
        base_consistency = metrics.get("base_consistency")
        if isinstance(base_consistency, dict) and base_consistency.get("rms") is not None:
            print(
                f'Base 3D consistency RMS: {float(base_consistency["rms"]) * 1000.0:.2f} mm '
                f'({int(base_consistency.get("count", 0))} points)'
            )
        elif setup == CalibrationSetup.EYE_IN_HAND:
            print('Base 3D consistency RMS: unavailable (no repeated base-frame corner observations)')
        self._print_frame_error_summary(frame_errors)
        if depth_enabled:
            print(f'Depth samples used: {sum(det.num_depth_corners for det in detections)}')
        print()
        if setup == CalibrationSetup.EYE_IN_HAND:
            print('Constraint: T_O2C = inv(T_C2F) @ inv(T_F2W) @ T_O2W')
        else:
            print('Constraint: T_O2C = inv(T_C2W) @ T_F2W @ T_O2F')
        if diagnostics_output:
            self._save_filter_diagnostics(diagnostics_output, diagnostics)

        return result

    def _build_frame_errors(self, detections, active_indices, metrics, all_frame_paths=None):
        def optional_float(value):
            return None if value is None else float(value)

        reprojection = metrics.get("reprojection_error", {}) if metrics else {}
        pose = metrics.get("pose_error", {}) if metrics else {}
        base = metrics.get("base_consistency", {}) if metrics else {}
        derived_reprojection = reprojection.get("per_image", [])
        optimized_reprojection = reprojection.get("per_image_optimized", [])
        derived_stats = reprojection.get("per_image_stats", [])
        optimized_stats = reprojection.get("per_image_optimized_stats", [])
        pose_errors = pose.get("per_image", [])
        base_errors = base.get("per_image", []) if isinstance(base, dict) else []

        frame_errors_by_index = {}
        for row, det in enumerate(detections):
            pose_error = pose_errors[row] if row < len(pose_errors) else {}
            base_error = base_errors[row] if row < len(base_errors) and isinstance(base_errors[row], dict) else {}
            translation = pose_error.get("translation") if isinstance(pose_error, dict) else None
            rotation = pose_error.get("rotation") if isinstance(pose_error, dict) else None
            derived_row = derived_stats[row] if row < len(derived_stats) and isinstance(derived_stats[row], dict) else {}
            optimized_row = (
                optimized_stats[row]
                if row < len(optimized_stats) and isinstance(optimized_stats[row], dict)
                else {}
            )
            frame_index = int(active_indices[row]) if row < len(active_indices) else int(row)
            frame_errors_by_index[frame_index] = {
                "index": frame_index,
                "image_path": det.image_path,
                "used": True,
                "corner_count": int(derived_row.get("count", det.num_corners)),
                "reprojection_error_px": float(derived_reprojection[row]) if row < len(derived_reprojection) else None,
                "reprojection_mean_px": optional_float(derived_row.get("mean")),
                "reprojection_rms_px": optional_float(derived_row.get("rms")),
                "reprojection_max_px": optional_float(derived_row.get("max")),
                "optimized_reprojection_error_px": (
                    float(optimized_reprojection[row]) if row < len(optimized_reprojection) else None
                ),
                "reference_reprojection_mean_px": optional_float(optimized_row.get("mean")),
                "reference_reprojection_rms_px": optional_float(optimized_row.get("rms")),
                "reference_reprojection_max_px": optional_float(optimized_row.get("max")),
                "translation_error": None if translation is None else float(translation),
                "rotation_error_deg": None if rotation is None else float(rotation),
                "base_consistency_rms": optional_float(base_error.get("rms")),
                "base_consistency_mean": optional_float(base_error.get("mean")),
                "base_consistency_max": optional_float(base_error.get("max")),
                "base_consistency_count": optional_float(base_error.get("count")),
            }

        if all_frame_paths is not None:
            for index, image_path in enumerate(all_frame_paths):
                if index not in frame_errors_by_index:
                    frame_errors_by_index[index] = {
                        "index": int(index),
                        "image_path": image_path,
                        "used": False,
                        "corner_count": None,
                        "reprojection_error_px": None,
                        "reprojection_mean_px": None,
                        "reprojection_rms_px": None,
                        "reprojection_max_px": None,
                        "optimized_reprojection_error_px": None,
                        "reference_reprojection_mean_px": None,
                        "reference_reprojection_rms_px": None,
                        "reference_reprojection_max_px": None,
                        "translation_error": None,
                        "rotation_error_deg": None,
                        "base_consistency_rms": None,
                        "base_consistency_mean": None,
                        "base_consistency_max": None,
                        "base_consistency_count": None,
                    }
        return [frame_errors_by_index[index] for index in sorted(frame_errors_by_index)]

    def _print_frame_error_summary(self, frame_errors):
        if not frame_errors:
            return
        print()
        print("Per-frame errors:")
        print("  global mean/rms/max(px): final hand-eye chain reprojection error")
        print("  pnp mean(px): per-frame PnP reference reprojection error")
        print("  base3d rms(mm): same corner consistency after transforming points to robot base frame")
        print("  pose trans(mm)/rot(deg): PnP reference pose vs final hand-eye chain pose")
        header = (
            f"{'idx':>4} {'corners':>7} {'global mean/rms/max(px)':>27} "
            f"{'pnp mean(px)':>12} {'base3d rms(mm)':>14} {'pose trans(mm)':>14} {'rot(deg)':>9}"
        )
        print(header)
        print("-" * len(header))
        for row in frame_errors:
            global_mean = self._format_optional(row.get("reprojection_mean_px", row.get("reprojection_error_px")))
            global_rms = self._format_optional(row.get("reprojection_rms_px"))
            global_max = self._format_optional(row.get("reprojection_max_px"))
            reference_mean = self._format_optional(
                row.get("reference_reprojection_mean_px", row.get("optimized_reprojection_error_px"))
            )
            base_rms = row.get("base_consistency_rms")
            base_rms_mm = None if base_rms is None else float(base_rms) * 1000.0
            trans = row.get("translation_error")
            trans_mm = None if trans is None else float(trans) * 1000.0
            corner_count = row.get("corner_count")
            corner_count_text = "--" if corner_count is None else str(int(corner_count))
            print(
                f"{int(row.get('index', 0)):>4} "
                f"{corner_count_text:>7} "
                f"{global_mean}/{global_rms}/{global_max:>9} "
                f"{reference_mean:>12} "
                f"{self._format_optional(base_rms_mm):>14} "
                f"{self._format_optional(trans_mm):>14} "
                f"{self._format_optional(row.get('rotation_error_deg')):>9}"
            )

    @staticmethod
    def _format_optional(value):
        return "--" if value is None else f"{float(value):.2f}"

    @staticmethod
    def _summarize_metrics(metrics):
        reprojection = metrics.get("reprojection_error", {}) if metrics else {}
        pose = metrics.get("pose_error", {}) if metrics else {}
        base = metrics.get("base_consistency", {}) if metrics else {}
        return {
            "reprojection_error_px": float(reprojection.get("mean", 0.0)),
            "optimized_reprojection_error_px": float(reprojection.get("mean_optimized", 0.0)),
            "translation_error_mm": float(pose.get("translation_mean", 0.0)) * 1000.0,
            "rotation_error_deg": float(pose.get("rotation_mean", 0.0)),
            "base_consistency_rms_mm": (
                None if not isinstance(base, dict) or base.get("rms") is None
                else float(base.get("rms")) * 1000.0
            ),
        }

    def _build_filter_diagnostics(self, detections):
        rows = {}
        for i, det in enumerate(detections):
            index = int(getattr(det, "frame_index", i))
            rows[index] = {
                "index": index,
                "image_path": det.image_path,
                "num_corners": det.num_corners,
                "num_markers": det.num_markers,
                "pnp_success": det.pnp_success,
                "pnp_reprojection_px": det.pnp_reprojection_error,
                "constraint_translation_mm": "",
                "constraint_rotation_deg": "",
                "filter_reason": "kept",
            }
        return rows

    def _quality_filter_reason(self, det, min_corners, min_markers, max_pnp_reprojection):
        if det.num_corners < min_corners:
            return "low_corners"
        if det.num_markers < min_markers:
            return "low_markers"
        if not det.pnp_success:
            return "pnp_failed"
        if det.pnp_reprojection_error > max_pnp_reprojection:
            return "high_pnp_reprojection"
        return "kept"

    def _quality_filter(self, detections, min_corners, min_markers, max_pnp_reprojection):
        consistent = []
        for i, det in enumerate(detections):
            if det.num_corners < min_corners:
                continue
            if det.num_markers < min_markers:
                continue
            if not det.pnp_success:
                continue
            if det.pnp_reprojection_error > max_pnp_reprojection:
                continue
            consistent.append(i)
        return consistent

    def _filter(self, init, poses, rot_thresh, trans_thresh, setup,
                max_iterations=3, mad_scale=3.5,
                diagnostics=None, active_indices=None):
        from .optimizer import (
            derive_object_to_camera, _fit_global_transforms_from_object_to_camera,
        )
        T_O2C_pnp = (
            init.T_O2C_measured
            if init.T_O2C_measured
            else init.T_O2C_pnp
            if init.T_O2C_pnp
            else init.T_O2C_list
        )
        consistent = list(range(len(poses)))
        for _ in range(max_iterations):
            if len(consistent) < 3:
                break
            fit_poses = [poses[i] for i in consistent]
            fit_pnp = [T_O2C_pnp[i] for i in consistent]
            primary, target = _fit_global_transforms_from_object_to_camera(
                setup, fit_poses, fit_pnp, init.T_C2F, init.T_O2W, max_nfev=1000
            )

            errors = []
            for i in consistent:
                T_O2C_der = derive_object_to_camera(setup, poses[i], primary, target)
                delta = np.linalg.inv(T_O2C_pnp[i]) @ T_O2C_der
                trans_err = np.linalg.norm(delta[:3, 3])
                rot_err = Rotation.from_matrix(delta[:3, :3]).magnitude()
                errors.append((i, trans_err, rot_err))
                if diagnostics is not None and active_indices is not None:
                    row = diagnostics[active_indices[i]]
                    row["constraint_translation_mm"] = trans_err * 1000.0
                    row["constraint_rotation_deg"] = rot_err * 180.0 / np.pi

            trans = np.array([e[1] for e in errors])
            rot = np.array([e[2] for e in errors])
            trans_limit = self._robust_limit(trans, trans_thresh, mad_scale)
            rot_limit = self._robust_limit(rot, np.deg2rad(rot_thresh), mad_scale)
            scores = [
                max(trans_err / trans_limit, rot_err / rot_limit)
                for _, trans_err, rot_err in errors
            ]
            worst = int(np.argmax(scores))
            if scores[worst] <= 1.0:
                break
            consistent = [
                i for j, (i, _, _) in enumerate(errors)
                if j != worst
            ]
        return consistent

    @staticmethod
    def _robust_limit(values, absolute_limit, mad_scale):
        if values.size == 0:
            return absolute_limit
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad < 1e-12:
            return absolute_limit
        return min(absolute_limit, median + mad_scale * 1.4826 * mad)

    def _save_filter_diagnostics(self, filepath, diagnostics):
        if not diagnostics:
            return
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        fields = [
            "index", "image_path", "num_corners", "num_markers",
            "pnp_success", "pnp_reprojection_px",
            "constraint_translation_mm", "constraint_rotation_deg",
            "filter_reason",
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            rows = diagnostics.values() if isinstance(diagnostics, dict) else diagnostics
            for row in rows:
                writer.writerow(row)

    def save(self, result: CalibrationResult, filepath: str):
        if result.setup == CalibrationSetup.EYE_IN_HAND:
            transforms = {'T_C2F': result.T_C2F, 'T_O2W': result.T_O2W}
        else:
            transforms = {'T_C2W': result.T_C2W, 'T_O2F': result.T_O2F}
        save_calibration_yaml(
            filepath,
            setup=result.setup,
            transforms=transforms,
            metrics={
                'reprojection_error': result.reprojection_error,
                'pose_error': result.pose_error,
                'base_consistency': result.base_consistency_error,
            },
            num_images=result.num_images,
            num_images_used=result.num_images_used,
            filtered_images=result.filtered_images,
            depth_used=result.depth_used,
        )
        print(f'Saved to {filepath}')

    def visualize(self, result: CalibrationResult, output_dir: str):
        if self._detections is None or self._image_paths is None:
            print('No detections stored - run calibrate first')
            return
        generate_visualization(
            self._image_paths,
            self.board.pattern_3d,
            self._detections,
            result.T_O2C_derived,
            self.camera,
            output_dir
        )

    def save_detections(self, output_dir: str):
        if self._detections is None or self._image_paths is None:
            print('No detections stored - run calibrate first')
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (img_path, det) in enumerate(zip(self._image_paths, self._detections)):
            if not det.success:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            undist = self.camera.undistort(img)
            
            # Draw detected corners with IDs
            for j, (pt, cid) in enumerate(zip(det.corners_2d.T, det.corner_ids)):
                cv2.circle(undist, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
                cv2.putText(undist, str(cid), (int(pt[0])+5, int(pt[1])-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Estimate pose and draw axes
            p3 = self.board.pattern_3d[:, det.corner_ids]
            pose, ok = self.camera.solve_pnp(p3, det.corners_2d, assume_undistorted=True)
            if ok:
                from scipy.spatial.transform import Rotation
                rvec = Rotation.from_matrix(pose[:3,:3]).as_rotvec().reshape(3,1).astype(np.float32)
                tvec = pose[:3,3].reshape(3,1).astype(np.float32)
                cv2.drawFrameAxes(undist, self.camera.K.astype(np.float32), 
                                  self.camera.D.astype(np.float32), rvec, tvec, 0.028)
                
                pose_text = f'Corners: {det.num_corners}/70 | Pose: t=[{tvec[0,0]:.3f},{tvec[1,0]:.3f},{tvec[2,0]:.3f}]'
            else:
                pose_text = f'Corners: {det.num_corners}/70 | No Pose'
            
            cv2.putText(undist, pose_text, (10, undist.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out_path = os.path.join(output_dir, f'detection_{i:03d}.png')
            cv2.imwrite(out_path, undist)
        
        print(f'Detection images saved to {output_dir}/ ({len(self._detections)} images)')
