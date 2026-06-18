use apex_manifolds::se3::{SE3Tangent, SE3};
use apex_manifolds::{LieGroup, ManifoldType};
use apex_solver::core::problem::Problem;
use apex_solver::factors::Factor;
use apex_solver::linalg::JacobianMode;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Rotation3, UnitQuaternion, Vector2, Vector3};
use opencv::calib3d;
use opencv::core::{self, Mat, Point2d, Point2f, Point3d, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc, objdetect};
use serde::{Deserialize, Serialize};
use serde_yaml::{Mapping, Value};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageFile {
    name: String,
    path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PoseFileRow {
    index: usize,
    content: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CalibrationRequest {
    image_dir: String,
    poses_file: String,
    marker: Option<String>,
    camera_params: Option<String>,
    camera_intrinsics: Option<CameraIntrinsics>,
    setup: String,
    pose_format: Option<String>,
    use_depth: String,
    squares_x: Option<usize>,
    squares_y: Option<usize>,
    square_length: Option<f64>,
    marker_length: Option<f64>,
    aruco_dict: Option<String>,
    filter_inconsistent: Option<bool>,
    excluded_image_indices: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CalibrationRun {
    output_path: String,
    #[serde(default)]
    stdout: String,
    #[serde(default)]
    stderr: String,
    #[serde(default)]
    setup: String,
    #[serde(default)]
    primary_transform_name: String,
    #[serde(default)]
    primary_matrix_rows: Vec<String>,
    #[serde(default)]
    secondary_transform_name: String,
    #[serde(default)]
    secondary_matrix_rows: Vec<String>,
    #[serde(default)]
    matrix_rows: Vec<String>,
    #[serde(default)]
    average_error_mm: f64,
    #[serde(default)]
    rotation_error_deg: f64,
    #[serde(default)]
    reprojection_error_px: f64,
    #[serde(default)]
    reprojection_rms_px: Option<f64>,
    #[serde(default)]
    base_consistency_mean_mm: Option<f64>,
    #[serde(default)]
    base_consistency_rms_mm: Option<f64>,
    #[serde(default)]
    base_consistency_max_mm: Option<f64>,
    #[serde(default)]
    base_consistency_count: Option<usize>,
    #[serde(default)]
    num_images: usize,
    #[serde(default)]
    num_images_used: usize,
    #[serde(default)]
    filtered_images: Vec<usize>,
    #[serde(default)]
    frame_errors: Vec<FrameError>,
    #[serde(default)]
    preview_frames: Vec<PreviewFrame>,
    #[serde(default)]
    depth_used: bool,
    #[serde(default)]
    message: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PreviewFrame {
    index: usize,
    #[serde(default)]
    image_path: String,
    #[serde(default = "default_true")]
    used: bool,
    camera_in_base: Vec<Vec<f64>>,
    board_in_base: Vec<Vec<f64>>,
    #[serde(default)]
    board_in_focus: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversionPreviewRequest {
    image_dir: String,
    poses_file: String,
    setup: String,
    pose_format: String,
    primary_transform_name: String,
    primary_matrix_rows: Vec<String>,
    secondary_transform_name: String,
    secondary_matrix_rows: Vec<String>,
    #[serde(default)]
    camera_intrinsics: Option<CameraIntrinsics>,
    #[serde(default)]
    squares_x: Option<usize>,
    #[serde(default)]
    squares_y: Option<usize>,
    #[serde(default)]
    square_length: Option<f64>,
    #[serde(default)]
    marker_length: Option<f64>,
    #[serde(default)]
    aruco_dict: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversionPreviewResult {
    #[serde(default)]
    preview_frames: Vec<PreviewFrame>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameError {
    index: usize,
    #[serde(default)]
    image_path: String,
    #[serde(default = "default_true")]
    used: bool,
    #[serde(default)]
    corner_count: Option<usize>,
    #[serde(default)]
    reprojection_mean_px: Option<f64>,
    #[serde(default)]
    reprojection_rms_px: Option<f64>,
    #[serde(default)]
    reprojection_max_px: Option<f64>,
    #[serde(default)]
    reference_reprojection_mean_px: Option<f64>,
    #[serde(default)]
    reference_reprojection_rms_px: Option<f64>,
    #[serde(default)]
    reference_reprojection_max_px: Option<f64>,
    #[serde(default)]
    reprojection_error_px: Option<f64>,
    #[serde(default)]
    optimized_reprojection_error_px: Option<f64>,
    #[serde(default)]
    base_consistency_mean_mm: Option<f64>,
    #[serde(default)]
    base_consistency_rms_mm: Option<f64>,
    #[serde(default)]
    base_consistency_max_mm: Option<f64>,
    #[serde(default)]
    base_consistency_count: Option<usize>,
    #[serde(default)]
    translation_error_mm: Option<f64>,
    #[serde(default)]
    rotation_error_deg: Option<f64>,
    #[serde(default)]
    used_chessboard_fallback: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharucoRequest {
    image_path: String,
    depth_path: Option<String>,
    camera_params: Option<String>,
    camera_intrinsics: Option<CameraIntrinsics>,
    squares_x: Option<usize>,
    squares_y: Option<usize>,
    square_length: Option<f64>,
    marker_length: Option<f64>,
    aruco_dict: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CameraIntrinsics {
    cx: f64,
    cy: f64,
    fx: f64,
    fy: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    distortion_coefficients: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct DetectionObservation {
    index: usize,
    image_path: String,
    corner_ids: Vec<usize>,
    image_points: Vec<Vector2<f64>>,
    marker_count: usize,
    used_chessboard_fallback: bool,
}

#[derive(Clone, Debug)]
struct SyntheticDetection {
    index: usize,
    image_path: String,
    corner_ids: Vec<usize>,
    image_points: Vec<Vector2<f64>>,
    marker_count: usize,
    used_chessboard_fallback: bool,
}

#[derive(Clone, Debug)]
struct DepthObservation {
    corner_ids: Vec<usize>,
    object_points: Vec<Vector3<f64>>,
    camera_points: Vec<Vector3<f64>>,
}

#[derive(Clone, Debug)]
struct HandeyeOptimization {
    primary_transform: Matrix4<f64>,
    secondary_transform: Matrix4<f64>,
    measured_object_to_camera: Vec<Matrix4<f64>>,
    object_to_camera_derived: Vec<Matrix4<f64>>,
    reprojection_mean_px: f64,
    reprojection_rms_px: f64,
}

#[derive(Clone, Debug)]
struct ReprojectionMetrics {
    mean_px: f64,
    rms_px: f64,
    per_frame: Vec<(f64, f64, f64)>,
}

#[derive(Clone, Debug)]
struct PoseErrorMetrics {
    translation_mean_mm: f64,
    rotation_mean_deg: f64,
    per_frame: Vec<(f64, f64)>,
}

#[derive(Clone, Debug)]
struct DistanceStats {
    count: usize,
    mean_m: f64,
    rms_m: f64,
    max_m: f64,
    per_frame: Vec<Option<(usize, f64, f64, f64)>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DepthMode {
    Off,
    Optional,
    Required,
}

#[derive(Clone)]
struct HandeyeProjectionFactor {
    setup: String,
    pose: Matrix4<f64>,
    object_points: Vec<Vector3<f64>>,
    image_points: Vec<Vector2<f64>>,
    depth_observation: Option<DepthObservation>,
    intrinsics: CameraIntrinsics,
}

#[derive(Clone)]
struct HandeyeConstraintFactor {
    setup: String,
    pose: Matrix4<f64>,
    measured_object_to_camera: Matrix4<f64>,
}

impl Factor for HandeyeProjectionFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let primary = SE3::from(params[0].clone());
        let secondary = SE3::from(params[1].clone());
        let residual = self.residual_for(&primary, &secondary);
        let jacobian = if compute_jacobian {
            Some(numerical_jacobian(
                &primary,
                &secondary,
                residual.len(),
                |left, right| self.residual_for(left, right),
            ))
        } else {
            None
        };
        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        self.image_points.len() * 2
            + self
                .depth_observation
                .as_ref()
                .map(|depth| depth.camera_points.len() * 3)
                .unwrap_or(0)
    }
}

impl HandeyeProjectionFactor {
    fn residual_for(&self, primary: &SE3, secondary: &SE3) -> DVector<f64> {
        let primary_matrix = primary.matrix();
        let secondary_matrix = secondary.matrix();
        let object_to_camera = derive_object_to_camera_matrix(
            &self.setup,
            &self.pose,
            &primary_matrix,
            &secondary_matrix,
        );
        let mut residuals = self
            .object_points
            .iter()
            .zip(self.image_points.iter())
            .flat_map(|(point, observed)| {
                let projected = project_point(&self.intrinsics, &object_to_camera, point);
                [projected.x - observed.x, projected.y - observed.y]
            })
            .collect::<Vec<_>>();
        if let Some(depth) = &self.depth_observation {
            for (point_object, observed) in depth.object_points.iter().zip(depth.camera_points.iter()) {
                let predicted = object_to_camera.fixed_view::<3, 3>(0, 0) * point_object
                    + object_to_camera.fixed_view::<3, 1>(0, 3);
                residuals.extend_from_slice(&[
                    predicted.x - observed.x,
                    predicted.y - observed.y,
                    predicted.z - observed.z,
                ]);
            }
        }
        DVector::from_vec(residuals)
    }
}

impl Factor for HandeyeConstraintFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let primary = SE3::from(params[0].clone());
        let secondary = SE3::from(params[1].clone());
        let residual = self.residual_for(&primary, &secondary);
        let jacobian = if compute_jacobian {
            Some(numerical_jacobian(
                &primary,
                &secondary,
                residual.len(),
                |left, right| self.residual_for(left, right),
            ))
        } else {
            None
        };
        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        6
    }
}

impl HandeyeConstraintFactor {
    fn residual_for(&self, primary: &SE3, secondary: &SE3) -> DVector<f64> {
        let primary_matrix = primary.matrix();
        let secondary_matrix = secondary.matrix();
        let predicted = derive_object_to_camera_matrix(
            &self.setup,
            &self.pose,
            &primary_matrix,
            &secondary_matrix,
        );
        let delta = matrix_to_se3(&self.measured_object_to_camera)
            .inverse(None)
            .compose(&matrix_to_se3(&predicted), None, None)
            .log(None);
        delta.into()
    }
}

fn numerical_jacobian(
    primary: &SE3,
    secondary: &SE3,
    residual_dim: usize,
    residual_fn: impl Fn(&SE3, &SE3) -> DVector<f64>,
) -> DMatrix<f64> {
    let epsilon = 1e-6;
    let mut jacobian = DMatrix::zeros(residual_dim, 12);
    for axis in 0..6 {
        let step = se3_axis_step(axis, epsilon);
        let backward = se3_axis_step(axis, -epsilon);

        let primary_plus = primary.plus(&step, None, None);
        let primary_minus = primary.plus(&backward, None, None);
        let diff = (residual_fn(&primary_plus, secondary) - residual_fn(&primary_minus, secondary))
            * (0.5 / epsilon);
        jacobian.set_column(axis, &diff);

        let secondary_plus = secondary.plus(&step, None, None);
        let secondary_minus = secondary.plus(&backward, None, None);
        let diff = (residual_fn(primary, &secondary_plus) - residual_fn(primary, &secondary_minus))
            * (0.5 / epsilon);
        jacobian.set_column(axis + 6, &diff);
    }
    jacobian
}

fn se3_axis_step(axis: usize, value: f64) -> SE3Tangent {
    let mut values = [0.0; 6];
    values[axis] = value;
    SE3Tangent::from_components(
        values[0], values[1], values[2], values[3], values[4], values[5],
    )
}

fn compose_matrix(translation: Vector3<f64>, rotation: Matrix3<f64>) -> Matrix4<f64> {
    let mut matrix = Matrix4::identity();
    matrix.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    matrix[(0, 3)] = translation.x;
    matrix[(1, 3)] = translation.y;
    matrix[(2, 3)] = translation.z;
    matrix
}

fn matrix_to_se3(matrix: &Matrix4<f64>) -> SE3 {
    let rotation = Rotation3::from_matrix_unchecked(matrix.fixed_view::<3, 3>(0, 0).into_owned());
    SE3::new(
        Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]),
        UnitQuaternion::from_rotation_matrix(&rotation),
    )
}

fn derive_object_to_camera_matrix(
    setup: &str,
    pose: &Matrix4<f64>,
    primary_transform: &Matrix4<f64>,
    target_transform: &Matrix4<f64>,
) -> Matrix4<f64> {
    if setup == "eye-in-hand" {
        primary_transform
            .try_inverse()
            .expect("primary transform should be invertible")
            * pose
                .try_inverse()
                .expect("pose transform should be invertible")
            * target_transform
    } else {
        primary_transform
            .try_inverse()
            .expect("primary transform should be invertible")
            * pose
            * target_transform
    }
}

fn project_point(
    intrinsics: &CameraIntrinsics,
    object_to_camera: &Matrix4<f64>,
    point: &Vector3<f64>,
) -> Vector2<f64> {
    let camera_point = object_to_camera.fixed_view::<3, 3>(0, 0) * point
        + object_to_camera.fixed_view::<3, 1>(0, 3);
    Vector2::new(
        intrinsics.fx * camera_point.x / camera_point.z + intrinsics.cx,
        intrinsics.fy * camera_point.y / camera_point.z + intrinsics.cy,
    )
}

#[cfg(test)]
fn translation_distance(left: &Matrix4<f64>, right: &Matrix4<f64>) -> f64 {
    let delta = matrix_to_se3(left)
        .inverse(None)
        .compose(&matrix_to_se3(right), None, None)
        .log(None);
    delta.rho().norm()
}

#[cfg(test)]
fn rotation_distance_deg(left: &Matrix4<f64>, right: &Matrix4<f64>) -> f64 {
    let delta = matrix_to_se3(left)
        .inverse(None)
        .compose(&matrix_to_se3(right), None, None)
        .log(None);
    delta.theta().norm().to_degrees()
}

fn optimize_handeye_from_measurements(
    setup: &str,
    intrinsics: &CameraIntrinsics,
    board_points: &[Vector3<f64>],
    observations: &[DetectionObservation],
    poses: &[Matrix4<f64>],
    measured_object_to_camera: &[Matrix4<f64>],
    depth_observations: &[Option<DepthObservation>],
) -> Result<HandeyeOptimization, String> {
    let (primary_init, secondary_init) =
        initialize_global_transforms(setup, &measured_object_to_camera, poses)?;

    let mut problem = Problem::new(JacobianMode::Dense);
    for (row, (pose, observation)) in poses.iter().zip(observations.iter()).enumerate() {
        let object_points = observation
            .corner_ids
            .iter()
            .map(|index| board_points[*index])
            .collect::<Vec<_>>();
        problem.add_residual_block(
            &["primary", "secondary"],
            Box::new(HandeyeProjectionFactor {
                setup: setup.to_string(),
                pose: *pose,
                object_points,
                image_points: observation.image_points.clone(),
                depth_observation: depth_observations.get(row).cloned().flatten(),
                intrinsics: intrinsics.clone(),
            }),
            None,
        );
    }

    let initial_values = HashMap::from([
        (
            "primary".to_string(),
            (ManifoldType::SE3, matrix_to_se3(&primary_init).into()),
        ),
        (
            "secondary".to_string(),
            (ManifoldType::SE3, matrix_to_se3(&secondary_init).into()),
        ),
    ]);
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(80)
        .with_cost_tolerance(1e-12)
        .with_parameter_tolerance(1e-12)
        .with_gradient_tolerance(1e-12)
        .with_damping(1e-3);
    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver
        .optimize(&problem, &initial_values)
        .map_err(|err| format!("Apex 求解失败: {err}"))?;
    let primary = variable_matrix(&result.parameters, "primary")?;
    let secondary = variable_matrix(&result.parameters, "secondary")?;

    let object_to_camera_derived = observations
        .iter()
        .zip(poses.iter())
        .map(|(_, pose)| derive_object_to_camera_matrix(setup, pose, &primary, &secondary))
        .collect::<Vec<_>>();
    let reprojection = compute_reprojection_metrics(
        intrinsics,
        board_points,
        observations,
        &object_to_camera_derived,
    );

    Ok(HandeyeOptimization {
        primary_transform: primary,
        secondary_transform: secondary,
        measured_object_to_camera: measured_object_to_camera.to_vec(),
        object_to_camera_derived,
        reprojection_mean_px: reprojection.mean_px,
        reprojection_rms_px: reprojection.rms_px,
    })
}

fn compute_reprojection_metrics(
    intrinsics: &CameraIntrinsics,
    board_points: &[Vector3<f64>],
    observations: &[DetectionObservation],
    object_to_camera_list: &[Matrix4<f64>],
) -> ReprojectionMetrics {
    let mut all_errors = Vec::new();
    let mut per_frame = Vec::with_capacity(observations.len());
    for (observation, object_to_camera) in observations.iter().zip(object_to_camera_list.iter()) {
        let errors = observation
            .corner_ids
            .iter()
            .zip(observation.image_points.iter())
            .map(|(corner_id, observed)| {
                let predicted =
                    project_point(intrinsics, object_to_camera, &board_points[*corner_id]);
                (predicted - observed).norm()
            })
            .collect::<Vec<_>>();
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        let rms =
            (errors.iter().map(|value| value * value).sum::<f64>() / errors.len() as f64).sqrt();
        let max = errors.iter().copied().fold(0.0_f64, f64::max);
        all_errors.extend(errors);
        per_frame.push((mean, rms, max));
    }
    let mean_px = all_errors.iter().sum::<f64>() / all_errors.len() as f64;
    let rms_px = (all_errors.iter().map(|value| value * value).sum::<f64>()
        / all_errors.len() as f64)
        .sqrt();
    ReprojectionMetrics {
        mean_px,
        rms_px,
        per_frame,
    }
}

trait ObservationLike {
    fn to_observation(&self) -> DetectionObservation;
}

impl ObservationLike for SyntheticDetection {
    fn to_observation(&self) -> DetectionObservation {
        DetectionObservation {
            index: self.index,
            image_path: self.image_path.clone(),
            corner_ids: self.corner_ids.clone(),
            image_points: self.image_points.clone(),
            marker_count: self.marker_count,
            used_chessboard_fallback: self.used_chessboard_fallback,
        }
    }
}

fn variable_matrix(
    parameters: &HashMap<String, apex_solver::core::problem::VariableEnum>,
    name: &str,
) -> Result<Matrix4<f64>, String> {
    let variable = parameters
        .get(name)
        .ok_or_else(|| format!("缺少优化变量 {name}"))?;
    Ok(SE3::from(variable.to_vector()).matrix())
}

fn initialize_global_transforms(
    setup: &str,
    measured_object_to_camera: &[Matrix4<f64>],
    poses: &[Matrix4<f64>],
) -> Result<(Matrix4<f64>, Matrix4<f64>), String> {
    let fallback_primary = if setup == "eye-in-hand" {
        Matrix4::identity()
    } else {
        poses[0]
            * measured_object_to_camera[0]
                .try_inverse()
                .ok_or_else(|| "PnP 初值不可逆".to_string())?
    };
    let mut seeds = vec![(
        fallback_primary,
        estimate_secondary_from_primary(
            setup,
            poses,
            measured_object_to_camera,
            &fallback_primary,
        )?,
    )];
    if setup == "eye-in-hand" {
        if let Ok(primary) = calibrate_handeye_primary(poses, measured_object_to_camera) {
            let secondary =
                estimate_secondary_from_primary(setup, poses, measured_object_to_camera, &primary)?;
            seeds.push((primary, secondary));
        }
    } else if let Ok(primary) = calibrate_handeye_primary(poses, measured_object_to_camera) {
        let secondary =
            estimate_secondary_from_primary(setup, poses, measured_object_to_camera, &primary)?;
        seeds.push((primary, secondary));
    }

    let mut best = None;
    for (primary_seed, secondary_seed) in seeds {
        let refined = optimize_handeye_constraints(
            setup,
            poses,
            measured_object_to_camera,
            &primary_seed,
            &secondary_seed,
        )?;
        let cost = handeye_constraint_cost(
            setup,
            poses,
            measured_object_to_camera,
            &refined.0,
            &refined.1,
        );
        if best.as_ref().is_none_or(|(_, best_cost)| cost < *best_cost) {
            best = Some((refined, cost));
        }
    }
    best.map(|(solution, _)| solution)
        .ok_or_else(|| "无法生成手眼标定初值".to_string())
}

fn average_transform(transforms: &[Matrix4<f64>]) -> Matrix4<f64> {
    let translation = transforms
        .iter()
        .map(|transform| Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]))
        .fold(Vector3::zeros(), |acc, value| acc + value)
        / transforms.len() as f64;
    let mut quaternion_sum = nalgebra::Vector4::zeros();
    for transform in transforms {
        let rotation =
            Rotation3::from_matrix_unchecked(transform.fixed_view::<3, 3>(0, 0).into_owned());
        let quat = UnitQuaternion::from_rotation_matrix(&rotation);
        let coords = quat.coords;
        if quaternion_sum.dot(&coords) < 0.0 {
            quaternion_sum -= coords;
        } else {
            quaternion_sum += coords;
        }
    }
    let quaternion_sum = quaternion_sum / quaternion_sum.norm();
    let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        quaternion_sum[3],
        quaternion_sum[0],
        quaternion_sum[1],
        quaternion_sum[2],
    ));
    compose_matrix(translation, rotation.to_rotation_matrix().into_inner())
}

fn estimate_secondary_from_primary(
    setup: &str,
    poses: &[Matrix4<f64>],
    measured_object_to_camera: &[Matrix4<f64>],
    primary: &Matrix4<f64>,
) -> Result<Matrix4<f64>, String> {
    let mut candidates = Vec::with_capacity(poses.len());
    for (pose, measured) in poses.iter().zip(measured_object_to_camera.iter()) {
        candidates.push(if setup == "eye-in-hand" {
            pose * primary * measured
        } else {
            pose.try_inverse()
                .ok_or_else(|| "机器人位姿不可逆".to_string())?
                * primary
                * measured
        });
    }
    Ok(average_transform(&candidates))
}

fn handeye_constraint_cost(
    setup: &str,
    poses: &[Matrix4<f64>],
    measured_object_to_camera: &[Matrix4<f64>],
    primary: &Matrix4<f64>,
    secondary: &Matrix4<f64>,
) -> f64 {
    poses
        .iter()
        .zip(measured_object_to_camera.iter())
        .map(|(pose, measured)| {
            let predicted = derive_object_to_camera_matrix(setup, pose, primary, secondary);
            let delta = matrix_to_se3(measured)
                .inverse(None)
                .compose(&matrix_to_se3(&predicted), None, None)
                .log(None);
            let residual: DVector<f64> = delta.into();
            residual.norm_squared()
        })
        .sum()
}

fn compute_pose_errors(
    setup: &str,
    poses: &[Matrix4<f64>],
    measured_object_to_camera: &[Matrix4<f64>],
    primary: &Matrix4<f64>,
    secondary: &Matrix4<f64>,
) -> PoseErrorMetrics {
    let per_frame = poses
        .iter()
        .zip(measured_object_to_camera.iter())
        .map(|(pose, measured)| {
            let predicted = derive_object_to_camera_matrix(setup, pose, primary, secondary);
            let delta = measured.try_inverse().unwrap_or_else(Matrix4::identity) * predicted;
            let rotation =
                Rotation3::from_matrix_unchecked(delta.fixed_view::<3, 3>(0, 0).into_owned());
            let translation =
                Vector3::new(delta[(0, 3)], delta[(1, 3)], delta[(2, 3)]).norm() * 1000.0;
            (translation, rotation.angle().to_degrees())
        })
        .collect::<Vec<_>>();
    let count = per_frame.len().max(1) as f64;
    PoseErrorMetrics {
        translation_mean_mm: per_frame
            .iter()
            .map(|(translation, _)| *translation)
            .sum::<f64>()
            / count,
        rotation_mean_deg: per_frame.iter().map(|(_, rotation)| *rotation).sum::<f64>() / count,
        per_frame,
    }
}

fn should_flip_chessboard_by_markers(
    marker_centers: &[Vector2<f64>],
    marker_ids: &[i32],
) -> bool {
    if marker_centers.len() < 2 || marker_centers.len() != marker_ids.len() {
        return false;
    }
    let mut order = (0..marker_centers.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        marker_centers[*left]
            .y
            .partial_cmp(&marker_centers[*right].y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top = order.into_iter().take(8).collect::<Vec<_>>();
    if top.len() < 2 {
        return false;
    }
    let mut top_by_x = top;
    top_by_x.sort_by(|left, right| {
        marker_centers[*left]
            .x
            .partial_cmp(&marker_centers[*right].x)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ids = top_by_x
        .iter()
        .map(|index| marker_ids[*index] as f64)
        .collect::<Vec<_>>();
    ids.windows(2).map(|pair| pair[1] - pair[0]).sum::<f64>() < 0.0
}

fn apply_homography_2d(h: &Matrix3<f64>, point: &Vector2<f64>) -> Option<Vector2<f64>> {
    let mapped = h * nalgebra::Vector3::new(point.x, point.y, 1.0);
    if mapped.z.abs() < 1e-9 {
        None
    } else {
        Some(Vector2::new(mapped.x / mapped.z, mapped.y / mapped.z))
    }
}

fn reorder_chessboard_corners_via_homography(
    detected: &[Vector2<f64>],
    board_points: &[Vector3<f64>],
    square_length: f64,
    image_to_board: &Matrix3<f64>,
) -> Option<Vec<Vector2<f64>>> {
    if detected.len() != board_points.len() || square_length <= 0.0 {
        return None;
    }
    let cols_x = board_points
        .iter()
        .map(|point| (point.x / square_length).round() as i32 - 1)
        .max()
        .map(|value| value + 1)? as usize;
    let rows_y = board_points
        .iter()
        .map(|point| (point.y / square_length).round() as i32 - 1)
        .max()
        .map(|value| value + 1)? as usize;
    if cols_x * rows_y != board_points.len() {
        return None;
    }

    let mut reordered = vec![Vector2::zeros(); detected.len()];
    let mut assigned = vec![false; detected.len()];
    for point in detected {
        let mapped = apply_homography_2d(image_to_board, point)?;
        let col = (mapped.x / square_length).round() as i32 - 1;
        let row = (mapped.y / square_length).round() as i32 - 1;
        if !(0..cols_x as i32).contains(&col) || !(0..rows_y as i32).contains(&row) {
            return None;
        }
        let index = row as usize * cols_x + col as usize;
        if assigned[index] {
            return None;
        }
        let expected = board_points[index];
        if (mapped.x - expected.x).abs() > square_length * 0.6
            || (mapped.y - expected.y).abs() > square_length * 0.6
        {
            return None;
        }
        reordered[index] = *point;
        assigned[index] = true;
    }
    assigned.into_iter().all(|value| value).then_some(reordered)
}

fn board_marker_reference_centers(
    board: &objdetect::CharucoBoard,
) -> Result<HashMap<i32, Vector2<f64>>, String> {
    let ids = board
        .get_ids()
        .map_err(|err| format!("读取 board marker ids 失败: {err}"))?;
    let obj_points = board
        .get_obj_points()
        .map_err(|err| format!("读取 board marker 点失败: {err}"))?;
    let mut centers = HashMap::new();
    for (marker_id, corners) in ids.iter().zip(obj_points.iter()) {
        if corners.is_empty() {
            continue;
        }
        let mean = corners.iter().fold(Vector2::zeros(), |acc, point| {
            acc + Vector2::new(point.x as f64, point.y as f64)
        }) / corners.len() as f64;
        centers.insert(marker_id, mean);
    }
    Ok(centers)
}

fn marker_homography_to_board(
    board: &objdetect::CharucoBoard,
    marker_centers: &[Vector2<f64>],
    marker_ids: &[i32],
) -> Result<Option<Matrix3<f64>>, String> {
    if marker_centers.len() < 4 || marker_centers.len() != marker_ids.len() {
        return Ok(None);
    }
    let reference = board_marker_reference_centers(board)?;
    let mut image_points = Vec::<Point2f>::new();
    let mut board_points = Vec::<Point2f>::new();
    for (center, marker_id) in marker_centers.iter().zip(marker_ids.iter()) {
        if let Some(board_center) = reference.get(marker_id) {
            image_points.push(Point2f::new(center.x as f32, center.y as f32));
            board_points.push(Point2f::new(board_center.x as f32, board_center.y as f32));
        }
    }
    if image_points.len() < 4 {
        return Ok(None);
    }
    let mut mask = Mat::default();
    let homography = calib3d::find_homography_def(
        &Vector::<Point2f>::from_iter(image_points),
        &Vector::<Point2f>::from_iter(board_points),
        &mut mask,
    )
    .map_err(|err| format!("估计 marker 单应矩阵失败: {err}"))?;
    if homography.empty() {
        return Ok(None);
    }
    Ok(Some(mat_to_matrix3(&homography)?))
}

fn reorder_chessboard_corners_for_board(
    mut corners: Vec<Vector2<f64>>,
    squares_x: usize,
    squares_y: usize,
    marker_info: Option<(&[Vector2<f64>], &[i32])>,
) -> Vec<Vector2<f64>> {
    if let Some((marker_centers, marker_ids)) = marker_info {
        if should_flip_chessboard_by_markers(marker_centers, marker_ids) {
            corners.reverse();
        }
    }

    let cols_x = squares_x.saturating_sub(1);
    let rows_y = squares_y.saturating_sub(1);
    if cols_x == 0 || rows_y == 0 || corners.len() != cols_x * rows_y {
        return corners;
    }

    let cols_per_scan = cols_x;
    let mut reordered = vec![Vector2::zeros(); corners.len()];
    for (det_idx, point) in corners.into_iter().enumerate() {
        let scan = det_idx / cols_per_scan;
        let pos = det_idx % cols_per_scan;
        let pattern_idx = pos * cols_x + scan;
        if pattern_idx < reordered.len() {
            reordered[pattern_idx] = point;
        }
    }
    reordered
}

const MIN_DETECTION_CORNERS: usize = 8;
const MIN_CHARUCO_MARKERS: usize = 4;
const MAX_PNP_REPROJECTION_PX: f64 = 1.5;
const MIN_FALLBACK_COVERAGE: f64 = 0.7;

fn detection_passes_quality_gate(
    detection: &DetectionObservation,
    total_board_corners: usize,
    reprojection_mean_px: f64,
) -> bool {
    if detection.corner_ids.len() < MIN_DETECTION_CORNERS {
        return false;
    }
    if !reprojection_mean_px.is_finite() || reprojection_mean_px > MAX_PNP_REPROJECTION_PX {
        return false;
    }
    if detection.used_chessboard_fallback {
        return total_board_corners > 0
            && detection.corner_ids.len() as f64 / total_board_corners as f64
                >= MIN_FALLBACK_COVERAGE;
    }
    detection.marker_count >= MIN_CHARUCO_MARKERS
}

fn filter_inconsistent_measurements(
    setup: &str,
    poses: &[Matrix4<f64>],
    measured_object_to_camera: &[Matrix4<f64>],
) -> Result<Vec<usize>, String> {
    let mut consistent = (0..poses.len()).collect::<Vec<_>>();
    for _ in 0..3 {
        if consistent.len() < 3 {
            break;
        }
        let fit_poses = consistent
            .iter()
            .map(|index| poses[*index])
            .collect::<Vec<_>>();
        let fit_measured = consistent
            .iter()
            .map(|index| measured_object_to_camera[*index])
            .collect::<Vec<_>>();
        let (primary, secondary) = initialize_global_transforms(setup, &fit_measured, &fit_poses)?;
        let errors = consistent
            .iter()
            .map(|index| {
                let predicted =
                    derive_object_to_camera_matrix(setup, &poses[*index], &primary, &secondary);
                let delta = measured_object_to_camera[*index]
                    .try_inverse()
                    .unwrap_or_else(Matrix4::identity)
                    * predicted;
                let translation = Vector3::new(delta[(0, 3)], delta[(1, 3)], delta[(2, 3)]).norm();
                let rotation =
                    Rotation3::from_matrix_unchecked(delta.fixed_view::<3, 3>(0, 0).into_owned())
                        .angle();
                (*index, translation, rotation)
            })
            .collect::<Vec<_>>();
        let translation_values = errors
            .iter()
            .map(|(_, translation, _)| *translation)
            .collect::<Vec<_>>();
        let rotation_values = errors
            .iter()
            .map(|(_, _, rotation)| *rotation)
            .collect::<Vec<_>>();
        let translation_limit = robust_limit(&translation_values, 0.020, 3.5);
        let rotation_limit = robust_limit(&rotation_values, 2.0_f64.to_radians(), 3.5);
        let worst = errors
            .iter()
            .enumerate()
            .map(|(row, (_, translation, rotation))| {
                let score = (*translation / translation_limit).max(*rotation / rotation_limit);
                (row, score)
            })
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
        let Some((worst_row, worst_score)) = worst else {
            break;
        };
        if worst_score <= 1.0 {
            break;
        }
        consistent.remove(worst_row);
    }
    Ok(consistent)
}

fn robust_limit(values: &[f64], absolute_limit: f64, mad_scale: f64) -> f64 {
    if values.is_empty() {
        return absolute_limit;
    }
    let center = median(values);
    let deviations = values
        .iter()
        .map(|value| (value - center).abs())
        .collect::<Vec<_>>();
    let mad = median(&deviations);
    if mad < 1e-12 {
        absolute_limit
    } else {
        absolute_limit.min(center + mad_scale * 1.4826 * mad)
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn compute_reference_consistency(
    setup: &str,
    board_points: &[Vector3<f64>],
    observations: &[DetectionObservation],
    poses: &[Matrix4<f64>],
    primary: &Matrix4<f64>,
    measured_object_to_camera: &[Matrix4<f64>],
    depth_observations: &[Option<DepthObservation>],
) -> Option<DistanceStats> {
    let mut points_by_corner: HashMap<usize, Vec<(usize, Vector3<f64>)>> = HashMap::new();
    let mut per_image_points = Vec::with_capacity(observations.len());

    for (row, observation) in observations.iter().enumerate() {
        let reference = if setup == "eye-in-hand" {
            poses[row] * primary
        } else {
            poses[row].try_inverse()? * primary
        };
        let mut image_points = Vec::new();
        if let Some(Some(depth)) = depth_observations.get(row) {
            for (corner_id, point_camera) in depth.corner_ids.iter().zip(depth.camera_points.iter()) {
                let point_reference = reference.fixed_view::<3, 3>(0, 0) * point_camera
                    + reference.fixed_view::<3, 1>(0, 3);
                points_by_corner
                    .entry(*corner_id)
                    .or_default()
                    .push((row, point_reference));
                image_points.push((*corner_id, point_reference));
            }
        } else {
            let measured = measured_object_to_camera[row];
            for corner_id in &observation.corner_ids {
                let point_object = board_points[*corner_id];
                let point_camera = measured.fixed_view::<3, 3>(0, 0) * point_object
                    + measured.fixed_view::<3, 1>(0, 3);
                let point_reference = reference.fixed_view::<3, 3>(0, 0) * point_camera
                    + reference.fixed_view::<3, 1>(0, 3);
                points_by_corner
                    .entry(*corner_id)
                    .or_default()
                    .push((row, point_reference));
                image_points.push((*corner_id, point_reference));
            }
        }
        per_image_points.push(image_points);
    }

    let means = points_by_corner
        .iter()
        .filter_map(|(corner_id, points)| {
            if points.len() < 2 {
                return None;
            }
            let mean = points
                .iter()
                .map(|(_, point)| *point)
                .fold(Vector3::zeros(), |acc, point| acc + point)
                / points.len() as f64;
            Some((*corner_id, mean))
        })
        .collect::<HashMap<_, _>>();
    if means.is_empty() {
        return None;
    }

    let mut all_errors = Vec::new();
    let mut per_frame = Vec::with_capacity(per_image_points.len());
    for image_points in per_image_points {
        let errors = image_points
            .iter()
            .filter_map(|(corner_id, point)| means.get(corner_id).map(|mean| (point - mean).norm()))
            .collect::<Vec<_>>();
        all_errors.extend(errors.iter().copied());
        per_frame.push(distance_stats_tuple(&errors));
    }
    if all_errors.is_empty() {
        return None;
    }
    let (count, mean_m, rms_m, max_m) = distance_stats_tuple(&all_errors)?;
    Some(DistanceStats {
        count,
        mean_m,
        rms_m,
        max_m,
        per_frame,
    })
}

fn distance_stats_tuple(values: &[f64]) -> Option<(usize, f64, f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let rms = (values.iter().map(|value| value * value).sum::<f64>() / count as f64).sqrt();
    let max = values.iter().copied().fold(0.0_f64, f64::max);
    Some((count, mean, rms, max))
}

fn optimize_handeye_constraints(
    setup: &str,
    poses: &[Matrix4<f64>],
    measured_object_to_camera: &[Matrix4<f64>],
    primary_init: &Matrix4<f64>,
    secondary_init: &Matrix4<f64>,
) -> Result<(Matrix4<f64>, Matrix4<f64>), String> {
    let mut problem = Problem::new(JacobianMode::Dense);
    for (pose, measured) in poses.iter().zip(measured_object_to_camera.iter()) {
        problem.add_residual_block(
            &["primary", "secondary"],
            Box::new(HandeyeConstraintFactor {
                setup: setup.to_string(),
                pose: *pose,
                measured_object_to_camera: *measured,
            }),
            None,
        );
    }
    let initial_values = HashMap::from([
        (
            "primary".to_string(),
            (ManifoldType::SE3, matrix_to_se3(primary_init).into()),
        ),
        (
            "secondary".to_string(),
            (ManifoldType::SE3, matrix_to_se3(secondary_init).into()),
        ),
    ]);
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(60)
        .with_cost_tolerance(1e-12)
        .with_parameter_tolerance(1e-12)
        .with_gradient_tolerance(1e-12)
        .with_damping(1e-4);
    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver
        .optimize(&problem, &initial_values)
        .map_err(|err| format!("约束初始优化失败: {err}"))?;
    Ok((
        variable_matrix(&result.parameters, "primary")?,
        variable_matrix(&result.parameters, "secondary")?,
    ))
}

fn solve_pnp_measurement(
    intrinsics: &CameraIntrinsics,
    board_points: &[Vector3<f64>],
    observation: &DetectionObservation,
) -> Result<Matrix4<f64>, String> {
    let object_points = observation
        .corner_ids
        .iter()
        .map(|index| {
            let point = board_points[*index];
            Point3d::new(point.x, point.y, point.z)
        })
        .collect::<Vector<Point3d>>();
    let image_points = observation
        .image_points
        .iter()
        .map(|point| Point2d::new(point.x, point.y))
        .collect::<Vector<Point2d>>();
    let camera_matrix = camera_matrix_mat(intrinsics)?;
    // Observations are detected on undistorted images, so the PnP solve uses zero distortion here.
    let distortion = zero_distortion_mat()?;
    let mut rvec = Mat::default();
    let mut tvec = Mat::default();
    let ok = calib3d::solve_pnp(
        &object_points,
        &image_points,
        &camera_matrix,
        &distortion,
        &mut rvec,
        &mut tvec,
        false,
        calib3d::SOLVEPNP_ITERATIVE,
    )
    .map_err(|err| format!("solvePnP 失败: {err}"))?;
    if !ok {
        return Err(format!("solvePnP 失败: {}", observation.image_path));
    }
    let mut rotation = Mat::default();
    calib3d::rodrigues_def(&rvec, &mut rotation).map_err(|err| format!("Rodrigues 失败: {err}"))?;
    let mut transform = Matrix4::identity();
    transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&mat_to_matrix3(&rotation)?);
    let translation = mat_to_vector3(&tvec)?;
    transform[(0, 3)] = translation.x;
    transform[(1, 3)] = translation.y;
    transform[(2, 3)] = translation.z;
    Ok(transform)
}

fn solve_from_depth(
    intrinsics: &CameraIntrinsics,
    board_points: &[Vector3<f64>],
    observation: &DetectionObservation,
    depth_path: &str,
) -> Result<(Matrix4<f64>, DepthObservation), String> {
    let (pixels, width, height) = load_depth_pixels(&PathBuf::from(depth_path))?;

    let n = observation.corner_ids.len();
    let mut board_pts = Vec::with_capacity(n);
    let mut cam_pts = Vec::with_capacity(n);
    let mut depth_corner_ids = Vec::with_capacity(n);
    let mut depth_object_points = Vec::with_capacity(n);
    let mut corner_cam_points = Vec::with_capacity(n);

    for (&corner_id, point) in observation
        .corner_ids
        .iter()
        .zip(observation.image_points.iter())
    {
        let u = point.x;
        let v = point.y;
        let px = u.round() as i32;
        let py = v.round() as i32;

        if px < 0 || py < 0 || px as u32 >= width || py as u32 >= height {
            continue;
        }

        let idx = (py as u32 * width + px as u32) as usize;
        let depth_mm = pixels[idx] as f64;
        if depth_mm < 1.0 {
            continue;
        }
        let depth_m = depth_mm / 1000.0;

        // Back-project pixel (u, v, depth) to 3D camera frame
        let x = (u - intrinsics.cx) * depth_m / intrinsics.fx;
        let y = (v - intrinsics.cy) * depth_m / intrinsics.fy;
        let z = depth_m;

        let board_pt = board_points[corner_id];
        board_pts.push(Vector3::new(board_pt.x, board_pt.y, board_pt.z));
        cam_pts.push(Vector3::new(x, y, z));
        depth_corner_ids.push(corner_id);
        depth_object_points.push(Vector3::new(board_pt.x, board_pt.y, board_pt.z));
        corner_cam_points.push(Vector3::new(x, y, z));
    }

    if board_pts.len() < 4 {
        return Err(format!(
            "深度图像中有效角点不足 4 个（共检测到 {n} 个，深度有效 {} 个）",
            board_pts.len()
        ));
    }

    // Umeyama/Procrustes: find T_O2C = [R|t] aligning board frame → camera frame
    let m = board_pts.len() as f64;
    let centroid_board = board_pts.iter().fold(Vector3::zeros(), |s, p| s + p) / m;
    let centroid_cam = cam_pts.iter().fold(Vector3::zeros(), |s, p| s + p) / m;

    let mut h = Matrix3::zeros();
    for i in 0..board_pts.len() {
        let qb = board_pts[i] - centroid_board;
        let qc = cam_pts[i] - centroid_cam;
        h += qc * qb.transpose();
    }

    let svd = nalgebra::linalg::SVD::new(h, true, true);
    let r = {
        let u = svd.u.as_ref().ok_or("SVD U 分解失败")?;
        let vt = svd.v_t.as_ref().ok_or("SVD V^T 分解失败")?;
        u * vt
    };
    // Ensure det(R) = 1 (rotation, not reflection)
    let r = if r.determinant() < 0.0 {
        let mut u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        u.column_mut(2).neg_mut();
        u * vt
    } else {
        r
    };

    let t = centroid_cam - r * centroid_board;

    let mut transform = Matrix4::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    transform[(0, 3)] = t.x;
    transform[(1, 3)] = t.y;
    transform[(2, 3)] = t.z;

    Ok((
        transform,
        DepthObservation {
            corner_ids: depth_corner_ids,
            object_points: depth_object_points,
            camera_points: corner_cam_points,
        },
    ))
}

fn parse_depth_mode(use_depth: &str) -> Result<DepthMode, String> {
    let normalized = use_depth.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "off" | "false" | "0" | "no" => Ok(DepthMode::Off),
        "optional" | "on" | "true" | "1" | "yes" => Ok(DepthMode::Optional),
        "required" => Ok(DepthMode::Required),
        other => Err(format!("不支持的深度模式: {other}")),
    }
}

fn requested_depth_path(request: &CharucoRequest) -> Option<&Path> {
    request.depth_path.as_deref().map(Path::new)
}

fn depth_path_candidates_for_rgb(rgb_path: &Path) -> Vec<PathBuf> {
    let directory = rgb_path.parent().unwrap_or_else(|| Path::new("."));
    let base = rgb_path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or_default();
    let stem = rgb_path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or_default();

    let (prefix, depth_root) = if base.contains("_Color") {
        (
            base.split("_Color").next().unwrap_or_default().to_string(),
            base.replace("_Color", "_Depth")
                .trim_end_matches(".png")
                .trim_end_matches(".jpg")
                .trim_end_matches(".jpeg")
                .to_string(),
        )
    } else {
        (
            stem.split('_').next().unwrap_or_default().to_string(),
            format!("{stem}_Depth"),
        )
    };

    let mut candidates = Vec::new();
    let mut push_candidate = |name: String| {
        let path = directory.join(name);
        if !candidates.iter().any(|candidate| candidate == &path) {
            candidates.push(path);
        }
    };
    push_candidate(format!("{depth_root}.png"));
    push_candidate(format!("{depth_root}.jpg"));
    push_candidate(format!("{depth_root}.jpeg"));
    push_candidate(format!("{depth_root}.raw"));
    if !prefix.is_empty() {
        push_candidate(format!("{prefix}.raw"));
    }
    candidates
}

fn match_depth_path_for_rgb(rgb_path: &Path) -> Result<PathBuf, String> {
    let candidates = depth_path_candidates_for_rgb(rgb_path);
    candidates
        .into_iter()
        .find(|candidate| candidate.exists())
        .ok_or_else(|| format!("未找到与 {:?} 匹配的深度图", rgb_path.file_name()))
}

fn resolve_object_to_camera_measurements(
    depth_mode: &DepthMode,
    intrinsics: &CameraIntrinsics,
    board_points: &[Vector3<f64>],
    observations: &[DetectionObservation],
) -> Result<(Vec<Matrix4<f64>>, Vec<Option<DepthObservation>>, bool), String> {
    let mut measurements = Vec::with_capacity(observations.len());
    let mut depth_observations = Vec::with_capacity(observations.len());
    let mut depth_used = false;

    for observation in observations {
        if *depth_mode != DepthMode::Off {
            let rgb_path = PathBuf::from(&observation.image_path);
            match match_depth_path_for_rgb(&rgb_path) {
                Ok(depth_path) => match solve_from_depth(
                    intrinsics,
                    board_points,
                    observation,
                    &depth_path.to_string_lossy(),
                ) {
                    Ok((transform, depth_observation)) => {
                        measurements.push(transform);
                        depth_observations.push(Some(depth_observation));
                        depth_used = true;
                        continue;
                    }
                    Err(_) => {
                        // If depth samples are unusable, fall back to the PnP estimate.
                    }
                },
                Err(err) => {
                    if *depth_mode == DepthMode::Required {
                        return Err(err);
                    }
                }
            }
        }

        measurements.push(solve_pnp_measurement(
            intrinsics,
            board_points,
            observation,
        )?);
        depth_observations.push(None);
    }

    Ok((measurements, depth_observations, depth_used))
}

fn calibrate_handeye_primary(
    poses: &[Matrix4<f64>],
    measured_object_to_camera: &[Matrix4<f64>],
) -> Result<Matrix4<f64>, String> {
    let mut r_gripper2base = Vector::<Mat>::new();
    let mut t_gripper2base = Vector::<Mat>::new();
    let mut r_target2cam = Vector::<Mat>::new();
    let mut t_target2cam = Vector::<Mat>::new();
    for pose in poses {
        let (rvec, tvec) = matrix_to_rvec_tvec(pose)?;
        r_gripper2base.push(rvec);
        t_gripper2base.push(tvec);
    }
    for transform in measured_object_to_camera {
        let (rvec, tvec) = matrix_to_rvec_tvec(transform)?;
        r_target2cam.push(rvec);
        t_target2cam.push(tvec);
    }
    let mut r_cam2gripper = Mat::default();
    let mut t_cam2gripper = Mat::default();
    calib3d::calibrate_hand_eye(
        &r_gripper2base,
        &t_gripper2base,
        &r_target2cam,
        &t_target2cam,
        &mut r_cam2gripper,
        &mut t_cam2gripper,
        calib3d::HandEyeCalibrationMethod::CALIB_HAND_EYE_TSAI,
    )
    .map_err(|err| format!("calibrateHandEye 失败: {err}"))?;
    matrix_from_rvec_tvec(&r_cam2gripper, &t_cam2gripper)
}

fn matrix_to_rvec_tvec(matrix: &Matrix4<f64>) -> Result<(Mat, Mat), String> {
    let rotation = Mat::from_slice_2d(&[
        &[matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
        &[matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
        &[matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
    ])
    .map_err(|err| format!("创建旋转矩阵失败: {err}"))?;
    let mut rvec = Mat::default();
    calib3d::rodrigues_def(&rotation, &mut rvec).map_err(|err| format!("Rodrigues 失败: {err}"))?;
    let tvec = Mat::from_slice_2d(&[&[matrix[(0, 3)]], &[matrix[(1, 3)]], &[matrix[(2, 3)]]])
        .map_err(|err| format!("创建平移向量失败: {err}"))?;
    Ok((rvec, tvec))
}

fn matrix_from_rvec_tvec(rvec: &Mat, tvec: &Mat) -> Result<Matrix4<f64>, String> {
    let rotation = if rvec.rows() == 3 && rvec.cols() == 3 {
        rvec.try_clone()
            .map_err(|err| format!("复制旋转矩阵失败: {err}"))?
    } else {
        let mut rotation = Mat::default();
        calib3d::rodrigues_def(rvec, &mut rotation)
            .map_err(|err| format!("Rodrigues 失败: {err}"))?;
        rotation
    };
    let mut transform = Matrix4::identity();
    transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&mat_to_matrix3(&rotation)?);
    let translation = mat_to_vector3(tvec)?;
    transform[(0, 3)] = translation.x;
    transform[(1, 3)] = translation.y;
    transform[(2, 3)] = translation.z;
    Ok(transform)
}

fn mat_to_matrix3(mat: &Mat) -> Result<Matrix3<f64>, String> {
    let mut matrix = Matrix3::zeros();
    for row in 0..3 {
        for col in 0..3 {
            matrix[(row, col)] = *mat
                .at_2d::<f64>(row as i32, col as i32)
                .map_err(|err| format!("读取旋转矩阵失败: {err}"))?;
        }
    }
    Ok(matrix)
}

fn mat_to_vector3(mat: &Mat) -> Result<Vector3<f64>, String> {
    if mat.rows() == 3 && mat.cols() == 1 {
        return Ok(Vector3::new(
            *mat.at_2d::<f64>(0, 0)
                .map_err(|err| format!("读取平移向量失败: {err}"))?,
            *mat.at_2d::<f64>(1, 0)
                .map_err(|err| format!("读取平移向量失败: {err}"))?,
            *mat.at_2d::<f64>(2, 0)
                .map_err(|err| format!("读取平移向量失败: {err}"))?,
        ));
    }
    if mat.rows() == 1 && mat.cols() == 3 {
        return Ok(Vector3::new(
            *mat.at_2d::<f64>(0, 0)
                .map_err(|err| format!("读取平移向量失败: {err}"))?,
            *mat.at_2d::<f64>(0, 1)
                .map_err(|err| format!("读取平移向量失败: {err}"))?,
            *mat.at_2d::<f64>(0, 2)
                .map_err(|err| format!("读取平移向量失败: {err}"))?,
        ));
    }
    Err(format!(
        "不支持的平移向量形状: {}x{}",
        mat.rows(),
        mat.cols()
    ))
}

fn camera_matrix_mat(intrinsics: &CameraIntrinsics) -> Result<Mat, String> {
    Mat::from_slice_2d(&[
        &[intrinsics.fx, 0.0, intrinsics.cx],
        &[0.0, intrinsics.fy, intrinsics.cy],
        &[0.0, 0.0, 1.0],
    ])
    .map_err(|err| format!("创建相机矩阵失败: {err}"))
}

fn distortion_mat(intrinsics: &CameraIntrinsics) -> Result<Mat, String> {
    let distortion = intrinsics
        .distortion_coefficients
        .clone()
        .unwrap_or_else(|| vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let rows = distortion.iter().map(|value| [*value]).collect::<Vec<_>>();
    Mat::from_slice_2d(&rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>())
        .map_err(|err| format!("创建畸变矩阵失败: {err}"))
}

fn zero_distortion_mat() -> Result<Mat, String> {
    Mat::from_slice_2d(&[&[0.0_f64], &[0.0], &[0.0], &[0.0], &[0.0]])
        .map_err(|err| format!("创建零畸变矩阵失败: {err}"))
}

fn predefined_dictionary_type(name: &str) -> objdetect::PredefinedDictionaryType {
    match name {
        "DICT_4X4_50" => objdetect::PredefinedDictionaryType::DICT_4X4_50,
        "DICT_4X4_100" => objdetect::PredefinedDictionaryType::DICT_4X4_100,
        "DICT_4X4_250" => objdetect::PredefinedDictionaryType::DICT_4X4_250,
        "DICT_4X4_1000" => objdetect::PredefinedDictionaryType::DICT_4X4_1000,
        "DICT_5X5_50" => objdetect::PredefinedDictionaryType::DICT_5X5_50,
        "DICT_5X5_100" => objdetect::PredefinedDictionaryType::DICT_5X5_100,
        "DICT_5X5_250" => objdetect::PredefinedDictionaryType::DICT_5X5_250,
        "DICT_5X5_1000" => objdetect::PredefinedDictionaryType::DICT_5X5_1000,
        "DICT_6X6_50" => objdetect::PredefinedDictionaryType::DICT_6X6_50,
        "DICT_6X6_100" => objdetect::PredefinedDictionaryType::DICT_6X6_100,
        "DICT_6X6_250" => objdetect::PredefinedDictionaryType::DICT_6X6_250,
        "DICT_6X6_1000" => objdetect::PredefinedDictionaryType::DICT_6X6_1000,
        "DICT_7X7_50" => objdetect::PredefinedDictionaryType::DICT_7X7_50,
        "DICT_7X7_100" => objdetect::PredefinedDictionaryType::DICT_7X7_100,
        "DICT_7X7_250" => objdetect::PredefinedDictionaryType::DICT_7X7_250,
        "DICT_7X7_1000" => objdetect::PredefinedDictionaryType::DICT_7X7_1000,
        _ => objdetect::PredefinedDictionaryType::DICT_5X5_50,
    }
}

fn build_charuco_board(
    squares_x: usize,
    squares_y: usize,
    square_length: f64,
    marker_length: f64,
    aruco_dict: &str,
) -> Result<objdetect::CharucoBoard, String> {
    let dictionary = objdetect::get_predefined_dictionary(predefined_dictionary_type(aruco_dict))
        .map_err(|err| format!("加载 ArUco 字典失败: {err}"))?;
    let mut board = objdetect::CharucoBoard::new_def(
        core::Size::new(squares_x as i32, squares_y as i32),
        square_length as f32,
        marker_length as f32,
        &dictionary,
    )
    .map_err(|err| format!("创建 ChArUco 板失败: {err}"))?;
    board
        .set_legacy_pattern(true)
        .map_err(|err| format!("设置 legacy pattern 失败: {err}"))?;
    Ok(board)
}

fn board_points_from_charuco(board: &objdetect::CharucoBoard) -> Result<Vec<Vector3<f64>>, String> {
    let corners = board
        .get_chessboard_corners()
        .map_err(|err| format!("读取 ChArUco 角点模型失败: {err}"))?;
    Ok(corners
        .iter()
        .map(|point| Vector3::new(point.x as f64, point.y as f64, point.z as f64))
        .collect())
}

fn preprocess_detection_gray(undistorted: &Mat) -> Result<Mat, String> {
    let mut gray = Mat::default();
    imgproc::cvt_color(
        undistorted,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .map_err(|err| format!("灰度转换失败: {err}"))?;
    Ok(gray)
}

fn configured_charuco_detector_params() -> Result<objdetect::DetectorParameters, String> {
    let mut detector_params = objdetect::DetectorParameters::default()
        .map_err(|err| format!("创建 ArUco 检测参数失败: {err}"))?;
    detector_params.set_corner_refinement_method(1);
    detector_params.set_corner_refinement_win_size(5);
    detector_params.set_corner_refinement_max_iterations(30);
    detector_params.set_corner_refinement_min_accuracy(0.001);
    detector_params.set_adaptive_thresh_win_size_min(3);
    detector_params.set_adaptive_thresh_win_size_max(23);
    detector_params.set_adaptive_thresh_win_size_step(10);
    detector_params.set_min_marker_perimeter_rate(0.02);
    detector_params.set_max_marker_perimeter_rate(4.0);
    Ok(detector_params)
}

fn detect_charuco_observation(
    image_path: &str,
    board: &objdetect::CharucoBoard,
    intrinsics: &CameraIntrinsics,
    squares_x: usize,
    squares_y: usize,
) -> Result<Option<DetectionObservation>, String> {
    fn refine_corners_subpix(gray: &Mat, corners: &mut Mat, label: &str) -> Result<(), String> {
        let criteria = core::TermCriteria::new(
            core::TermCriteria_Type::COUNT as i32 + core::TermCriteria_Type::EPS as i32,
            30,
            0.001,
        )
        .map_err(|err| format!("创建{label}亚像素终止条件失败: {err}"))?;
        imgproc::corner_sub_pix(
            gray,
            corners,
            core::Size::new(5, 5),
            core::Size::new(-1, -1),
            criteria,
        )
        .map_err(|err| format!("{label}亚像素优化失败: {err}"))?;
        Ok(())
    }

    let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)
        .map_err(|err| format!("读取图像失败: {err}"))?;
    if image.empty() {
        return Ok(None);
    }
    let camera_matrix = camera_matrix_mat(intrinsics)?;
    let distortion = distortion_mat(intrinsics)?;
    let mut undistorted = Mat::default();
    calib3d::undistort(
        &image,
        &mut undistorted,
        &camera_matrix,
        &distortion,
        &camera_matrix,
    )
    .map_err(|err| format!("图像去畸变失败: {err}"))?;
    let gray = preprocess_detection_gray(&undistorted)?;
    let mut charuco_params = objdetect::CharucoParameters::default()
        .map_err(|err| format!("创建 Charuco 参数失败: {err}"))?;
    charuco_params.set_camera_matrix(
        camera_matrix
            .try_clone()
            .map_err(|err| format!("复制相机矩阵失败: {err}"))?,
    );
    charuco_params.set_dist_coeffs(zero_distortion_mat()?);
    let detector_params = configured_charuco_detector_params()?;
    let refine_params = objdetect::RefineParameters::new_def()
        .map_err(|err| format!("创建 ArUco refine 参数失败: {err}"))?;
    let detector =
        objdetect::CharucoDetector::new(board, &charuco_params, &detector_params, refine_params)
            .map_err(|err| format!("创建 CharucoDetector 失败: {err}"))?;
    let mut corners = Mat::default();
    let mut ids = Mat::default();
    let mut marker_corners = Vector::<Vector<Point2f>>::new();
    let mut marker_ids = Mat::default();
    let mut chessboard_fallback_points: Option<Vec<Vector2<f64>>> = None;
    detector
        .detect_board(
            &gray,
            &mut corners,
            &mut ids,
            &mut marker_corners,
            &mut marker_ids,
        )
        .map_err(|err| format!("ChArUco 检测失败: {err}"))?;
    if corners.rows() >= 4 && ids.rows() > 0 {
        refine_corners_subpix(&gray, &mut corners, "ChArUco")?;
    }
    if corners.rows() < 4 || ids.rows() == 0 {
        let dictionary = board
            .get_dictionary()
            .map_err(|err| format!("读取 ChArUco 字典失败: {err}"))?;
        let aruco_detector = objdetect::ArucoDetector::new(
            &dictionary,
            &detector_params,
            objdetect::RefineParameters::new_def()
                .map_err(|err| format!("创建 ArUco refine 参数失败: {err}"))?,
        )
        .map_err(|err| format!("创建 ArUcoDetector 失败: {err}"))?;
        let mut rejected = Vector::<Vector<Point2f>>::new();
        marker_corners = Vector::<Vector<Point2f>>::new();
        marker_ids = Mat::default();
        aruco_detector
            .detect_markers(&gray, &mut marker_corners, &mut marker_ids, &mut rejected)
            .map_err(|err| format!("ArUco marker 检测失败: {err}"))?;
        if marker_ids.rows() > 0 {
            detector
                .detect_board(
                    &gray,
                    &mut corners,
                    &mut ids,
                    &mut marker_corners,
                    &mut marker_ids,
                )
                .map_err(|err| format!("ChArUco marker 插值失败: {err}"))?;
            if corners.rows() >= 4 && ids.rows() > 0 {
                refine_corners_subpix(&gray, &mut corners, "ChArUco")?;
            }
        }
    }
    if corners.rows() < 4 || ids.rows() == 0 {
        let chessboard_size = core::Size::new((squares_x - 1) as i32, (squares_y - 1) as i32);
        let found = calib3d::find_chessboard_corners(
            &gray,
            chessboard_size,
            &mut corners,
            calib3d::CALIB_CB_ADAPTIVE_THRESH + calib3d::CALIB_CB_NORMALIZE_IMAGE,
        )
        .map_err(|err| format!("棋盘格角点检测失败: {err}"))?;
        if found {
            refine_corners_subpix(&gray, &mut corners, "棋盘格")?;
            let mut detected = Vec::with_capacity(corners.rows() as usize);
            for row in 0..corners.rows() {
                let point = *corners
                    .at_2d::<core::Point2f>(row, 0)
                    .map_err(|err| format!("读取棋盘格角点坐标失败: {err}"))?;
                detected.push(Vector2::new(point.x as f64, point.y as f64));
            }
            let marker_info = if marker_ids.rows() > 0 && !marker_corners.is_empty() {
                let centers = marker_corners
                    .iter()
                    .map(|corners| {
                        let mean = corners
                            .iter()
                            .fold(Vector2::zeros(), |acc, point| {
                                acc + Vector2::new(point.x as f64, point.y as f64)
                            })
                            / corners.len() as f64;
                        mean
                    })
                    .collect::<Vec<_>>();
                let ids = (0..marker_ids.rows())
                    .map(|row| {
                        marker_ids
                            .at_2d::<i32>(row, 0)
                            .copied()
                            .map_err(|err| format!("读取 marker id 失败: {err}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Some((centers, ids))
            } else {
                None
            };
            let homography_reordered = marker_info
                .as_ref()
                .and_then(|(centers, ids)| {
                    marker_homography_to_board(board, centers, ids)
                        .ok()
                        .flatten()
                        .and_then(|homography| {
                            board_points_from_charuco(board)
                                .ok()
                                .and_then(|board_points| {
                                    reorder_chessboard_corners_via_homography(
                                        &detected,
                                        &board_points,
                                        board.get_square_length().ok()? as f64,
                                        &homography,
                                    )
                                })
                        })
                });
            chessboard_fallback_points = Some(homography_reordered.unwrap_or_else(|| {
                reorder_chessboard_corners_for_board(
                    detected,
                    squares_x,
                    squares_y,
                    marker_info
                        .as_ref()
                        .map(|(centers, ids)| (centers.as_slice(), ids.as_slice())),
                )
            }));
        }
    }
    if corners.rows() < 4 || ids.rows() == 0 {
        if chessboard_fallback_points.is_none() {
            return Ok(None);
        }
    }
    let used_chessboard_fallback = chessboard_fallback_points.is_some();
    let (corner_ids, image_points) = if let Some(points) = chessboard_fallback_points {
        ((0..points.len()).collect::<Vec<_>>(), points)
    } else {
        let mut corner_ids = Vec::with_capacity(ids.rows() as usize);
        let mut image_points = Vec::with_capacity(ids.rows() as usize);
        for row in 0..ids.rows() {
            let corner_id = *ids
                .at_2d::<i32>(row, 0)
                .map_err(|err| format!("读取角点 id 失败: {err}"))?;
            let point = *corners
                .at_2d::<core::Point2f>(row, 0)
                .map_err(|err| format!("读取角点坐标失败: {err}"))?;
            corner_ids.push(corner_id as usize);
            image_points.push(Vector2::new(point.x as f64, point.y as f64));
        }
        (corner_ids, image_points)
    };
    Ok(Some(DetectionObservation {
        index: 0,
        image_path: image_path.to_string(),
        corner_ids,
        image_points,
        marker_count: marker_ids.rows().max(0) as usize,
        used_chessboard_fallback,
    }))
}

#[tauri::command]
fn read_camera_params(folder: String) -> Result<Option<CameraIntrinsics>, String> {
    let path = PathBuf::from(folder).join("camera_params.yaml");
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path).map_err(|err| format!("读取相机内参失败: {err}"))?;
    parse_camera_params_yaml(&content)
        .map(Some)
        .map_err(|err| format!("解析相机内参失败: {err}"))
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharucoDetection {
    image_path: String,
    output_path: String,
    success: bool,
    num_corners: usize,
    num_markers: usize,
    #[serde(default)]
    used_chessboard_fallback: bool,
    message: String,
    #[serde(default)]
    corner_rows: Vec<CharucoCornerRow>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharucoCornerRow {
    id: usize,
    image_point: [f64; 2],
    #[serde(default)]
    camera_point: Option<[f64; 3]>,
}

#[tauri::command]
fn list_rgb_images(folder: String) -> Result<Vec<ImageFile>, String> {
    list_images_by_kind(folder, is_rgb_image)
}

#[tauri::command]
fn list_depth_images(folder: String) -> Result<Vec<ImageFile>, String> {
    list_depth_images_with_raw_fallback(folder)
}

#[tauri::command]
fn create_depth_preview(depth_path: String) -> Result<ImageFile, String> {
    let source_path = PathBuf::from(&depth_path);
    let (values, width, height) = load_depth_pixels(&source_path)?;
    let pixels = colorize_depth_values(&values, width, height)?;
    let preview_path = depth_preview_path(&source_path)?;

    if let Some(parent) = preview_path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("创建深度预览目录失败: {err}"))?;
    }
    image::save_buffer(
        &preview_path,
        &pixels,
        width,
        height,
        image::ColorType::Rgb8,
    )
    .map_err(|err| format!("保存深度预览失败: {err}"))?;

    let name = preview_path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| "深度预览文件名不是有效 UTF-8".to_string())?
        .to_string();
    Ok(ImageFile {
        name,
        path: preview_path.to_string_lossy().to_string(),
    })
}

fn list_depth_images_with_raw_fallback(folder: String) -> Result<Vec<ImageFile>, String> {
    let entries = fs::read_dir(&folder).map_err(|err| format!("读取图像文件夹失败: {err}"))?;
    let mut preferred = Vec::new();
    let mut raw_by_prefix = std::collections::BTreeMap::new();
    let mut preferred_prefixes = std::collections::BTreeSet::new();

    for entry in entries {
        let entry = entry.map_err(|err| format!("读取图像文件失败: {err}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or_else(|| "图像文件名不是有效 UTF-8".to_string())?
            .to_string();
        if is_depth_image(&path) {
            if let Some(prefix) = image_frame_prefix(&name) {
                preferred_prefixes.insert(prefix.to_string());
            }
            preferred.push(ImageFile {
                name: name.clone(),
                path: path.to_string_lossy().to_string(),
            });
        } else if is_raw_depth_candidate(&path) {
            let Some(prefix) = image_frame_prefix(&name) else {
                continue;
            };
            raw_by_prefix
                .entry(prefix.to_string())
                .or_insert_with(|| ImageFile {
                    name: name.clone(),
                    path: path.to_string_lossy().to_string(),
                });
        }
    }

    for (prefix, entry) in raw_by_prefix {
        if !preferred_prefixes.contains(&prefix) {
            preferred.push(entry);
        }
    }

    preferred.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(preferred)
}

fn load_depth_pixels(path: &Path) -> Result<(Vec<u16>, u32, u32), String> {
    if path
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("raw"))
    {
        load_raw_depth_pixels(path)
    } else {
        let depth_image = image::open(path).map_err(|err| format!("读取深度图失败: {err}"))?;
        let gray = depth_image.to_luma16();
        let (width, height) = gray.dimensions();
        let values = gray.pixels().map(|pixel| pixel.0[0]).collect();
        Ok((values, width, height))
    }
}

fn load_raw_depth_pixels(path: &Path) -> Result<(Vec<u16>, u32, u32), String> {
    let (width, height) = matching_rgb_dimensions(path)?;
    let bytes = fs::read(path).map_err(|err| format!("读取 raw 深度图失败: {err}"))?;
    if bytes.len() % 2 != 0 {
        return Err("raw 深度图字节数不是 16-bit 对齐".to_string());
    }
    let values: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    let expected_len = width as usize * height as usize;
    if values.len() != expected_len {
        return Err(format!(
            "raw 深度图尺寸与像素数量不匹配: 期望 {expected_len}, 实际 {}",
            values.len()
        ));
    }
    Ok((values, width, height))
}

fn matching_rgb_dimensions(depth_path: &Path) -> Result<(u32, u32), String> {
    let directory = depth_path
        .parent()
        .ok_or_else(|| "深度图路径缺少父目录".to_string())?;
    let depth_name = depth_path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| "深度图文件名不是有效 UTF-8".to_string())?;
    let prefix = image_frame_prefix(depth_name)
        .ok_or_else(|| "raw 深度图文件名中没有可匹配的编号前缀".to_string())?;

    let entries = fs::read_dir(directory).map_err(|err| format!("读取图像文件夹失败: {err}"))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("读取图像文件失败: {err}"))?;
        let path = entry.path();
        if !path.is_file() || !is_rgb_image(&path) {
            continue;
        }
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if image_frame_prefix(name).as_deref() != Some(prefix) {
            continue;
        }
        return image::image_dimensions(&path)
            .map_err(|err| format!("读取匹配 RGB 图像尺寸失败: {err}"));
    }

    Err(format!("未找到与 raw 深度图编号 {prefix} 对应的 RGB 图像"))
}

fn depth_preview_path(depth_path: &Path) -> Result<PathBuf, String> {
    let stem = depth_path
        .file_stem()
        .and_then(OsStr::to_str)
        .ok_or_else(|| "深度图文件名不是有效 UTF-8".to_string())?;
    Ok(std::env::temp_dir()
        .join("handeye-manager-depth-preview")
        .join(format!("{stem}_preview.png")))
}

fn colorize_depth_values(values: &[u16], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let expected_len = width as usize * height as usize;
    if values.len() != expected_len {
        return Err(format!(
            "深度图尺寸与像素数量不匹配: 期望 {expected_len}, 实际 {}",
            values.len()
        ));
    }

    let mut min_depth = u16::MAX;
    let mut max_depth = 0_u16;
    for &value in values.iter().filter(|&&value| value > 0) {
        min_depth = min_depth.min(value);
        max_depth = max_depth.max(value);
    }

    let mut pixels = Vec::with_capacity(expected_len * 3);
    for &value in values {
        let color = if value == 0 || max_depth == 0 {
            [0, 0, 0]
        } else {
            let normalized = if max_depth == min_depth {
                1.0
            } else {
                (value.saturating_sub(min_depth)) as f32 / (max_depth - min_depth) as f32
            };
            jet_color(normalized)
        };
        pixels.extend_from_slice(&color);
    }
    Ok(pixels)
}

fn jet_color(t: f32) -> [u8; 3] {
    let clamped = t.clamp(0.0, 1.0);
    let r = (1.5 - (4.0 * clamped - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * clamped - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * clamped - 1.0).abs()).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

#[tauri::command]
fn read_pose_file(file: String) -> Result<Vec<PoseFileRow>, String> {
    let content = fs::read_to_string(&file).map_err(|err| format!("读取位姿文件失败: {err}"))?;
    Ok(content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .enumerate()
        .map(|(index, content)| PoseFileRow {
            index: index + 1,
            content,
        })
        .collect())
}

#[tauri::command]
fn save_text_file(path: String, content: String) -> Result<(), String> {
    let target = PathBuf::from(&path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("创建结果目录失败: {err}"))?;
    }
    fs::write(&target, content).map_err(|err| format!("保存结果文件失败: {err}"))
}

fn list_images_by_kind(
    folder: String,
    predicate: fn(&Path) -> bool,
) -> Result<Vec<ImageFile>, String> {
    let mut files = Vec::new();
    let entries = fs::read_dir(&folder).map_err(|err| format!("读取图像文件夹失败: {err}"))?;

    for entry in entries {
        let entry = entry.map_err(|err| format!("读取图像文件失败: {err}"))?;
        let path = entry.path();
        if !path.is_file() || !predicate(&path) {
            continue;
        }
        let name = path
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or_else(|| "图像文件名不是有效 UTF-8".to_string())?
            .to_string();
        files.push(ImageFile {
            name,
            path: path.to_string_lossy().to_string(),
        });
    }

    files.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(files)
}

#[tauri::command]
fn run_handeye_calibration(request: CalibrationRequest) -> Result<CalibrationRun, String> {
    let image_dir = PathBuf::from(&request.image_dir);
    let output_path = image_dir.join("calibration_result.yaml");
    let payload = CalibrationHttpRequest {
        image_dir: request.image_dir,
        poses_file: request.poses_file,
        marker: request.marker.unwrap_or_else(|| "charuco".to_string()),
        camera_params: request.camera_params,
        camera_intrinsics: request.camera_intrinsics,
        setup: request.setup,
        pose_format: request.pose_format.unwrap_or_else(|| "sxyz".to_string()),
        use_depth: request.use_depth,
        squares_x: request.squares_x.unwrap_or(14),
        squares_y: request.squares_y.unwrap_or(9),
        square_length: request.square_length.unwrap_or(0.020),
        marker_length: request.marker_length.unwrap_or(0.015),
        aruco_dict: request
            .aruco_dict
            .unwrap_or_else(|| "DICT_5X5_50".to_string()),
        filter_inconsistent: request.filter_inconsistent,
        excluded_image_indices: request.excluded_image_indices.unwrap_or_default(),
        output_path: output_path.to_string_lossy().to_string(),
    };
    run_handeye_calibration_direct(&payload)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CalibrationHttpRequest {
    image_dir: String,
    poses_file: String,
    marker: String,
    camera_params: Option<String>,
    camera_intrinsics: Option<CameraIntrinsics>,
    setup: String,
    pose_format: String,
    use_depth: String,
    squares_x: usize,
    squares_y: usize,
    square_length: f64,
    marker_length: f64,
    aruco_dict: String,
    filter_inconsistent: Option<bool>,
    excluded_image_indices: Vec<usize>,
    output_path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ConversionPreviewHttpRequest {
    image_dir: String,
    poses_file: String,
    setup: String,
    pose_format: String,
    primary_transform_name: String,
    primary_matrix: String,
    secondary_transform_name: String,
    secondary_matrix: String,
    camera_intrinsics: Option<CameraIntrinsics>,
    squares_x: usize,
    squares_y: usize,
    square_length: f64,
    marker_length: f64,
    aruco_dict: String,
}

fn run_handeye_calibration_direct(
    payload: &CalibrationHttpRequest,
) -> Result<CalibrationRun, String> {
    let depth_mode = parse_depth_mode(&payload.use_depth)?;
    let intrinsics = resolve_camera_intrinsics(
        payload.camera_intrinsics.clone(),
        payload.camera_params.clone(),
    )?;
    let image_entries = list_rgb_images(payload.image_dir.clone())?;
    let poses = load_pose_transforms(&payload.poses_file, &payload.pose_format, false)?;
    let board = build_charuco_board(
        payload.squares_x,
        payload.squares_y,
        payload.square_length,
        payload.marker_length,
        &payload.aruco_dict,
    )?;
    let board_points = board_points_from_charuco(&board)?;

    let excluded = payload
        .excluded_image_indices
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let frame_count = image_entries.len().min(poses.len());
    let mut detections = Vec::new();
    let mut frame_errors = Vec::new();
    let mut filtered_images = Vec::new();

    for index in 0..frame_count {
        let image_path = &image_entries[index].path;
        if excluded.contains(&index) {
            filtered_images.push(index);
            frame_errors.push(FrameError {
                index,
                image_path: image_path.clone(),
                used: false,
                corner_count: None,
                reprojection_mean_px: None,
                reprojection_rms_px: None,
                reprojection_max_px: None,
                reference_reprojection_mean_px: None,
                reference_reprojection_rms_px: None,
                reference_reprojection_max_px: None,
                reprojection_error_px: None,
                optimized_reprojection_error_px: None,
                base_consistency_mean_mm: None,
                base_consistency_rms_mm: None,
                base_consistency_max_mm: None,
                base_consistency_count: None,
                translation_error_mm: None,
                rotation_error_deg: None,
                used_chessboard_fallback: false,
            });
            continue;
        }

        match detect_charuco_observation(
            image_path,
            &board,
            &intrinsics,
            payload.squares_x,
            payload.squares_y,
        )? {
            Some(mut detection) if detection.corner_ids.len() >= 4 => {
                detection.index = index;
                detections.push(SyntheticDetection {
                    index,
                    image_path: image_path.clone(),
                    corner_ids: detection.corner_ids.clone(),
                    image_points: detection.image_points.clone(),
                    marker_count: detection.marker_count,
                    used_chessboard_fallback: detection.used_chessboard_fallback,
                });
                frame_errors.push(FrameError {
                    index,
                    image_path: image_path.clone(),
                    used: true,
                    corner_count: Some(detection.corner_ids.len()),
                    reprojection_mean_px: None,
                    reprojection_rms_px: None,
                    reprojection_max_px: None,
                    reference_reprojection_mean_px: None,
                    reference_reprojection_rms_px: None,
                    reference_reprojection_max_px: None,
                    reprojection_error_px: None,
                    optimized_reprojection_error_px: None,
                    base_consistency_mean_mm: None,
                    base_consistency_rms_mm: None,
                    base_consistency_max_mm: None,
                    base_consistency_count: None,
                    translation_error_mm: None,
                    rotation_error_deg: None,
                    used_chessboard_fallback: detection.used_chessboard_fallback,
                });
            }
            _ => {
                frame_errors.push(FrameError {
                    index,
                    image_path: image_path.clone(),
                    used: false,
                    corner_count: None,
                    reprojection_mean_px: None,
                    reprojection_rms_px: None,
                    reprojection_max_px: None,
                    reference_reprojection_mean_px: None,
                    reference_reprojection_rms_px: None,
                    reference_reprojection_max_px: None,
                    reprojection_error_px: None,
                    optimized_reprojection_error_px: None,
                    base_consistency_mean_mm: None,
                    base_consistency_rms_mm: None,
                    base_consistency_max_mm: None,
                    base_consistency_count: None,
                    translation_error_mm: None,
                    rotation_error_deg: None,
                    used_chessboard_fallback: false,
                });
            }
        }
    }

    if detections.len() < 3 {
        return Err("有效 ChArUco 检测少于 3 帧，无法进行手眼标定".to_string());
    }
    let active_pose_indices = detections.iter().map(|d| d.index).collect::<Vec<_>>();
    let mut active_poses = active_pose_indices
        .iter()
        .map(|index| poses[*index])
        .collect::<Vec<_>>();
    let mut observations = detections
        .iter()
        .map(ObservationLike::to_observation)
        .collect::<Vec<_>>();
    let (mut measured_object_to_camera, mut depth_observations, depth_used) = resolve_object_to_camera_measurements(
        &depth_mode,
        &intrinsics,
        &board_points,
        &observations,
    )?;
    let total_board_corners = board_points.len();
    let quality_keep = observations
        .iter()
        .zip(measured_object_to_camera.iter())
        .enumerate()
        .filter_map(|(row, (observation, measured))| {
            let reprojection = compute_reprojection_metrics(
                &intrinsics,
                &board_points,
                std::slice::from_ref(observation),
                std::slice::from_ref(measured),
            );
            detection_passes_quality_gate(observation, total_board_corners, reprojection.mean_px)
                .then_some(row)
        })
        .collect::<Vec<_>>();
    if quality_keep.len() >= 3 && quality_keep.len() < detections.len() {
        let keep_set = quality_keep
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let rejected = (0..detections.len())
            .filter(|local_index| !keep_set.contains(local_index))
            .map(|local_index| active_pose_indices[local_index])
            .collect::<Vec<_>>();
        filtered_images.extend(rejected.iter().copied());
        for index in rejected {
            mark_frame_filtered(&mut frame_errors, index);
        }
        detections = quality_keep
            .iter()
            .map(|index| detections[*index].clone())
            .collect();
        active_poses = quality_keep.iter().map(|index| active_poses[*index]).collect();
        observations = quality_keep
            .iter()
            .map(|index| observations[*index].clone())
            .collect();
        measured_object_to_camera = quality_keep
            .iter()
            .map(|index| measured_object_to_camera[*index])
            .collect();
        depth_observations = quality_keep
            .iter()
            .map(|index| depth_observations[*index].clone())
            .collect();
        filtered_images.sort_unstable();
        filtered_images.dedup();
    }

    if payload.filter_inconsistent.unwrap_or(true) {
        let keep = filter_inconsistent_measurements(
            &payload.setup,
            &active_poses,
            &measured_object_to_camera,
        )?;
        if keep.len() >= 3 && keep.len() < detections.len() {
            let keep_set = keep
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            let rejected = (0..detections.len())
                .filter(|local_index| !keep_set.contains(local_index))
                .map(|local_index| active_pose_indices[local_index])
                .collect::<Vec<_>>();
            filtered_images.extend(rejected.iter().copied());
            for index in rejected {
                mark_frame_filtered(&mut frame_errors, index);
            }
            detections = keep
                .iter()
                .map(|index| detections[*index].clone())
                .collect();
            active_poses = keep.iter().map(|index| active_poses[*index]).collect();
            observations = keep
                .iter()
                .map(|index| observations[*index].clone())
                .collect();
            measured_object_to_camera = keep
                .iter()
                .map(|index| measured_object_to_camera[*index])
                .collect();
            depth_observations = keep
                .iter()
                .map(|index| depth_observations[*index].clone())
                .collect();
            filtered_images.sort_unstable();
            filtered_images.dedup();
        }
    }

    if detections.len() < 3 {
        return Err("一致性过滤后有效 ChArUco 检测少于 3 帧，无法进行手眼标定".to_string());
    }
    let solution = optimize_handeye_from_measurements(
        &payload.setup,
        &intrinsics,
        &board_points,
        &observations,
        &active_poses,
        &measured_object_to_camera,
        &depth_observations,
    )?;
    let pose_errors = compute_pose_errors(
        &payload.setup,
        &active_poses,
        &solution.measured_object_to_camera,
        &solution.primary_transform,
        &solution.secondary_transform,
    );
    let base_consistency = compute_reference_consistency(
        &payload.setup,
        &board_points,
        &observations,
        &active_poses,
        &solution.primary_transform,
        &solution.measured_object_to_camera,
        &depth_observations,
    );
    let derived_reprojection = compute_reprojection_metrics(
        &intrinsics,
        &board_points,
        &observations,
        &solution.object_to_camera_derived,
    );
    let reference_reprojection = compute_reprojection_metrics(
        &intrinsics,
        &board_points,
        &observations,
        &solution.measured_object_to_camera,
    );
    let (primary_name, secondary_name) = if payload.setup == "eye-in-hand" {
        ("T_C2F".to_string(), "T_O2W".to_string())
    } else {
        ("T_C2W".to_string(), "T_O2F".to_string())
    };
    for row in 0..observations.len() {
        let (derived_mean, derived_rms, derived_max) = derived_reprojection.per_frame[row];
        let (reference_mean, reference_rms, reference_max) = reference_reprojection.per_frame[row];
        if let Some(frame) = frame_errors
            .iter_mut()
            .find(|frame| frame.index == detections[row].index)
        {
            frame.reprojection_mean_px = Some(derived_mean);
            frame.reprojection_rms_px = Some(derived_rms);
            frame.reprojection_max_px = Some(derived_max);
            frame.reference_reprojection_mean_px = Some(reference_mean);
            frame.reference_reprojection_rms_px = Some(reference_rms);
            frame.reference_reprojection_max_px = Some(reference_max);
            frame.reprojection_error_px = Some(derived_mean);
            frame.optimized_reprojection_error_px = Some(reference_mean);
            frame.translation_error_mm = pose_errors
                .per_frame
                .get(row)
                .map(|(translation, _)| *translation);
            frame.rotation_error_deg = pose_errors
                .per_frame
                .get(row)
                .map(|(_, rotation)| *rotation);
            if let Some(consistency) = &base_consistency {
                if let Some(Some((count, mean, rms, max))) = consistency.per_frame.get(row) {
                    frame.base_consistency_count = Some(*count);
                    frame.base_consistency_mean_mm = Some(mean * 1000.0);
                    frame.base_consistency_rms_mm = Some(rms * 1000.0);
                    frame.base_consistency_max_mm = Some(max * 1000.0);
                }
            }
        }
    }

    let local_index_by_frame = detections
        .iter()
        .enumerate()
        .map(|(local_index, detection)| (detection.index, local_index))
        .collect::<HashMap<_, _>>();
    let preview_frames = (0..frame_count)
        .map(|index| {
            let local_index = local_index_by_frame.get(&index).copied();
            let (camera_in_base, board_in_base, board_in_focus) =
                preview_frame_transforms_with_measurement(
                    &payload.setup,
                    &poses[index],
                    &solution.primary_transform,
                    &solution.secondary_transform,
                    local_index.map(|row| &solution.measured_object_to_camera[row]),
                );
            PreviewFrame {
                index,
                image_path: image_entries[index].path.clone(),
                used: local_index.is_some(),
                camera_in_base: matrix_to_rows(&camera_in_base),
                board_in_base: matrix_to_rows(&board_in_base),
                board_in_focus: matrix_to_rows(&board_in_focus),
            }
        })
        .collect::<Vec<_>>();

    let run = CalibrationRun {
        output_path: payload.output_path.clone(),
        stdout: String::new(),
        stderr: String::new(),
        setup: payload.setup.clone(),
        primary_transform_name: primary_name,
        primary_matrix_rows: format_matrix_rows(&solution.primary_transform),
        secondary_transform_name: secondary_name,
        secondary_matrix_rows: format_matrix_rows(&solution.secondary_transform),
        matrix_rows: format_matrix_rows(&solution.primary_transform),
        average_error_mm: pose_errors.translation_mean_mm,
        rotation_error_deg: pose_errors.rotation_mean_deg,
        reprojection_error_px: solution.reprojection_mean_px,
        reprojection_rms_px: Some(solution.reprojection_rms_px),
        base_consistency_mean_mm: base_consistency.as_ref().map(|stats| stats.mean_m * 1000.0),
        base_consistency_rms_mm: base_consistency.as_ref().map(|stats| stats.rms_m * 1000.0),
        base_consistency_max_mm: base_consistency.as_ref().map(|stats| stats.max_m * 1000.0),
        base_consistency_count: base_consistency.as_ref().map(|stats| stats.count),
        num_images: frame_count,
        num_images_used: detections.len(),
        filtered_images,
        frame_errors,
        preview_frames,
        depth_used,
        message: format!(
            "{} 标定完成；有效数据 {}/{}；平均平移误差 {:.3} mm",
            if payload.setup == "eye-in-hand" {
                "T_C2F"
            } else {
                "T_C2W"
            },
            detections.len(),
            frame_count,
            pose_errors.translation_mean_mm
        ),
    };
    save_calibration_run_yaml(&run)?;
    Ok(run)
}

fn mark_frame_filtered(frame_errors: &mut [FrameError], index: usize) {
    if let Some(frame) = frame_errors.iter_mut().find(|frame| frame.index == index) {
        frame.used = false;
        frame.corner_count = None;
        frame.reprojection_mean_px = None;
        frame.reprojection_rms_px = None;
        frame.reprojection_max_px = None;
        frame.reference_reprojection_mean_px = None;
        frame.reference_reprojection_rms_px = None;
        frame.reference_reprojection_max_px = None;
        frame.reprojection_error_px = None;
        frame.optimized_reprojection_error_px = None;
        frame.base_consistency_mean_mm = None;
        frame.base_consistency_rms_mm = None;
        frame.base_consistency_max_mm = None;
        frame.base_consistency_count = None;
        frame.translation_error_mm = None;
        frame.rotation_error_deg = None;
    }
}

#[tauri::command]
fn build_conversion_preview(
    request: ConversionPreviewRequest,
) -> Result<ConversionPreviewResult, String> {
    let payload = ConversionPreviewHttpRequest {
        image_dir: request.image_dir,
        poses_file: request.poses_file,
        setup: request.setup,
        pose_format: request.pose_format,
        primary_transform_name: request.primary_transform_name,
        primary_matrix: flatten_matrix_rows(&request.primary_matrix_rows),
        secondary_transform_name: request.secondary_transform_name,
        secondary_matrix: flatten_matrix_rows(&request.secondary_matrix_rows),
        camera_intrinsics: request.camera_intrinsics,
        squares_x: request.squares_x.unwrap_or(14),
        squares_y: request.squares_y.unwrap_or(9),
        square_length: request.square_length.unwrap_or(0.020),
        marker_length: request.marker_length.unwrap_or(0.015),
        aruco_dict: request
            .aruco_dict
            .unwrap_or_else(|| "DICT_5X5_50".to_string()),
    };
    build_conversion_preview_direct(&payload)
}

fn build_conversion_preview_direct(
    payload: &ConversionPreviewHttpRequest,
) -> Result<ConversionPreviewResult, String> {
    let image_paths = list_rgb_images(payload.image_dir.clone())?;
    let poses = load_pose_transforms(&payload.poses_file, &payload.pose_format, false)?;
    let primary = parse_flattened_matrix(&payload.primary_matrix)?;
    let secondary = parse_flattened_matrix(&payload.secondary_matrix)?;
    let frame_count = image_paths.len().min(poses.len());

    let board_and_points = payload.camera_intrinsics.as_ref().and_then(|_| {
        build_charuco_board(
            payload.squares_x,
            payload.squares_y,
            payload.square_length,
            payload.marker_length,
            &payload.aruco_dict,
        )
        .ok()
        .and_then(|board| {
            board_points_from_charuco(&board)
                .ok()
                .map(|points| (board, points))
        })
    });

    let mut preview_frames = Vec::with_capacity(frame_count);

    for (index, (image, pose)) in image_paths
        .into_iter()
        .zip(poses.into_iter())
        .take(frame_count)
        .enumerate()
    {
        let (camera_in_base, board_in_base, board_in_focus, used) =
            if let Some((ref board, ref board_points)) = board_and_points {
                let intrinsics = payload.camera_intrinsics.as_ref().expect("camera_intrinsics checked above");
                match detect_charuco_observation(
                    &image.path,
                    board,
                    intrinsics,
                    payload.squares_x,
                    payload.squares_y,
                ) {
                    Ok(Some(obs)) => match solve_pnp_measurement(intrinsics, board_points, &obs) {
                        Ok(t_o2c) => {
                            let (cam, base, focus) = preview_frame_transforms_with_measurement(
                                &payload.setup,
                                &pose,
                                &primary,
                                &secondary,
                                Some(&t_o2c),
                            );
                            (cam, base, focus, true)
                        }
                        Err(_) => {
                            let (cam, base, focus) =
                                preview_frame_transforms(&payload.setup, &pose, &primary, &secondary);
                            (cam, base, focus, false)
                        }
                    },
                    _ => {
                        let (cam, base, focus) =
                            preview_frame_transforms(&payload.setup, &pose, &primary, &secondary);
                        (cam, base, focus, false)
                    }
                }
            } else {
                let (cam, base, focus) =
                    preview_frame_transforms(&payload.setup, &pose, &primary, &secondary);
                (cam, base, focus, true)
            };

        preview_frames.push(PreviewFrame {
            index,
            image_path: image.path,
            used,
            camera_in_base: matrix_to_rows(&camera_in_base),
            board_in_base: matrix_to_rows(&board_in_base),
            board_in_focus: matrix_to_rows(&board_in_focus),
        });
    }

    Ok(ConversionPreviewResult { preview_frames })
}

fn flatten_matrix_rows(matrix_rows: &[String]) -> String {
    matrix_rows
        .iter()
        .flat_map(|row| row.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_flattened_matrix(matrix: &str) -> Result<Matrix4<f64>, String> {
    let values = parse_yaml_number_list(matrix)?;
    if values.len() != 16 {
        return Err(format!(
            "变换矩阵必须包含 16 个数值，实际为 {}",
            values.len()
        ));
    }
    Ok(Matrix4::from_row_slice(&values))
}

fn load_pose_transforms(
    filepath: &str,
    pose_format: &str,
    invert: bool,
) -> Result<Vec<Matrix4<f64>>, String> {
    let rows = fs::read_to_string(filepath).map_err(|err| format!("读取位姿文件失败: {err}"))?;
    let order = scipy_euler_order(pose_format);
    let mut poses = Vec::new();
    for line in rows.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let values = parse_yaml_number_list(trimmed)?;
        if values.len() < 6 {
            return Err(format!("位姿行至少需要 6 个数值: {trimmed}"));
        }
        let pose = pose_from_row(&values[..6], &order)?;
        poses.push(if invert {
            pose.try_inverse()
                .ok_or_else(|| "位姿矩阵不可逆".to_string())?
        } else {
            pose
        });
    }
    Ok(poses)
}

fn scipy_euler_order(pose_format: &str) -> String {
    let pose_format = if pose_format.is_empty() {
        "sxyz"
    } else {
        pose_format
    };
    let chars: Vec<char> = pose_format.chars().collect();
    if chars.len() == 4 && matches!(chars[0], 's' | 'r') {
        let axes: String = chars[1..].iter().collect();
        if chars[0] == 's' {
            axes.to_lowercase()
        } else {
            axes.to_uppercase()
        }
    } else {
        pose_format.to_string()
    }
}

fn pose_from_row(values: &[f64], order: &str) -> Result<Matrix4<f64>, String> {
    let translation = Vector3::new(values[0] * 0.001, values[1] * 0.001, values[2] * 0.001);
    let rotation = euler_rotation_matrix(order, values[3], values[4], values[5], true)?;
    let mut pose = Matrix4::identity();
    pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    pose[(0, 3)] = translation.x;
    pose[(1, 3)] = translation.y;
    pose[(2, 3)] = translation.z;
    Ok(pose)
}

fn euler_rotation_matrix(
    order: &str,
    a: f64,
    b: f64,
    c: f64,
    degrees: bool,
) -> Result<Matrix3<f64>, String> {
    let (a, b, c) = if degrees {
        (a.to_radians(), b.to_radians(), c.to_radians())
    } else {
        (a, b, c)
    };
    let rotations = [a, b, c];
    let mut result = Matrix3::identity();
    let intrinsic = order
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase());
    let axes: Vec<char> = order.chars().map(|ch| ch.to_ascii_lowercase()).collect();
    if axes.len() != 3 {
        return Err(format!("不支持的欧拉角顺序: {order}"));
    }
    if intrinsic {
        for (axis, angle) in axes.iter().zip(rotations) {
            result *= axis_rotation(*axis, angle)?;
        }
    } else {
        for (axis, angle) in axes.iter().zip(rotations) {
            result = axis_rotation(*axis, angle)? * result;
        }
    }
    Ok(result)
}

fn axis_rotation(axis: char, angle: f64) -> Result<Matrix3<f64>, String> {
    let rotation = match axis {
        'x' => Rotation3::from_axis_angle(&Vector3::x_axis(), angle),
        'y' => Rotation3::from_axis_angle(&Vector3::y_axis(), angle),
        'z' => Rotation3::from_axis_angle(&Vector3::z_axis(), angle),
        _ => return Err(format!("不支持的旋转轴: {axis}")),
    };
    Ok(rotation.into_inner())
}

fn preview_frame_transforms(
    setup: &str,
    pose: &Matrix4<f64>,
    primary: &Matrix4<f64>,
    secondary: &Matrix4<f64>,
) -> (Matrix4<f64>, Matrix4<f64>, Matrix4<f64>) {
    preview_frame_transforms_with_measurement(setup, pose, primary, secondary, None)
}

fn preview_frame_transforms_with_measurement(
    setup: &str,
    pose: &Matrix4<f64>,
    primary: &Matrix4<f64>,
    secondary: &Matrix4<f64>,
    object_to_camera: Option<&Matrix4<f64>>,
) -> (Matrix4<f64>, Matrix4<f64>, Matrix4<f64>) {
    if setup == "eye-in-hand" {
        let camera_in_base = pose * primary;
        let board_in_base = object_to_camera
            .map(|measured| camera_in_base * measured)
            .unwrap_or(*secondary);
        (camera_in_base, board_in_base, board_in_base)
    } else {
        let camera_in_base = *primary;
        let board_in_base = object_to_camera
            .map(|measured| primary * measured)
            .unwrap_or_else(|| pose * secondary);
        let board_in_focus = object_to_camera
            .and_then(|measured| {
                pose.try_inverse()
                    .map(|pose_inv| pose_inv * primary * measured)
            })
            .unwrap_or(*secondary);
        (camera_in_base, board_in_base, board_in_focus)
    }
}

fn matrix_to_rows(matrix: &Matrix4<f64>) -> Vec<Vec<f64>> {
    (0..4)
        .map(|row| (0..4).map(|col| matrix[(row, col)]).collect())
        .collect()
}

fn format_matrix_rows(matrix: &Matrix4<f64>) -> Vec<String> {
    (0..4)
        .map(|row| {
            (0..4)
                .map(|col| format!("{:.7}", matrix[(row, col)]))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect()
}

fn resolve_camera_intrinsics(
    intrinsics: Option<CameraIntrinsics>,
    camera_params: Option<String>,
) -> Result<CameraIntrinsics, String> {
    if let Some(intrinsics) = intrinsics {
        return Ok(intrinsics);
    }
    if let Some(path) = camera_params {
        let content =
            fs::read_to_string(&path).map_err(|err| format!("读取相机内参文件失败: {err}"))?;
        return parse_camera_params_yaml(&content);
    }
    Err("缺少相机内参".to_string())
}

fn save_calibration_run_yaml(run: &CalibrationRun) -> Result<(), String> {
    let mut root = Mapping::new();
    root.insert(Value::from("setup"), Value::from(run.setup.clone()));
    root.insert(
        Value::from("transforms"),
        Value::Mapping({
            let mut transforms = Mapping::new();
            if !run.primary_transform_name.is_empty() && !run.primary_matrix_rows.is_empty() {
                transforms.insert(
                    Value::from(run.primary_transform_name.clone()),
                    matrix_block_value(&run.primary_matrix_rows)?,
                );
            }
            if !run.secondary_transform_name.is_empty() && !run.secondary_matrix_rows.is_empty() {
                transforms.insert(
                    Value::from(run.secondary_transform_name.clone()),
                    matrix_block_value(&run.secondary_matrix_rows)?,
                );
            }
            transforms
        }),
    );
    root.insert(
        Value::from("metrics"),
        Value::Mapping({
            let mut metrics = Mapping::new();
            metrics.insert(
                Value::from("reprojection_error"),
                Value::from(run.reprojection_error_px),
            );
            metrics.insert(
                Value::from("reprojection_rms_px"),
                option_value(run.reprojection_rms_px),
            );
            metrics.insert(
                Value::from("translation_error_mm"),
                Value::from(run.average_error_mm),
            );
            metrics.insert(
                Value::from("rotation_error_deg"),
                Value::from(run.rotation_error_deg),
            );
            metrics.insert(
                Value::from("base_consistency_rms_mm"),
                option_value(run.base_consistency_rms_mm),
            );
            metrics
        }),
    );
    root.insert(
        Value::from("num_images"),
        Value::from(run.num_images as i64),
    );
    root.insert(
        Value::from("num_images_used"),
        Value::from(run.num_images_used as i64),
    );
    root.insert(
        Value::from("filtered_images"),
        Value::Sequence(
            run.filtered_images
                .iter()
                .map(|index| Value::from(*index as i64))
                .collect(),
        ),
    );
    root.insert(Value::from("depth_used"), Value::from(run.depth_used));

    let yaml = serde_yaml::to_string(&Value::Mapping(root))
        .map_err(|err| format!("序列化标定 YAML 失败: {err}"))?;
    fs::write(&run.output_path, yaml).map_err(|err| format!("写入标定 YAML 失败: {err}"))
}

fn matrix_block_value(rows: &[String]) -> Result<Value, String> {
    let flattened = flatten_matrix_rows(rows);
    let values = parse_yaml_number_list(&flattened)?;
    if values.len() != 16 {
        return Err(format!(
            "变换矩阵必须包含 16 个数值，实际为 {}",
            values.len()
        ));
    }
    let mut mapping = Mapping::new();
    mapping.insert(Value::from("rows"), Value::from(4));
    mapping.insert(Value::from("cols"), Value::from(4));
    mapping.insert(
        Value::from("data"),
        Value::Sequence(values.into_iter().map(Value::from).collect()),
    );
    Ok(Value::Mapping(mapping))
}

fn option_value(value: Option<f64>) -> Value {
    value.map(Value::from).unwrap_or(Value::Null)
}

fn charuco_detect_and_draw(
    image_path: &str,
    intrinsics: &CameraIntrinsics,
    squares_x: usize,
    squares_y: usize,
    square_length: f64,
    marker_length: f64,
    aruco_dict: &str,
    output_dir: &Path,
    depth_path: Option<&str>,
) -> Result<CharucoDetection, String> {
    let board =
        build_charuco_board(squares_x, squares_y, square_length, marker_length, aruco_dict)?;
    let board_points = board_points_from_charuco(&board)?;

    let obs = detect_charuco_observation(image_path, &board, intrinsics, squares_x, squares_y)?;

    let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)
        .map_err(|err| format!("读取图像失败: {err}"))?;
    if image.empty() {
        return Err(format!("无法读取图像: {image_path}"));
    }

    let camera_mat = camera_matrix_mat(intrinsics)?;
    let dist = distortion_mat(intrinsics)?;
    let mut overlay = Mat::default();
    calib3d::undistort(&image, &mut overlay, &camera_mat, &dist, &camera_mat)
        .map_err(|err| format!("图像去畸变失败: {err}"))?;

    let overlay_rows = overlay.rows();

    let stem = Path::new(image_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
        .replace("_Color", "");
    let output_path = output_dir.join(format!("detection_{stem}.png"));

    let (success, num_corners, num_markers, used_chessboard_fallback, message, corner_rows) =
        if let Some(detection) = obs {
            let n_corners = detection.corner_ids.len();
            let mut rows = Vec::with_capacity(n_corners);

            for (&corner_id, point) in
                detection.corner_ids.iter().zip(detection.image_points.iter())
            {
                let pt = core::Point::new(point.x.round() as i32, point.y.round() as i32);
                let _ = imgproc::circle_def(
                    &mut overlay,
                    pt,
                    4,
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                );
                let _ = imgproc::put_text(
                    &mut overlay,
                    &corner_id.to_string(),
                    core::Point::new(point.x.round() as i32 + 5, point.y.round() as i32 - 5),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.35,
                    core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                    1,
                    8,
                    false,
                );
                rows.push(CharucoCornerRow {
                    id: corner_id,
                    image_point: [point.x, point.y],
                    camera_point: None,
                });
            }

            if n_corners >= 4 {
                let t_o2c = if let Some(depth) = depth_path {
                    solve_from_depth(intrinsics, &board_points, &detection, depth).map(|(t, depth_obs)| {
                        let by_corner = depth_obs
                            .corner_ids
                            .iter()
                            .zip(depth_obs.camera_points.iter())
                            .map(|(corner_id, point)| (*corner_id, [point.x, point.y, point.z]))
                            .collect::<HashMap<_, _>>();
                        for row in rows.iter_mut() {
                            if let Some(point) = by_corner.get(&row.id) {
                                row.camera_point = Some(*point);
                            }
                        }
                        t
                    })
                } else {
                    solve_pnp_measurement(intrinsics, &board_points, &detection)
                };
                if let Ok(t_o2c) = t_o2c {
                    if let Ok((rvec, tvec)) = matrix_to_rvec_tvec(&t_o2c) {
                        if let Ok(zero_dist) = zero_distortion_mat() {
                            let axis_length = (square_length * 3.0).max(0.02) as f32;
                            let _ = calib3d::draw_frame_axes_def(
                                &mut overlay,
                                &camera_mat,
                                &zero_dist,
                                &rvec,
                                &tvec,
                                axis_length,
                            );
                        }
                    }
                }
            }

            let status = format!("Corners: {n_corners} | Rust detection");
            let text_pt = core::Point::new(10, overlay_rows - 12);
            let _ = imgproc::put_text(
                &mut overlay,
                &status,
                text_pt,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
                8,
                false,
            );

            fs::create_dir_all(output_dir)
                .map_err(|err| format!("创建输出目录失败: {err}"))?;
            let _ = imgcodecs::imwrite_def(&output_path.to_string_lossy(), &overlay);

            (
                true,
                n_corners,
                detection.marker_count,
                detection.used_chessboard_fallback,
                "ok".to_string(),
                rows,
            )
        } else {
            fs::create_dir_all(output_dir)
                .map_err(|err| format!("创建输出目录失败: {err}"))?;
            let _ = imgcodecs::imwrite_def(&output_path.to_string_lossy(), &overlay);

            (false, 0, 0, false, "ChArUco detection failed".to_string(), vec![])
        };

    Ok(CharucoDetection {
        image_path: image_path.to_string(),
        output_path: output_path.to_string_lossy().to_string(),
        success,
        num_corners,
        num_markers,
        used_chessboard_fallback,
        message,
        corner_rows,
    })
}

#[tauri::command]
fn detect_charuco(request: CharucoRequest) -> Result<CharucoDetection, String> {
    let image_path = PathBuf::from(&request.image_path);
    let output_dir = image_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("detection");
    let squares_x = request.squares_x.unwrap_or(14);
    let squares_y = request.squares_y.unwrap_or(9);
    let square_length = request.square_length.unwrap_or(0.020);
    let marker_length = request.marker_length.unwrap_or(0.015);
    let aruco_dict = request
        .aruco_dict
        .clone()
        .unwrap_or_else(|| "DICT_5X5_50".to_string());

    // 优先使用 Rust 检测路径（与 3D 预览使用相同的 detect_charuco_observation）
    if let Some(intrinsics) = &request.camera_intrinsics {
        return charuco_detect_and_draw(
            &request.image_path,
            intrinsics,
            squares_x,
            squares_y,
            square_length,
            marker_length,
            &aruco_dict,
            &output_dir,
            requested_depth_path(&request).and_then(Path::to_str),
        );
    }
    if let Some(camera_params) = &request.camera_params {
        if let Ok(content) = fs::read_to_string(camera_params) {
            if let Ok(intrinsics) = parse_camera_params_yaml(&content) {
                return charuco_detect_and_draw(
                    &request.image_path,
                    &intrinsics,
                    squares_x,
                    squares_y,
                    square_length,
                    marker_length,
                    &aruco_dict,
                    &output_dir,
                    requested_depth_path(&request).and_then(Path::to_str),
                );
            }
        }
    }

    Err("需要相机内参才能进行 ChArUco 检测，请先在标定界面选择包含 camera_params.yaml 的数据文件夹".to_string())
}

fn parse_camera_params_yaml(content: &str) -> Result<CameraIntrinsics, String> {
    let Some(data_text) = parse_yaml_matrix_data_block(content, &["camera_matrix", "K"]) else {
        return Err("camera_matrix.data 未找到".to_string());
    };

    let values = parse_yaml_number_list(&data_text)?;
    if values.len() < 9 {
        return Err(format!(
            "camera_matrix.data 需要至少 9 个数值，实际为 {}",
            values.len()
        ));
    }
    Ok(CameraIntrinsics {
        fx: values[0],
        cx: values[2],
        fy: values[4],
        cy: values[5],
        distortion_coefficients: parse_yaml_matrix_data_block(
            content,
            &[
                "distortion_coefficients",
                "D",
                "dist_coeffs",
                "distCoeffs",
                "distortion",
            ],
        )
        .map(|data| parse_yaml_number_list(&data))
        .transpose()?,
    })
}

fn parse_yaml_matrix_data_block(content: &str, keys: &[&str]) -> Option<String> {
    let mut in_matrix = false;
    let mut collecting_data = false;
    let mut data_text = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if keys
            .iter()
            .any(|key| trimmed.starts_with(&format!("{key}:")))
        {
            in_matrix = true;
            collecting_data = false;
            data_text.clear();
            continue;
        }
        if !in_matrix {
            continue;
        }
        if trimmed.contains(':')
            && !trimmed.starts_with("rows:")
            && !trimmed.starts_with("cols:")
            && !trimmed.starts_with("data:")
        {
            break;
        }
        if let Some(data) = trimmed.strip_prefix("data:") {
            collecting_data = true;
            data_text.push_str(data);
            data_text.push(' ');
            if data.contains(']') {
                break;
            }
            continue;
        }
        if collecting_data {
            data_text.push_str(trimmed);
            data_text.push(' ');
            if trimmed.contains(']') {
                break;
            }
        }
    }

    if data_text.trim().is_empty() {
        None
    } else {
        Some(data_text)
    }
}

fn parse_yaml_number_list(text: &str) -> Result<Vec<f64>, String> {
    text.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("无效数值 {value}: {err}"))
        })
        .collect()
}

fn is_rgb_image(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower.ends_with("_color.png") || lower.ends_with("_color.jpg") || lower.ends_with("_color.jpeg")
}

fn is_depth_image(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower.ends_with("_depth.png") || lower.ends_with("_depth.jpg") || lower.ends_with("_depth.jpeg")
}

fn is_raw_depth_candidate(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("raw"))
        && image_frame_prefix(name).is_some()
}

fn image_frame_prefix(name: &str) -> Option<&str> {
    let end = name
        .char_indices()
        .find(|(_, ch)| !ch.is_ascii_digit())
        .map(|(index, _)| index)
        .unwrap_or(name.len());
    if end == 0 {
        None
    } else {
        Some(&name[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_test_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("handeye-manager-{name}-{unique}"));
        fs::create_dir_all(&path).expect("temp test dir should be creatable");
        path
    }

    #[test]
    fn parse_camera_params_yaml_reads_intrinsics_from_ros_matrix() {
        let intrinsics = parse_camera_params_yaml(
            "camera_matrix:\n  rows: 3\n  cols: 3\n  data: [610.75, 0.0, 321.5, 0.0, 611.5, 242.25, 0.0, 0.0, 1.0]\ndistortion_coefficients:\n  rows: 1\n  cols: 5\n  data: [0, 0, 0, 0, 0]\n",
        )
        .expect("camera matrix should parse");

        assert_eq!(intrinsics.fx, 610.75);
        assert_eq!(intrinsics.fy, 611.5);
        assert_eq!(intrinsics.cx, 321.5);
        assert_eq!(intrinsics.cy, 242.25);
        assert_eq!(
            intrinsics.distortion_coefficients,
            Some(vec![0.0, 0.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn colorize_depth_values_maps_nonzero_range_to_rgb_pixels() {
        let pixels = colorize_depth_values(&[0, 1000, 1500, 2000], 2, 2)
            .expect("depth values should colorize");

        assert_eq!(pixels.len(), 12);
        assert_eq!(&pixels[0..3], &[0, 0, 0]);
        assert_ne!(&pixels[3..6], &pixels[6..9]);
        assert_ne!(&pixels[6..9], &pixels[9..12]);
    }

    #[test]
    fn list_depth_images_falls_back_to_raw_when_depth_png_is_missing() {
        let folder = temp_test_dir("depth-list");
        fs::write(folder.join("001_Depth.png"), []).expect("png placeholder should be written");
        fs::write(folder.join("001.raw"), []).expect("raw placeholder should be written");
        fs::write(folder.join("002.raw"), []).expect("raw placeholder should be written");
        fs::write(folder.join("003_Color.png"), []).expect("rgb placeholder should be written");

        let images = list_depth_images(folder.to_string_lossy().to_string())
            .expect("depth listing should succeed");

        let names: Vec<_> = images.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["001_Depth.png", "002.raw"]);
    }

    #[test]
    fn create_depth_preview_reads_raw_using_matching_rgb_dimensions() {
        let folder = temp_test_dir("depth-preview");
        let color_path = folder.join("001_Color.png");
        image::save_buffer(
            &color_path,
            &[0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            2,
            2,
            image::ColorType::Rgb8,
        )
        .expect("rgb image should be written");

        let raw_path = folder.join("001.raw");
        let raw_bytes: Vec<u8> = [0_u16, 1000, 1500, 2000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        fs::write(&raw_path, raw_bytes).expect("raw depth should be written");

        let preview = create_depth_preview(raw_path.to_string_lossy().to_string())
            .expect("raw depth preview should be created");
        let preview_image = image::open(&preview.path)
            .expect("preview image should be readable")
            .to_rgb8();

        assert_eq!(preview_image.dimensions(), (2, 2));
        assert_eq!(preview_image.get_pixel(0, 0).0, [0, 0, 0]);
        assert_ne!(
            preview_image.get_pixel(1, 0).0,
            preview_image.get_pixel(0, 1).0
        );
    }

    #[test]
    fn save_text_file_writes_requested_content() {
        let folder = temp_test_dir("save-text");
        let path = folder.join("nested").join("result.yaml");

        save_text_file(
            path.to_string_lossy().to_string(),
            "setup: eye-in-hand\n".to_string(),
        )
        .expect("yaml file should be written");

        let content = fs::read_to_string(path).expect("saved yaml should be readable");
        assert_eq!(content, "setup: eye-in-hand\n");
    }

    #[test]
    fn build_preview_frames_uses_pose_format_and_transforms() {
        let folder = temp_test_dir("preview-frames");
        fs::write(folder.join("001_Color.png"), []).expect("image placeholder should be written");
        fs::write(folder.join("002_Color.png"), []).expect("image placeholder should be written");
        let poses_file = folder.join("poses.csv");
        fs::write(&poses_file, "100,200,300,0,0,0\n400,500,600,0,0,90\n")
            .expect("poses should be written");

        let request = ConversionPreviewRequest {
            image_dir: folder.to_string_lossy().to_string(),
            poses_file: poses_file.to_string_lossy().to_string(),
            setup: "eye-in-hand".to_string(),
            pose_format: "sxyz".to_string(),
            primary_transform_name: "T_C2F".to_string(),
            primary_matrix_rows: vec![
                "1, 0, 0, 0.1".to_string(),
                "0, 1, 0, 0.2".to_string(),
                "0, 0, 1, 0.3".to_string(),
                "0, 0, 0, 1".to_string(),
            ],
            secondary_transform_name: "T_O2W".to_string(),
            secondary_matrix_rows: vec![
                "1, 0, 0, 1.0".to_string(),
                "0, 1, 0, 2.0".to_string(),
                "0, 0, 1, 3.0".to_string(),
                "0, 0, 0, 1".to_string(),
            ],
            camera_intrinsics: None,
            squares_x: None,
            squares_y: None,
            square_length: None,
            marker_length: None,
            aruco_dict: None,
        };

        let result = build_conversion_preview(request).expect("preview should build");
        assert_eq!(result.preview_frames.len(), 2);
        assert_eq!(
            result.preview_frames[0].image_path,
            folder.join("001_Color.png").to_string_lossy()
        );
        assert_eq!(result.preview_frames[0].camera_in_base[0][3], 0.2);
        assert_eq!(result.preview_frames[0].camera_in_base[1][3], 0.4);
        assert_eq!(result.preview_frames[0].camera_in_base[2][3], 0.6);
        assert_eq!(result.preview_frames[0].board_in_base[0][3], 1.0);
        assert_eq!(result.preview_frames[0].board_in_base[1][3], 2.0);
        assert_eq!(result.preview_frames[1].index, 1);
        assert!(!result.preview_frames[1].camera_in_base.is_empty());
    }

    #[test]
    fn static_euler_order_matches_scipy_lowercase_extrinsic_rotation() {
        let rotation =
            euler_rotation_matrix("xyz", 10.0, 20.0, 30.0, true).expect("rotation should parse");
        let expected = Matrix3::from_row_slice(&[
            0.813797681349,
            -0.440969610530,
            0.378522306370,
            0.469846310393,
            0.882564119259,
            0.018028311236,
            -0.342020143326,
            0.163175911167,
            0.925416578398,
        ]);

        assert!((rotation - expected).abs().max() < 1e-12);
    }

    #[test]
    fn save_calibration_yaml_writes_expected_schema() {
        let folder = temp_test_dir("save-calibration-yaml");
        let path = folder.join("calibration_result.yaml");
        let run = CalibrationRun {
            output_path: path.to_string_lossy().to_string(),
            stdout: String::new(),
            stderr: String::new(),
            setup: "eye-in-hand".to_string(),
            primary_transform_name: "T_C2F".to_string(),
            primary_matrix_rows: vec![
                "1.0000000, 0.0000000, 0.0000000, 0.1000000".to_string(),
                "0.0000000, 1.0000000, 0.0000000, 0.2000000".to_string(),
                "0.0000000, 0.0000000, 1.0000000, 0.3000000".to_string(),
                "0.0000000, 0.0000000, 0.0000000, 1.0000000".to_string(),
            ],
            secondary_transform_name: "T_O2W".to_string(),
            secondary_matrix_rows: vec![
                "1.0000000, 0.0000000, 0.0000000, 1.1000000".to_string(),
                "0.0000000, 1.0000000, 0.0000000, 1.2000000".to_string(),
                "0.0000000, 0.0000000, 1.0000000, 1.3000000".to_string(),
                "0.0000000, 0.0000000, 0.0000000, 1.0000000".to_string(),
            ],
            matrix_rows: vec![],
            average_error_mm: 2.5,
            rotation_error_deg: 0.8,
            reprojection_error_px: 0.42,
            reprojection_rms_px: Some(0.51),
            base_consistency_mean_mm: Some(1.2),
            base_consistency_rms_mm: Some(1.6),
            base_consistency_max_mm: Some(2.4),
            base_consistency_count: Some(84),
            num_images: 5,
            num_images_used: 4,
            filtered_images: vec![3],
            frame_errors: vec![],
            preview_frames: vec![],
            depth_used: false,
            message: "done".to_string(),
        };

        save_calibration_run_yaml(&run).expect("yaml should be written");
        let content = fs::read_to_string(path).expect("yaml should be readable");
        assert!(content.contains("setup: eye-in-hand"));
        assert!(content.contains("T_C2F:"));
        assert!(content.contains("T_O2W:"));
        assert!(content.contains("num_images_used: 4"));
        assert!(content.contains("depth_used: false"));
    }

    #[test]
    fn preview_transforms_use_measured_object_to_camera_when_available() {
        let pose = compose_matrix(Vector3::new(0.1, 0.2, 0.3), Matrix3::identity());
        let primary = compose_matrix(Vector3::new(0.01, 0.02, 0.03), Matrix3::identity());
        let secondary = compose_matrix(Vector3::new(1.0, 2.0, 3.0), Matrix3::identity());
        let measured = compose_matrix(Vector3::new(0.001, 0.002, 0.003), Matrix3::identity());

        let (_, board_in_base, board_in_focus) = preview_frame_transforms_with_measurement(
            "eye-in-hand",
            &pose,
            &primary,
            &secondary,
            Some(&measured),
        );
        assert!((board_in_base[(0, 3)] - 0.111).abs() < 1e-12);
        assert!((board_in_base[(1, 3)] - 0.222).abs() < 1e-12);
        assert_eq!(board_in_base, board_in_focus);

        let (_, board_in_base, board_in_focus) = preview_frame_transforms_with_measurement(
            "eye-to-hand",
            &pose,
            &primary,
            &secondary,
            Some(&measured),
        );
        assert!((board_in_base[(0, 3)] - 0.011).abs() < 1e-12);
        assert!((board_in_base[(1, 3)] - 0.022).abs() < 1e-12);
        assert!((board_in_focus[(0, 3)] + 0.089).abs() < 1e-12);
        assert!((board_in_focus[(1, 3)] + 0.178).abs() < 1e-12);

        let (_, board_in_base_no_pnp, _) = preview_frame_transforms_with_measurement(
            "eye-to-hand",
            &pose,
            &primary,
            &secondary,
            None,
        );
        assert!((board_in_base_no_pnp[(0, 3)] - 1.1).abs() < 1e-12);
    }

    #[test]
    fn pose_error_metrics_report_translation_and_rotation() {
        let pose = Matrix4::identity();
        let measured = vec![Matrix4::identity()];
        let primary = Matrix4::identity();
        let secondary = compose_matrix(
            Vector3::new(0.001, 0.0, 0.0),
            Rotation3::from_euler_angles(0.0, 0.0, 1.0_f64.to_radians()).into_inner(),
        );

        let metrics = compute_pose_errors("eye-to-hand", &[pose], &measured, &primary, &secondary);

        assert!((metrics.translation_mean_mm - 1.0).abs() < 1e-9);
        assert!((metrics.rotation_mean_deg - 1.0).abs() < 1e-9);
        assert_eq!(metrics.per_frame.len(), 1);
    }

    #[test]
    fn eye_in_hand_initialization_falls_back_when_tsai_seed_fails() {
        let poses = vec![Matrix4::identity(), Matrix4::identity(), Matrix4::identity()];
        let measured = vec![Matrix4::identity(), Matrix4::identity(), Matrix4::identity()];

        let (primary, secondary) = initialize_global_transforms("eye-in-hand", &measured, &poses)
            .expect("fallback initialization should succeed");

        assert!(translation_distance(&primary, &Matrix4::identity()) < 1e-12);
        assert!(rotation_distance_deg(&primary, &Matrix4::identity()) < 1e-12);
        assert!(translation_distance(&secondary, &Matrix4::identity()) < 1e-12);
        assert!(rotation_distance_deg(&secondary, &Matrix4::identity()) < 1e-12);
    }

    #[test]
    fn chessboard_fallback_reorders_corners_and_flips_with_marker_ids() {
        let detected = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        ];
        let marker_centers = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(1.0, 1.0),
        ];
        let marker_ids = vec![3, 2, 1, 0];

        let reordered = reorder_chessboard_corners_for_board(
            detected,
            3,
            3,
            Some((&marker_centers, &marker_ids)),
        );

        assert_eq!(reordered.len(), 4);
        assert_eq!(reordered[0], Vector2::new(1.0, 1.0));
        assert_eq!(reordered[1], Vector2::new(0.0, 1.0));
        assert_eq!(reordered[2], Vector2::new(1.0, 0.0));
        assert_eq!(reordered[3], Vector2::new(0.0, 0.0));
    }

    #[test]
    fn homography_reorder_restores_row_major_corner_ids_after_180_rotation() {
        let detected = vec![
            Vector2::new(2.0, 2.0),
            Vector2::new(1.0, 2.0),
            Vector2::new(2.0, 1.0),
            Vector2::new(1.0, 1.0),
        ];
        let board_points = vec![
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(2.0, 1.0, 0.0),
            Vector3::new(1.0, 2.0, 0.0),
            Vector3::new(2.0, 2.0, 0.0),
        ];

        let reordered = reorder_chessboard_corners_via_homography(
            &detected,
            &board_points,
            1.0,
            &Matrix3::identity(),
        )
        .expect("homography reorder should produce a valid mapping");

        assert_eq!(reordered[0], Vector2::new(1.0, 1.0));
        assert_eq!(reordered[1], Vector2::new(2.0, 1.0));
        assert_eq!(reordered[2], Vector2::new(1.0, 2.0));
        assert_eq!(reordered[3], Vector2::new(2.0, 2.0));
    }

    #[test]
    fn detection_quality_gate_rejects_low_quality_fallback_frames() {
        let detection = DetectionObservation {
            index: 0,
            image_path: "frame.png".to_string(),
            corner_ids: (0..12).collect(),
            image_points: (0..12)
                .map(|value| Vector2::new(value as f64, 0.0))
                .collect(),
            marker_count: 0,
            used_chessboard_fallback: true,
        };

        assert!(!detection_passes_quality_gate(&detection, 40, 0.2));
        assert!(detection_passes_quality_gate(&detection, 16, 0.2));
    }

    #[test]
    fn detection_quality_gate_rejects_high_reprojection_error() {
        let detection = DetectionObservation {
            index: 0,
            image_path: "frame.png".to_string(),
            corner_ids: (0..20).collect(),
            image_points: (0..20)
                .map(|value| Vector2::new(value as f64, 0.0))
                .collect(),
            marker_count: 8,
            used_chessboard_fallback: false,
        };

        assert!(detection_passes_quality_gate(&detection, 40, 0.8));
        assert!(!detection_passes_quality_gate(&detection, 40, 2.0));
    }

    #[test]
    fn reference_consistency_prefers_depth_camera_points_when_available() {
        let board_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.1, 0.0, 0.0),
        ];
        let observation = DetectionObservation {
            index: 0,
            image_path: "frame.png".to_string(),
            corner_ids: vec![0, 1],
            image_points: vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)],
            marker_count: 2,
            used_chessboard_fallback: false,
        };
        let poses = vec![Matrix4::identity(), Matrix4::identity()];
        let primary = Matrix4::identity();
        let measured = vec![Matrix4::identity(), Matrix4::identity()];
        let depth_points = vec![
            Some(DepthObservation {
                corner_ids: vec![0, 1],
                object_points: board_points.clone(),
                camera_points: vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)],
            }),
            Some(DepthObservation {
                corner_ids: vec![0, 1],
                object_points: board_points.clone(),
                camera_points: vec![Vector3::new(0.02, 0.0, 1.0), Vector3::new(0.12, 0.0, 1.0)],
            }),
        ];

        let stats = compute_reference_consistency(
            "eye-in-hand",
            &board_points,
            &[observation.clone(), observation],
            &poses,
            &primary,
            &measured,
            &depth_points,
        )
        .expect("depth-backed consistency should be available");

        assert!(stats.rms_m > 0.009);
        assert!(stats.rms_m < 0.011);
    }

    #[test]
    fn reprojection_metrics_keep_derived_and_reference_errors_separate() {
        let intrinsics = CameraIntrinsics {
            cx: 320.0,
            cy: 240.0,
            fx: 1000.0,
            fy: 1000.0,
            distortion_coefficients: None,
        };
        let board_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.1, 0.0, 0.0)];
        let reference_pose = compose_matrix(Vector3::new(0.0, 0.0, 1.0), Matrix3::identity());
        let derived_pose = compose_matrix(Vector3::new(0.01, 0.0, 1.0), Matrix3::identity());
        let observation = DetectionObservation {
            index: 0,
            image_path: "frame.png".to_string(),
            corner_ids: vec![0, 1],
            image_points: board_points
                .iter()
                .map(|point| project_point(&intrinsics, &reference_pose, point))
                .collect(),
            marker_count: 2,
            used_chessboard_fallback: false,
        };

        let reference = compute_reprojection_metrics(
            &intrinsics,
            &board_points,
            &[observation.clone()],
            &[reference_pose],
        );
        let derived = compute_reprojection_metrics(
            &intrinsics,
            &board_points,
            &[observation],
            &[derived_pose],
        );

        assert!(reference.mean_px < 1e-12);
        assert!((derived.mean_px - 10.0).abs() < 1e-9);
        assert_ne!(reference.per_frame[0].0, derived.per_frame[0].0);
    }

    #[test]
    fn consistency_filter_rejects_flipped_object_to_camera_measurement() {
        let primary = compose_matrix(
            Vector3::new(0.05, -0.02, 0.08),
            Rotation3::from_euler_angles(0.1, -0.05, 0.08).into_inner(),
        );
        let target = compose_matrix(
            Vector3::new(0.45, 0.12, 0.70),
            Rotation3::from_euler_angles(-0.08, 0.06, 0.12).into_inner(),
        );
        let poses = vec![
            compose_matrix(
                Vector3::new(0.30, -0.10, 0.50),
                Rotation3::from_euler_angles(0.2, 0.1, -0.15).into_inner(),
            ),
            compose_matrix(
                Vector3::new(0.35, 0.05, 0.55),
                Rotation3::from_euler_angles(0.05, -0.12, 0.20).into_inner(),
            ),
            compose_matrix(
                Vector3::new(0.25, 0.12, 0.62),
                Rotation3::from_euler_angles(-0.10, 0.18, 0.10).into_inner(),
            ),
            compose_matrix(
                Vector3::new(0.40, -0.04, 0.58),
                Rotation3::from_euler_angles(0.16, -0.04, -0.22).into_inner(),
            ),
            compose_matrix(
                Vector3::new(0.34, 0.09, 0.57),
                Rotation3::from_euler_angles(-0.04, 0.15, 0.18).into_inner(),
            ),
        ];
        let mut measured = poses
            .iter()
            .map(|pose| derive_object_to_camera_matrix("eye-in-hand", pose, &primary, &target))
            .collect::<Vec<_>>();
        measured[2] *= compose_matrix(
            Vector3::zeros(),
            Rotation3::from_euler_angles(0.0, 0.0, std::f64::consts::PI).into_inner(),
        );

        let keep = filter_inconsistent_measurements("eye-in-hand", &poses, &measured)
            .expect("filter should fit remaining measurements");

        assert_eq!(keep, vec![0, 1, 3, 4]);
    }

    #[test]
    fn depth_mode_parser_handles_off_optional_and_required() {
        assert!(matches!(
            parse_depth_mode("off").expect("off should parse"),
            DepthMode::Off
        ));
        assert!(matches!(
            parse_depth_mode("optional").expect("optional should parse"),
            DepthMode::Optional
        ));
        assert!(matches!(
            parse_depth_mode("required").expect("required should parse"),
            DepthMode::Required
        ));
        assert!(parse_depth_mode("maybe").is_err());
    }

    #[test]
    fn match_depth_path_prefers_depth_image_and_falls_back_to_raw_prefix() {
        let folder = temp_test_dir("match-depth-path");
        let rgb_path = folder.join("001_Color.png");
        fs::write(&rgb_path, []).expect("rgb placeholder should be written");

        let raw_path = folder.join("001.raw");
        fs::write(&raw_path, []).expect("raw depth placeholder should be written");

        assert_eq!(
            match_depth_path_for_rgb(&rgb_path).expect("depth path should resolve"),
            raw_path
        );

        let depth_png = folder.join("001_Depth.png");
        fs::write(&depth_png, []).expect("depth placeholder should be written");
        assert_eq!(
            match_depth_path_for_rgb(&rgb_path).expect("png depth should win"),
            depth_png
        );
    }

    #[test]
    fn measurement_resolution_marks_depth_used_when_enabled() {
        let folder = temp_test_dir("depth-measurement");
        let rgb_path = folder.join("001_Color.png");
        image::save_buffer(
            &rgb_path,
            &[0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            3,
            3,
            image::ColorType::Rgb8,
        )
        .expect("rgb image should be written");
        let raw_path = folder.join("001.raw");
        let mut raw_values = vec![0_u16; 9];
        raw_values[0] = 1000;
        raw_values[1] = 1000;
        raw_values[3] = 1000;
        raw_values[4] = 1000;
        let raw_bytes: Vec<u8> = raw_values.into_iter().flat_map(u16::to_le_bytes).collect();
        fs::write(&raw_path, raw_bytes).expect("raw depth should be written");

        let intrinsics = CameraIntrinsics {
            cx: 0.0,
            cy: 0.0,
            fx: 10.0,
            fy: 10.0,
            distortion_coefficients: None,
        };
        let board_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.1, 0.0, 0.0),
            Vector3::new(0.0, 0.1, 0.0),
            Vector3::new(0.1, 0.1, 0.0),
        ];
        let observation = DetectionObservation {
            index: 0,
            image_path: rgb_path.to_string_lossy().to_string(),
            corner_ids: vec![0, 1, 2, 3],
            image_points: vec![
                Vector2::new(0.0, 0.0),
                Vector2::new(1.0, 0.0),
                Vector2::new(0.0, 1.0),
                Vector2::new(1.0, 1.0),
            ],
            marker_count: 4,
            used_chessboard_fallback: false,
        };

        let (measurements, depth_observations, depth_used) = resolve_object_to_camera_measurements(
            &DepthMode::Optional,
            &intrinsics,
            &board_points,
            &[observation],
        )
        .expect("measurements should resolve");

        assert!(depth_used);
        assert_eq!(measurements.len(), 1);
        assert_eq!(depth_observations.len(), 1);
        assert!(depth_observations[0].is_some());
        assert!((measurements[0][(2, 3)] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn charuco_request_keeps_depth_path_available() {
        let request = CharucoRequest {
            image_path: "frame.png".to_string(),
            depth_path: Some("/tmp/frame.raw".to_string()),
            camera_params: None,
            camera_intrinsics: None,
            squares_x: None,
            squares_y: None,
            square_length: None,
            marker_length: None,
            aruco_dict: None,
        };

        assert_eq!(
            requested_depth_path(&request),
            Some(Path::new("/tmp/frame.raw"))
        );
    }

    #[test]
    fn optimize_handeye_from_observations_recovers_eye_in_hand_transforms() {
        let intrinsics = CameraIntrinsics {
            cx: 320.0,
            cy: 240.0,
            fx: 640.0,
            fy: 640.0,
            distortion_coefficients: Some(vec![0.0, 0.0, 0.0, 0.0, 0.0]),
        };
        let board_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.04, 0.0, 0.0),
            Vector3::new(0.08, 0.0, 0.0),
            Vector3::new(0.0, 0.04, 0.0),
            Vector3::new(0.04, 0.04, 0.0),
            Vector3::new(0.08, 0.04, 0.0),
        ];
        let primary = compose_matrix(
            Vector3::new(0.05, -0.02, 0.08),
            Rotation3::from_euler_angles(0.1, -0.05, 0.08).into_inner(),
        );
        let target = compose_matrix(
            Vector3::new(0.45, 0.12, 0.70),
            Rotation3::from_euler_angles(-0.08, 0.06, 0.12).into_inner(),
        );
        let poses = vec![
            compose_matrix(
                Vector3::new(0.30, -0.10, 0.50),
                Rotation3::from_euler_angles(0.2, 0.1, -0.15).into_inner(),
            ),
            compose_matrix(
                Vector3::new(0.35, 0.05, 0.55),
                Rotation3::from_euler_angles(0.05, -0.12, 0.20).into_inner(),
            ),
            compose_matrix(
                Vector3::new(0.25, 0.12, 0.62),
                Rotation3::from_euler_angles(-0.10, 0.18, 0.10).into_inner(),
            ),
            compose_matrix(
                Vector3::new(0.40, -0.04, 0.58),
                Rotation3::from_euler_angles(0.16, -0.04, -0.22).into_inner(),
            ),
        ];
        let detections = poses
            .iter()
            .enumerate()
            .map(|(index, pose)| {
                let object_to_camera =
                    derive_object_to_camera_matrix("eye-in-hand", pose, &primary, &target);
                let image_points = board_points
                    .iter()
                    .map(|point| project_point(&intrinsics, &object_to_camera, point))
                    .collect::<Vec<_>>();
                SyntheticDetection {
                    index,
                    image_path: format!("frame_{index:03}.png"),
                    corner_ids: (0..board_points.len()).collect(),
                    image_points,
                    marker_count: board_points.len(),
                    used_chessboard_fallback: false,
                }
            })
            .collect::<Vec<_>>();
        let observations = detections
            .iter()
            .map(ObservationLike::to_observation)
            .collect::<Vec<_>>();
        let measured = poses
            .iter()
            .map(|pose| derive_object_to_camera_matrix("eye-in-hand", pose, &primary, &target))
            .collect::<Vec<_>>();
        let (primary_init, secondary_init) =
            initialize_global_transforms("eye-in-hand", &measured, &poses)
                .expect("initialization should succeed");

        let solution = optimize_handeye_from_measurements(
            "eye-in-hand",
            &intrinsics,
            &board_points,
            &observations,
            &poses,
            &measured,
            &vec![None; observations.len()],
        )
        .expect("synthetic optimization should succeed");

        eprintln!("primary init = {:?}", primary_init);
        eprintln!("target init = {:?}", secondary_init);
        eprintln!("primary true = {:?}", primary);
        eprintln!("primary est = {:?}", solution.primary_transform);
        eprintln!("target true = {:?}", target);
        eprintln!("target est = {:?}", solution.secondary_transform);
        assert!(translation_distance(&solution.primary_transform, &primary) < 1e-3);
        assert!(rotation_distance_deg(&solution.primary_transform, &primary) < 0.2);
        assert!(translation_distance(&solution.secondary_transform, &target) < 1e-3);
        assert!(rotation_distance_deg(&solution.secondary_transform, &target) < 0.2);
        assert!(solution.reprojection_mean_px < 1e-6);
    }

    #[test]
    fn configured_charuco_detector_params_tune_main_path_for_precision() {
        let params =
            configured_charuco_detector_params().expect("detector parameters should be created");

        assert_eq!(params.corner_refinement_method(), 1);
        assert_eq!(params.corner_refinement_win_size(), 5);
        assert_eq!(params.corner_refinement_max_iterations(), 30);
        assert!((params.corner_refinement_min_accuracy() - 0.001).abs() < f64::EPSILON);
        assert_eq!(params.adaptive_thresh_win_size_min(), 3);
        assert_eq!(params.adaptive_thresh_win_size_max(), 23);
        assert_eq!(params.adaptive_thresh_win_size_step(), 10);
        assert!((params.min_marker_perimeter_rate() - 0.02).abs() < f64::EPSILON);
        assert!((params.max_marker_perimeter_rate() - 4.0).abs() < f64::EPSILON);
        assert!(!params.use_aruco3_detection());
    }

    #[test]
    fn preprocess_detection_gray_matches_plain_grayscale() {
        let input = Mat::new_rows_cols_with_default(
            6,
            8,
            core::CV_8UC3,
            core::Scalar::new(32.0, 96.0, 160.0, 0.0),
        )
        .expect("test image should be created");

        let gray =
            preprocess_detection_gray(&input).expect("gray preprocessing should succeed");
        let mut expected = Mat::default();
        imgproc::cvt_color(
            &input,
            &mut expected,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .expect("plain grayscale should succeed");

        assert_eq!(gray.rows(), 6);
        assert_eq!(gray.cols(), 8);
        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.typ() & core::CV_MAT_DEPTH_MASK, core::CV_8U);
        let diff = core::norm2_def(&gray, &expected).expect("gray diff should compute");
        assert!(diff.abs() < f64::EPSILON);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            list_rgb_images,
            list_depth_images,
            create_depth_preview,
            read_camera_params,
            read_pose_file,
            save_text_file,
            run_handeye_calibration,
            build_conversion_preview,
            detect_charuco
        ])
        .run(tauri::generate_context!())
        .expect("error while running HandEyeManager UI");
}
