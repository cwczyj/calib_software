use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;

const PYTHON_API_HOST: &str = "127.0.0.1";
const PYTHON_API_PORT: u16 = 18765;
static PYTHON_API_CHILD: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

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

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CharucoHttpRequest {
    image_path: String,
    depth_path: Option<String>,
    camera_params: Option<String>,
    camera_intrinsics: Option<CameraIntrinsics>,
    squares_x: usize,
    squares_y: usize,
    square_length: f64,
    marker_length: f64,
    aruco_dict: String,
    output_dir: String,
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
    image::save_buffer(&preview_path, &pixels, width, height, image::ColorType::Rgb8)
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
            raw_by_prefix.entry(prefix.to_string()).or_insert_with(|| ImageFile {
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
        aruco_dict: request.aruco_dict.unwrap_or_else(|| "DICT_5X5_50".to_string()),
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
}

fn run_handeye_calibration_direct(
    payload: &CalibrationHttpRequest,
) -> Result<CalibrationRun, String> {
    let mut args = vec![
        "gui_api.py".to_string(),
        "run-calibration".to_string(),
        payload.image_dir.clone(),
        payload.poses_file.clone(),
        "--setup".to_string(),
        payload.setup.clone(),
        "--pose_format".to_string(),
        payload.pose_format.clone(),
        "--use_depth".to_string(),
        payload.use_depth.clone(),
        "--squares_x".to_string(),
        payload.squares_x.to_string(),
        "--squares_y".to_string(),
        payload.squares_y.to_string(),
        "--square_length".to_string(),
        payload.square_length.to_string(),
        "--marker_length".to_string(),
        payload.marker_length.to_string(),
        "--aruco_dict".to_string(),
        payload.aruco_dict.clone(),
        "--output".to_string(),
        payload.output_path.clone(),
    ];
    if let Some(camera_params) = &payload.camera_params {
        args.extend(["-c".to_string(), camera_params.clone()]);
    }
    if let Some(intrinsics) = &payload.camera_intrinsics {
        args.extend([
            "--cx".to_string(),
            intrinsics.cx.to_string(),
            "--cy".to_string(),
            intrinsics.cy.to_string(),
            "--fx".to_string(),
            intrinsics.fx.to_string(),
            "--fy".to_string(),
            intrinsics.fy.to_string(),
        ]);
        if let Some(distortion) = &intrinsics.distortion_coefficients {
            let joined = distortion
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            if !joined.is_empty() {
                args.push(format!("--distortion_coefficients={joined}"));
            }
        }
    }
    if !payload.excluded_image_indices.is_empty() {
        args.extend([
            "--excluded_image_indices".to_string(),
            payload
                .excluded_image_indices
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ]);
    }

    let output = run_python(args)?;
    if !output.status.success() {
        return Err(format!(
            "标定失败\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_calibration_run(String::from_utf8_lossy(&output.stdout).trim())
}

#[tauri::command]
fn build_conversion_preview(request: ConversionPreviewRequest) -> Result<ConversionPreviewResult, String> {
    let payload = ConversionPreviewHttpRequest {
        image_dir: request.image_dir,
        poses_file: request.poses_file,
        setup: request.setup,
        pose_format: request.pose_format,
        primary_transform_name: request.primary_transform_name,
        primary_matrix: flatten_matrix_rows(&request.primary_matrix_rows),
        secondary_transform_name: request.secondary_transform_name,
        secondary_matrix: flatten_matrix_rows(&request.secondary_matrix_rows),
    };
    build_conversion_preview_direct(&payload)
}

fn build_conversion_preview_direct(
    payload: &ConversionPreviewHttpRequest,
) -> Result<ConversionPreviewResult, String> {
    let args = vec![
        "gui_api.py".to_string(),
        "build-preview".to_string(),
        payload.image_dir.clone(),
        payload.poses_file.clone(),
        "--setup".to_string(),
        payload.setup.clone(),
        "--pose_format".to_string(),
        payload.pose_format.clone(),
        "--primary_transform_name".to_string(),
        payload.primary_transform_name.clone(),
        "--secondary_transform_name".to_string(),
        payload.secondary_transform_name.clone(),
        "--primary_matrix".to_string(),
        payload.primary_matrix.clone(),
        "--secondary_matrix".to_string(),
        payload.secondary_matrix.clone(),
    ];
    let output = run_python(args)?;
    if !output.status.success() {
        return Err(format!(
            "生成输出预览失败\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_str(String::from_utf8_lossy(&output.stdout).trim())
        .map_err(|err| format!("解析输出预览失败: {err}"))
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

#[tauri::command]
fn detect_charuco(request: CharucoRequest) -> Result<CharucoDetection, String> {
    let image_path = PathBuf::from(&request.image_path);
    let output_dir = image_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("detection");
    let image_path_string = request.image_path.clone();
    let camera_params = request.camera_params.clone();
    let depth_path = request.depth_path.clone();
    let camera_intrinsics = request.camera_intrinsics.clone();
    let squares_x = request.squares_x.unwrap_or(14);
    let squares_y = request.squares_y.unwrap_or(9);
    let square_length = request.square_length.unwrap_or(0.020);
    let marker_length = request.marker_length.unwrap_or(0.015);
    let aruco_dict = request
        .aruco_dict
        .clone()
        .unwrap_or_else(|| "DICT_5X5_50".to_string());
    ensure_python_api_server()?;

    let payload = CharucoHttpRequest {
        image_path: image_path_string.clone(),
        depth_path: depth_path.clone(),
        camera_params: camera_params.clone(),
        camera_intrinsics: camera_intrinsics.clone(),
        squares_x,
        squares_y,
        square_length,
        marker_length,
        aruco_dict: aruco_dict.clone(),
        output_dir: output_dir.to_string_lossy().to_string(),
    };
    let response = post_python_api("/detect-charuco", &payload)?;
    if !response.contains("\"cornerRows\"")
        || (depth_path.is_some() && !response.contains("\"cameraPoint\""))
    {
        return detect_charuco_direct(
            image_path_string,
            depth_path,
            camera_params,
            camera_intrinsics,
            squares_x,
            squares_y,
            square_length,
            marker_length,
            aruco_dict,
            output_dir,
        );
    }
    parse_charuco_detection(&response)
}

fn detect_charuco_direct(
    image_path: String,
    depth_path: Option<String>,
    camera_params: Option<String>,
    camera_intrinsics: Option<CameraIntrinsics>,
    squares_x: usize,
    squares_y: usize,
    square_length: f64,
    marker_length: f64,
    aruco_dict: String,
    output_dir: PathBuf,
) -> Result<CharucoDetection, String> {
    let mut args = vec![
        "gui_api.py".to_string(),
        "detect-charuco".to_string(),
        image_path,
        "--output_dir".to_string(),
        output_dir.to_string_lossy().to_string(),
        "--squares_x".to_string(),
        squares_x.to_string(),
        "--squares_y".to_string(),
        squares_y.to_string(),
        "--square_length".to_string(),
        square_length.to_string(),
        "--marker_length".to_string(),
        marker_length.to_string(),
        "--aruco_dict".to_string(),
        aruco_dict,
    ];
    if let Some(camera_params) = camera_params {
        args.extend(["-c".to_string(), camera_params]);
    }
    if let Some(depth_path) = depth_path {
        args.extend(["--depth_path".to_string(), depth_path]);
    }
    if let Some(intrinsics) = camera_intrinsics {
        args.extend([
            "--cx".to_string(),
            intrinsics.cx.to_string(),
            "--cy".to_string(),
            intrinsics.cy.to_string(),
            "--fx".to_string(),
            intrinsics.fx.to_string(),
            "--fy".to_string(),
            intrinsics.fy.to_string(),
        ]);
        if let Some(distortion) = intrinsics.distortion_coefficients {
            let joined = distortion
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            if !joined.is_empty() {
                args.push(format!("--distortion_coefficients={joined}"));
            }
        }
    }

    let output = run_python(args)?;
    if !output.status.success() {
        return Err(format!(
            "ChArUco 识别失败\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_charuco_detection(String::from_utf8_lossy(&output.stdout).trim())
}

fn parse_charuco_detection(response: &str) -> Result<CharucoDetection, String> {
    serde_json::from_str(&response).map_err(|err| format!("解析 ChArUco 结果失败: {err}"))
}

fn parse_calibration_run(response: &str) -> Result<CalibrationRun, String> {
    serde_json::from_str(response)
        .or_else(|_| {
            response
                .lines()
                .rev()
                .map(str::trim)
                .find(|line| line.starts_with('{') && line.ends_with('}'))
                .ok_or_else(|| {
                    serde_json::Error::io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "stdout 中没有 JSON 标定结果",
                    ))
                })
                .and_then(serde_json::from_str)
        })
        .map_err(|err| format!("解析标定结果失败: {err}"))
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
    if end == 0 { None } else { Some(&name[..end]) }
}

fn run_python(args: Vec<String>) -> Result<std::process::Output, String> {
    let script_dir = repo_root().join("st_handeye_calibration");
    for python in ["python3", "python"] {
        match Command::new(python)
            .args(&args)
            .current_dir(&script_dir)
            .output()
        {
            Ok(output) => return Ok(output),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(format!("启动 Python 失败: {err}")),
        }
    }
    Err("未找到 python3 或 python".to_string())
}

fn ensure_python_api_server() -> Result<(), String> {
    if python_api_is_healthy() {
        return Ok(());
    }

    let child_slot = PYTHON_API_CHILD.get_or_init(|| Mutex::new(None));
    let mut child_guard = child_slot
        .lock()
        .map_err(|_| "Python API 服务状态锁定失败".to_string())?;

    if python_api_is_healthy() {
        return Ok(());
    }
    if child_guard
        .as_mut()
        .is_some_and(|child| child.try_wait().ok().flatten().is_none())
    {
        return wait_for_python_api();
    }

    let script_dir = repo_root().join("st_handeye_calibration");
    let args = [
        "api_server.py",
        "--host",
        PYTHON_API_HOST,
        "--port",
        &PYTHON_API_PORT.to_string(),
    ];
    for python in ["python3", "python"] {
        match Command::new(python)
            .args(args)
            .current_dir(&script_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => {
                *child_guard = Some(child);
                return wait_for_python_api();
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(format!("启动 Python API 服务失败: {err}")),
        }
    }
    Err("未找到 python3 或 python".to_string())
}

fn wait_for_python_api() -> Result<(), String> {
    for _ in 0..50 {
        if python_api_is_healthy() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err("Python API 服务启动超时".to_string())
}

fn python_api_is_healthy() -> bool {
    get_python_api("/health").is_ok()
}

fn get_python_api(path: &str) -> Result<String, String> {
    request_python_api("GET", path, "")
}

fn post_python_api<T: Serialize>(path: &str, payload: &T) -> Result<String, String> {
    let body = serde_json::to_string(payload).map_err(|err| format!("序列化请求失败: {err}"))?;
    request_python_api("POST", path, &body)
}

fn request_python_api(method: &str, path: &str, body: &str) -> Result<String, String> {
    let mut stream = TcpStream::connect((PYTHON_API_HOST, PYTHON_API_PORT))
        .map_err(|err| format!("连接 Python API 服务失败: {err}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(300)))
        .map_err(|err| format!("设置 Python API 读取超时失败: {err}"))?;
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {PYTHON_API_HOST}:{PYTHON_API_PORT}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.as_bytes().len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|err| format!("发送 Python API 请求失败: {err}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|err| format!("读取 Python API 响应失败: {err}"))?;
    parse_http_response(&response)
}

fn parse_http_response(response: &str) -> Result<String, String> {
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| "Python API 响应格式错误".to_string())?;
    let status_line = headers.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        return Err(format!("Python API 请求失败: {status_line}\n{body}"));
    }
    Ok(body.to_string())
}

#[cfg(test)]
fn is_missing_python_api_route(error: &str) -> bool {
    error.contains(" 404 ") || error.contains("\"not found\"") || error.contains("Not Found")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
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
    fn parse_charuco_detection_defaults_missing_corner_rows_to_empty() {
        let detection = parse_charuco_detection(
            r#"{"imagePath":"image.png","outputPath":"detection.png","success":true,"numCorners":24,"numMarkers":12,"message":"ok"}"#,
        )
        .expect("legacy ChArUco response should parse");

        assert!(detection.corner_rows.is_empty());
        assert_eq!(detection.num_corners, 24);
    }

    #[test]
    fn parse_charuco_detection_reads_corner_rows() {
        let detection = parse_charuco_detection(
            r#"{"imagePath":"image.png","outputPath":"detection.png","success":true,"numCorners":1,"numMarkers":1,"message":"ok","cornerRows":[{"id":7,"imagePoint":[766.78,518.89],"cameraPoint":[0.028,0.014,1.0]}]}"#,
        )
        .expect("current ChArUco response should parse");

        assert_eq!(detection.corner_rows.len(), 1);
        assert_eq!(detection.corner_rows[0].id, 7);
        assert_eq!(detection.corner_rows[0].image_point, [766.78, 518.89]);
        assert_eq!(
            detection.corner_rows[0].camera_point,
            Some([0.028, 0.014, 1.0])
        );
    }

    #[test]
    fn parse_calibration_run_reads_structured_result() {
        let result = parse_calibration_run(
            r#"{"outputPath":"result.yaml","setup":"eye-to-hand","primaryTransformName":"T_C2W","matrixRows":["1.0000000, 0.0000000, 0.0000000, 0.1000000"],"averageErrorMm":2.5,"rotationErrorDeg":0.8,"reprojectionErrorPx":0.42,"reprojectionRmsPx":0.51,"baseConsistencyMeanMm":1.1,"baseConsistencyRmsMm":1.5,"baseConsistencyMaxMm":2.3,"baseConsistencyCount":80,"numImages":5,"numImagesUsed":4,"filteredImages":[3],"frameErrors":[{"index":0,"imagePath":"001_Color.png","used":true,"cornerCount":42,"reprojectionMeanPx":0.21,"reprojectionRmsPx":0.27,"reprojectionMaxPx":0.8,"referenceReprojectionMeanPx":0.18,"referenceReprojectionRmsPx":0.22,"referenceReprojectionMaxPx":0.62,"reprojectionErrorPx":0.21,"optimizedReprojectionErrorPx":0.18,"baseConsistencyMeanMm":1.0,"baseConsistencyRmsMm":1.2,"baseConsistencyMaxMm":2.0,"baseConsistencyCount":42,"translationErrorMm":1.2,"rotationErrorDeg":0.3}],"depthUsed":false,"message":"done"}"#,
        )
        .expect("current calibration response should parse");

        assert_eq!(result.output_path, "result.yaml");
        assert_eq!(result.setup, "eye-to-hand");
        assert_eq!(result.primary_transform_name, "T_C2W");
        assert_eq!(result.matrix_rows.len(), 1);
        assert_eq!(result.average_error_mm, 2.5);
        assert_eq!(result.num_images_used, 4);
        assert_eq!(result.reprojection_rms_px, Some(0.51));
        assert_eq!(result.base_consistency_mean_mm, Some(1.1));
        assert_eq!(result.base_consistency_rms_mm, Some(1.5));
        assert_eq!(result.base_consistency_max_mm, Some(2.3));
        assert_eq!(result.base_consistency_count, Some(80));
        assert_eq!(result.filtered_images, vec![3]);
        assert_eq!(result.frame_errors.len(), 1);
        assert_eq!(result.frame_errors[0].image_path, "001_Color.png");
        assert_eq!(result.frame_errors[0].corner_count, Some(42));
        assert_eq!(result.frame_errors[0].reprojection_rms_px, Some(0.27));
        assert_eq!(result.frame_errors[0].reference_reprojection_mean_px, Some(0.18));
        assert_eq!(result.frame_errors[0].base_consistency_rms_mm, Some(1.2));
        assert_eq!(result.frame_errors[0].base_consistency_count, Some(42));
        assert_eq!(result.frame_errors[0].translation_error_mm, Some(1.2));
    }

    #[test]
    fn parse_calibration_run_ignores_cli_progress_logs_before_json() {
        let result = parse_calibration_run(
            "Loading from /tmp/session...\nDetecting corners...\nOptimizing...\n{\"outputPath\":\"result.yaml\",\"setup\":\"eye-in-hand\",\"primaryTransformName\":\"T_C2F\",\"matrixRows\":[\"1.0000000, 0.0000000, 0.0000000, 0.0000000\"],\"averageErrorMm\":1.2,\"numImages\":6,\"numImagesUsed\":5,\"message\":\"done\"}\n",
        )
        .expect("CLI fallback output should parse its final JSON line");

        assert_eq!(result.output_path, "result.yaml");
        assert_eq!(result.primary_transform_name, "T_C2F");
        assert_eq!(result.average_error_mm, 1.2);
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
    fn stale_python_api_404_triggers_calibration_fallback() {
        assert!(is_missing_python_api_route(
            "Python API 请求失败: HTTP/1.0 404 Not Found\n{\"error\": \"not found\"}"
        ));
        assert!(!is_missing_python_api_route(
            "Python API 请求失败: HTTP/1.0 500 Internal Server Error\n{\"detail\":\"Not enough detections\"}"
        ));
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
        assert_ne!(preview_image.get_pixel(1, 0).0, preview_image.get_pixel(0, 1).0);
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
