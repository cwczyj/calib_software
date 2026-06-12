"""I/O utilities for hand-eye calibration.

Handles reading/writing matrices, camera parameters (ROS YAML), pattern generation,
and dataset loading. Compatible with the C++ version's data formats.
"""
import os
import numpy as np


def read_matrix_csv(filename):
    """Read a whitespace-separated CSV file into a numpy array.

    Matches C++ Dataset::read_matrix() in calibrate.cpp.
    """
    with open(filename, 'r') as f:
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                values = [float(x.strip()) for x in line.split(',')]
            else:
                values = [float(x) for x in line.split()]
            rows.append(values)
    if not rows:
        return np.array([])
    return np.array(rows)


def write_matrix_csv(matrix, filename):
    """Write a numpy matrix to a whitespace-separated CSV file.

    Matches the C++ output format (Eigen matrix streaming to ofstream).
    """
    m = np.asarray(matrix)
    with open(filename, 'w') as f:
        for i in range(m.shape[0]):
            row = ' '.join(f'{m[i, j]:.10g}' for j in range(m.shape[1]))
            f.write(row + '\n')


def _read_yaml_matrix(ifs):
    """Parse a matrix block from a ROS camera YAML file.

    Matches C++ Dataset::read_matrix_from_yaml() in calibrate.cpp.
    Expects format: 'rows: N\\ncols: M\\ndata: [v1, v2, ...]'
    """
    token = None
    rows = 0
    cols = 0

    # Read 'rows: N'
    for line in ifs:
        line = line.strip()
        if line.startswith('rows:'):
            rows = int(line.split(':')[1].strip())
            break

    # Read 'cols: M'
    for line in ifs:
        line = line.strip()
        if line.startswith('cols:'):
            cols = int(line.split(':')[1].strip())
            break

    # Read 'data: [...]' — may span multiple lines
    data_str = ''
    found_data = False
    for line in ifs:
        line_stripped = line.strip()
        if line_stripped.startswith('data:'):
            found_data = True
            data_str += line_stripped[len('data:'):].strip()
            if ']' in data_str:
                break
        elif found_data:
            data_str += ' ' + line_stripped
            if ']' in data_str:
                break

    # Parse bracketed values: [v1, v2, v3, ...]
    data_str = data_str.strip()
    if data_str.startswith('['):
        data_str = data_str[1:]
    if data_str.endswith(']'):
        data_str = data_str[:-1]

    values = [float(x.strip()) for x in data_str.split(',') if x.strip()]
    return np.array(values).reshape(rows, cols)


def read_ros_camera_params(filename):
    """Read camera intrinsics from common ROS/OpenCV YAML formats.

    Matches C++ Dataset::read_ros_camera_params() in calibrate.cpp.
    Returns (camera_matrix 3x3, distortion_coeffs 1xN).
    """
    parsed = _read_camera_params_with_yaml(filename)
    if parsed is not None:
        return parsed

    with open(filename, 'r') as ifs:
        lines = list(ifs)

    camera_matrix = None
    distortion = None

    # Find and parse camera_matrix block
    for i, line in enumerate(lines):
        if 'camera_matrix' in line and ':' in line:
            # Create a sub-iterator from the next lines
            from io import StringIO
            sub = StringIO('\n'.join(lines[i + 1:]))
            camera_matrix = _read_yaml_matrix(sub)
            break

    # Find and parse distortion_coefficients block
    for i, line in enumerate(lines):
        if 'distortion_coefficients' in line and ':' in line:
            from io import StringIO
            sub = StringIO('\n'.join(lines[i + 1:]))
            distortion = _read_yaml_matrix(sub)
            break

    if camera_matrix is None:
        raise ValueError(f'camera_matrix not found in {filename}')
    if distortion is None:
        raise ValueError(f'distortion_coefficients not found in {filename}')

    # C++ transposes distortion after reading
    distortion = distortion.T.flatten()

    return camera_matrix, distortion


def _sanitize_yaml_text(text):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('%YAML:'):
            continue
        line = line.replace('!!opencv-matrix', '')
        lines.append(line)
    return '\n'.join(lines)


def _matrix_from_yaml_value(value, shape=None):
    if value is None:
        return None
    if isinstance(value, dict):
        data = value.get('data')
        rows = value.get('rows')
        cols = value.get('cols')
        if data is None:
            return None
        arr = np.asarray(data, dtype=np.float64)
        if rows is not None and cols is not None:
            return arr.reshape(int(rows), int(cols))
        if shape is not None:
            return arr.reshape(shape)
        return arr
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None:
        return arr.reshape(shape)
    return arr


def _read_camera_params_with_yaml(filename):
    try:
        import yaml
    except ImportError:
        return None

    with open(filename, 'r') as f:
        text = _sanitize_yaml_text(f.read())

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None

    camera_value = None
    for key in ('camera_matrix', 'K', 'cameraMatrix', 'intrinsic_matrix'):
        if key in data:
            camera_value = data[key]
            break

    distortion_value = None
    for key in ('distortion_coefficients', 'D', 'dist_coeffs', 'distCoeffs', 'distortion'):
        if key in data:
            distortion_value = data[key]
            break

    camera_matrix = _matrix_from_yaml_value(camera_value, (3, 3))
    distortion = _matrix_from_yaml_value(distortion_value)
    if camera_matrix is None or distortion is None:
        return None

    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    return camera_matrix, distortion


def write_ros_transform_yaml(filename, name, matrix):
    """Write a 4x4 transform matrix to a ROS YAML file.

    Matches the ROS camera_params.yaml format for camera_matrix:
    ```yaml
    name:
      rows: 4
      cols: 4
      data: [m00, m01, ..., m33]
    ```

    Args:
        filename: Output YAML file path.
        name: Transform block name (e.g., 'hand2eye', 'object2world').
        matrix: 4x4 numpy array (Eigen::Isometry3d / np.eye(4) format).
    """
    m = np.asarray(matrix)
    if m.shape != (4, 4):
        raise ValueError(f'matrix must be 4x4, got {m.shape}')

    data = m.flatten()
    data_str = ', '.join(f'{v:.10g}' for v in data)

    with open(filename, 'w') as f:
        f.write(f'{name}:\n')
        f.write(f'  rows: 4\n')
        f.write(f'  cols: 4\n')
        f.write(f'  data: [{data_str}]\n')


def write_calibration_results_yaml(filename, T_C2F, T_O2W):
    """Write T_C2F and T_O2W transforms to a single YAML file.

    Coordinate systems: F=Flange, C=Camera, O=Object, W=World
    Transform naming: T_A2B means P_B = T_A2B @ P_A

    Format:
    ```yaml
    T_C2F:
      rows: 4
      cols: 4
      data: [...]
    T_O2W:
      rows: 4
      cols: 4
      data: [...]
    ```

    Args:
        filename: Output YAML file path.
        T_C2F: 4x4 Camera-to-Flange transform (P_F = T_C2F @ P_C).
        T_O2W: 4x4 Object-to-World transform (P_W = T_O2W @ P_O).
    """
    from datetime import datetime

    with open(filename, 'w') as f:
        f.write(f'# Hand-eye calibration results (generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")})\n')
        f.write('# Coordinate systems: F=Flange, C=Camera, O=Object(board), W=World\n')
        f.write('# Transform naming: T_A2B means P_B = T_A2B @ P_A\n')
        f.write('\n')
        T_C2F_arr = np.asarray(T_C2F)
        if T_C2F_arr.shape != (4, 4):
            raise ValueError(f'T_C2F must be 4x4, got {T_C2F_arr.shape}')
        data_c2f = T_C2F_arr.flatten()
        data_str_c2f = ', '.join(f'{v:.10g}' for v in data_c2f)
        f.write('T_C2F:\n')
        f.write('  rows: 4\n')
        f.write('  cols: 4\n')
        f.write(f'  data: [{data_str_c2f}]\n')
        f.write('\n')
        T_O2W_arr = np.asarray(T_O2W)
        if T_O2W_arr.shape != (4, 4):
            raise ValueError(f'T_O2W must be 4x4, got {T_O2W_arr.shape}')
        data_o2w = T_O2W_arr.flatten()
        data_str_o2w = ', '.join(f'{v:.10g}' for v in data_o2w)
        f.write('T_O2W:\n')
        f.write('  rows: 4\n')
        f.write('  cols: 4\n')
        f.write(f'  data: [{data_str_o2w}]\n')


def make_pattern_3d(pattern_rows, pattern_cols, spacing):
    """Generate 3D pattern points for an asymmetric circle grid.

    Matches C++ pattern generation in calibrate.cpp:72-83.
    Returns a 3xN numpy array of pattern point coordinates.

    For asymmetric circle grid: even columns have no y-offset,
    odd columns have y-offset of spacing/2.
    """
    n_points = pattern_rows * pattern_cols
    pattern_3d = np.zeros((3, n_points))
    for j in range(pattern_cols):
        for i in range(pattern_rows):
            y_offset = spacing / 2 if j % 2 != 0 else 0.0
            idx = j * pattern_rows + i
            pattern_3d[0, idx] = j * spacing / 2
            pattern_3d[1, idx] = i * spacing + y_offset
            pattern_3d[2, idx] = 0.0
    return pattern_3d


def read_dataset(dataset_dir, camera_params_file=None, visualize=False,
                 pattern_rows=4, pattern_cols=11, spacing=0.032):
    """Read a calibration dataset from a directory.

    Matches C++ Dataset::read() in calibrate.cpp.
    Expects files: {NNN}_image.jpg, {NNN}_pose.csv,
    optional {NNN}_camera_matrix.csv, {NNN}_distortion.csv.

    Returns dict with keys: camera_matrix, distortion, pattern_3d, pattern_2ds, world2hands.
    """
    pattern_3d = make_pattern_3d(pattern_rows, pattern_cols, spacing)
    camera_matrix = None
    distortion = None

    if camera_params_file:
        camera_matrix, distortion = read_ros_camera_params(camera_params_file)

    import cv2
    pattern_2ds = []
    world2hands = []

    for fname in sorted(os.listdir(dataset_dir)):
        if '_image.jpg' not in fname:
            continue

        data_id = fname[fname.index('_image.jpg') - 3:fname.index('_image.jpg')]
        image_path = os.path.join(dataset_dir, f'{data_id}_image.jpg')
        pose_path = os.path.join(dataset_dir, f'{data_id}_pose.csv')

        image = cv2.imread(image_path)
        handpose = read_matrix_csv(pose_path)

        if camera_params_file is None:
            cam_path = os.path.join(dataset_dir, f'{data_id}_camera_matrix.csv')
            dist_path = os.path.join(dataset_dir, f'{data_id}_distortion.csv')
            camera_matrix = read_matrix_csv(cam_path)
            distortion = read_matrix_csv(dist_path).flatten()

        cv_K = np.array(camera_matrix, dtype=np.float64)
        cv_dist = np.array(distortion, dtype=np.float64)
        undistorted = cv2.undistort(image, cv_K, cv_dist)

        cv_grid_2d = None
        found, cv_grid_2d = cv2.findCirclesGrid(
            undistorted,
            (pattern_rows, pattern_cols),
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )

        if not found:
            print(f'failed to find circles in {fname}')
            continue

        grid_2d = np.zeros((2, pattern_rows * pattern_cols))
        for i in range(pattern_rows * pattern_cols):
            grid_2d[0, i] = cv_grid_2d[i][0]
            grid_2d[1, i] = cv_grid_2d[i][1]

        world2hand = np.linalg.inv(handpose)
        pattern_2ds.append(grid_2d)
        world2hands.append(world2hand)

        if visualize:
            cv2.drawChessboardCorners(
                undistorted, (pattern_rows, pattern_cols), cv_grid_2d, found
            )
            small = cv2.resize(undistorted, (undistorted.shape[1] // 4, undistorted.shape[0] // 4))
            cv2.imshow('undistorted', small)
            cv2.waitKey(100)

    if visualize:
        cv2.destroyAllWindows()

    print(f'num_images: {len(pattern_2ds)}')

    return {
        'camera_matrix': camera_matrix,
        'distortion': distortion,
        'pattern_3d': pattern_3d,
        'pattern_2ds': pattern_2ds,
        'world2hands': world2hands,
    }


def _get_aruco_dict(dict_name):
    import cv2
    dict_map = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
        'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
        'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    if dict_name not in dict_map:
        raise ValueError(f'Unknown ArUco dictionary: {dict_name}. Available: {list(dict_map.keys())}')
    return cv2.aruco.getPredefinedDictionary(dict_map[dict_name])


def make_aruco_pattern_3d(marker_size):
    """Generate 3D corner points for a single ArUco marker.

    ArUco corner ordering: TL(0), TR(1), BR(2), BL(3).
    Origin at marker center, marker_size in meters.
    Returns 3x4 numpy array.
    """
    s = marker_size / 2.0
    return np.array([
        [-s, -s, 0],   # corner 0: top-left
        [ s, -s, 0],   # corner 1: top-right
        [ s,  s, 0],   # corner 2: bottom-right
        [-s,  s, 0],   # corner 3: bottom-left
    ]).T


def read_aruco_dataset(dataset_dir, camera_params_file, poses_csv=None,
                       marker_size=0.1, aruco_dict='DICT_5X5_50',
                       pose_unit_trans=0.001, pose_unit_rot_deg=True,
                       visualize=False):
    """Read an ArUco-based calibration dataset.

    Args:
        dataset_dir: Directory with {NNN}_Color.png images.
        camera_params_file: Path to camera_params.yaml (ROS format).
        poses_csv: Path to poses.csv. Each row: tx,ty,tz,rx,ry,rz.
                   If None, reads from dataset_dir/poses.csv.
        marker_size: Physical marker side length in meters (default 0.1 = 100mm).
        aruco_dict: ArUco dictionary name (e.g. 'DICT_5X5_50').
        pose_unit_trans: Multiplier to convert pose translations to meters
                         (0.001 for mm input, 1.0 for meters input).
        pose_unit_rot_deg: True if rotation values are in degrees.
        visualize: Show detected markers.

    Returns dict with keys: camera_matrix, distortion, pattern_3d, pattern_2ds, world2hands.
    """
    import cv2
    from scipy.spatial.transform import Rotation

    camera_matrix, distortion = read_ros_camera_params(camera_params_file)
    pattern_3d = make_aruco_pattern_3d(marker_size)
    aruco_dict_obj = _get_aruco_dict(aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict_obj, detector_params)

    if poses_csv is None:
        poses_csv = os.path.join(dataset_dir, 'poses.csv')
    pose_data = read_matrix_csv(poses_csv)

    # Build lookup: image index -> pose row
    image_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.endswith('.png') and '_Color.png' in f
    ])

    cv_K = np.array(camera_matrix, dtype=np.float64)
    cv_dist = np.array(distortion, dtype=np.float64)

    pattern_2ds = []
    world2hands = []

    for idx, fname in enumerate(image_files):
        if idx >= pose_data.shape[0]:
            print(f'warning: more images ({len(image_files)}) than poses ({pose_data.shape[0]})')
            break

        image_path = os.path.join(dataset_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            print(f'failed to read {fname}')
            continue

        undistorted = cv2.undistort(image, cv_K, cv_dist)
        corners, ids, rejected = detector.detectMarkers(undistorted)

        if ids is None or len(ids) == 0:
            print(f'no ArUco marker detected in {fname}')
            continue

        # Use first detected marker
        marker_corners = corners[0][0]  # shape (4, 2)
        grid_2d = np.zeros((2, 4))
        for i in range(4):
            grid_2d[0, i] = marker_corners[i][0]
            grid_2d[1, i] = marker_corners[i][1]

        # Parse pose: tx, ty, tz, rx, ry, rz
        row = pose_data[idx]
        t = row[:3] * pose_unit_trans
        r = row[3:6]

        handpose = np.eye(4)
        handpose[:3, 3] = t
        handpose[:3, :3] = Rotation.from_euler('xyz', r, degrees=pose_unit_rot_deg).as_matrix()
        world2hand = np.linalg.inv(handpose)

        pattern_2ds.append(grid_2d)
        world2hands.append(world2hand)

        if visualize:
            cv2.aruco.drawDetectedMarkers(undistorted, corners, ids)
            small = cv2.resize(undistorted, (undistorted.shape[1] // 2, undistorted.shape[0] // 2))
            cv2.imshow('aruco', small)
            cv2.waitKey(200)

    if visualize:
        cv2.destroyAllWindows()

    print(f'num_images: {len(pattern_2ds)} / {len(image_files)}')

    return {
        'camera_matrix': camera_matrix,
        'distortion': distortion,
        'pattern_3d': pattern_3d,
        'pattern_2ds': pattern_2ds,
        'world2hands': world2hands,
    }


def _get_charuco_dict(dict_name):
    """Get OpenCV Charuco dictionary by name.

    Same as _get_aruco_dict but explicitly named for Charuco usage.
    """
    import cv2
    dict_map = {
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
    if dict_name not in dict_map:
        raise ValueError(f'Unknown ArUco dictionary: {dict_name}. Available: {list(dict_map.keys())}')
    return cv2.aruco.getPredefinedDictionary(dict_map[dict_name])


def make_charuco_pattern_3d(squares_x, squares_y, square_length):
    """Generate 3D corner points for a ChArUco board.

    Matches OpenCV CharucoBoard::getChessboardCorners() ordering:
    - Row-major: X increases first, then Y
    - Starts from first internal corner at (square_length, square_length)
    - Number of corners = (squares_x - 1) * (squares_y - 1)

    Args:
        squares_x: Number of squares in X direction (columns).
        squares_y: Number of squares in Y direction (rows).
        square_length: Square side length in meters.

    Returns:
        3xN numpy array of corner positions in board frame (Z=0 plane).
    """
    corners_x = squares_x - 1
    corners_y = squares_y - 1
    n_corners = corners_x * corners_y
    pattern_3d = np.zeros((3, n_corners))

    idx = 0
    for row in range(corners_y):
        for col in range(corners_x):
            pattern_3d[0, idx] = (col + 1) * square_length
            pattern_3d[1, idx] = (row + 1) * square_length
            pattern_3d[2, idx] = 0.0
            idx += 1

    return pattern_3d


def get_charuco_board(squares_x, squares_y, square_length, marker_length, aruco_dict):
    """Create a ChArUco board object.

    Args:
        squares_x: Number of squares in X direction.
        squares_y: Number of squares in Y direction.
        square_length: Chessboard square side length in meters.
        marker_length: ArUco marker side length in meters.
        aruco_dict: ArUco dictionary name (e.g., 'DICT_5X5_100') or cv2.aruco.Dictionary object.

    Returns:
        cv2.aruco.CharucoBoard object.
    """
    import cv2

    if isinstance(aruco_dict, str):
        aruco_dict_obj = _get_charuco_dict(aruco_dict)
    else:
        aruco_dict_obj = aruco_dict

    # OpenCV 4.7+ uses CharucoBoard_create as constructor function
    # OpenCV 4.8+ also supports creating via CharucoBoard() constructor
    try:
        # Try newer API first (OpenCV 4.8+)
        board = cv2.aruco.CharucoBoard(
            size=(squares_x, squares_y),
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=aruco_dict_obj
        )
    except TypeError:
        # Fallback to older API (OpenCV 4.6-4.7)
        board = cv2.aruco.CharucoBoard_create(
            squaresX=squares_x,
            squaresY=squares_y,
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=aruco_dict_obj
        )

    return board


def read_charuco_dataset(dataset_dir, camera_params_file, poses_csv=None,
                         squares_x=8, squares_y=11, square_length=0.014,
                         marker_length=0.010, aruco_dict='DICT_5X5_100',
                         pose_unit_trans=0.001, pose_unit_rot_deg=True,
                         visualize=False, save_detection_dir=None,
                         use_chessboard_fallback=True):
    """Read a ChArUco board-based calibration dataset.

    Args:
        dataset_dir: Directory with {NNN}_Color.png images.
        camera_params_file: Path to camera_params.yaml (ROS format).
        poses_csv: Path to poses.csv. Each row: tx,ty,tz,rx,ry,rz.
                   If None, reads from dataset_dir/poses.csv.
        squares_x: Number of squares in X direction (default 8).
        squares_y: Number of squares in Y direction (default 11).
        square_length: Chessboard square side length in meters (default 0.014 = 14mm).
        marker_length: ArUco marker side length in meters (default 0.010 = 10mm).
        aruco_dict: ArUco dictionary name (default 'DICT_5X5_100').
        pose_unit_trans: Multiplier to convert pose translations to meters
                         (0.001 for mm input, 1.0 for meters input).
        pose_unit_rot_deg: True if rotation values are in degrees.
        visualize: Show detected boards.
        save_detection_dir: If provided, save detection visualization images to this directory.
                            Creates detection_{NNN}.png (RGB format) with drawn markers (green) and corners (red).
        use_chessboard_fallback: If True, use traditional cv2.findChessboardCorners when
                                  CharucoDetector fails to detect corners (OpenCV 4.13 compatibility).

    Returns dict with keys: camera_matrix, distortion, pattern_3d, pattern_2ds, world2hands.
    """
    import cv2
    from scipy.spatial.transform import Rotation

    camera_matrix, distortion = read_ros_camera_params(camera_params_file)
    pattern_3d = make_charuco_pattern_3d(squares_x, squares_y, square_length)
    board = get_charuco_board(squares_x, squares_y, square_length, marker_length, aruco_dict)

    expected_corners = (squares_x - 1) * (squares_y - 1)
    min_corners = max(4, expected_corners // 4)
    chessboard_size = (squares_x - 1, squares_y - 1)  # Internal corners for fallback

    if poses_csv is None:
        poses_csv = os.path.join(dataset_dir, 'poses.csv')
    pose_data = read_matrix_csv(poses_csv)

    image_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.endswith('.png') and '_Color.png' in f
    ])

    cv_K = np.array(camera_matrix, dtype=np.float64)
    cv_dist = np.array(distortion, dtype=np.float64)

    if save_detection_dir:
        os.makedirs(save_detection_dir, exist_ok=True)

    pattern_2ds = []
    world2hands = []
    detection_data = []

    # Subpixel refinement criteria for chessboard fallback
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Try to use CharucoDetector (OpenCV 4.7+) or fall back to older API
    use_new_api = False
    try:
        charuco_params = cv2.aruco.CharucoParameters()
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
        use_new_api = True
    except AttributeError:
        aruco_dict_obj = _get_charuco_dict(aruco_dict) if isinstance(aruco_dict, str) else aruco_dict
        detector_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict_obj, detector_params)
        use_new_api = False

    for idx, fname in enumerate(image_files):
        if idx >= pose_data.shape[0]:
            print(f'warning: more images ({len(image_files)}) than poses ({pose_data.shape[0]})')
            break

        image_path = os.path.join(dataset_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            print(f'failed to read {fname}')
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        undistorted = cv2.undistort(gray, cv_K, cv_dist)

        charuco_corners = None
        charuco_ids = None
        marker_corners = None
        marker_ids = None
        use_chessboard = False

        # Try CharucoDetector first
        if use_new_api:
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(undistorted)
        else:
            marker_corners, marker_ids, _ = aruco_detector.detectMarkers(undistorted)
            if marker_ids is not None and len(marker_ids) >= 4:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, undistorted, board
                )

        # Check Charuco result, fallback to traditional chessboard detection
        if charuco_corners is None or len(charuco_corners) < min_corners:
            if use_chessboard_fallback:
                if marker_corners is None or len(marker_corners) == 0:
                    aruco_detector_fallback = cv2.aruco.ArucoDetector(
                        _get_charuco_dict(aruco_dict) if isinstance(aruco_dict, str) else aruco_dict,
                        cv2.aruco.DetectorParameters()
                    )
                    marker_corners, marker_ids, _ = aruco_detector_fallback.detectMarkers(undistorted)
                
                ret, corners = cv2.findChessboardCorners(
                    undistorted, chessboard_size,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                if ret:
                    corners_refined = cv2.cornerSubPix(
                        undistorted, corners, (5, 5), (-1, -1), subpix_criteria
                    )
                    
                    # Use marker IDs to avoid chessboard 180deg flip ambiguity
                    need_flip = False
                    if marker_ids is not None and len(marker_ids) >= 4:
                        marker_centers = np.array([c.mean(axis=1)[0] for c in marker_corners])
                        sorted_by_y = np.argsort(marker_centers[:, 1])
                        top_markers_idx = sorted_by_y[:8]
                        top_marker_ids = marker_ids.flatten()[top_markers_idx]
                        sorted_by_x = np.argsort(marker_centers[top_markers_idx, 0])
                        
                        if len(top_marker_ids) >= 2:
                            id_diff = np.diff(top_marker_ids[sorted_by_x])
                            if np.sum(id_diff) < 0:
                                need_flip = True
                    
                    if need_flip:
                        corners_refined = corners_refined[::-1]
                    
                    # OpenCV findChessboardCorners uses column-major order for tall boards.
                    # Detection: det_idx = scan * cols + pos
                    # where cols=patternSize.cols (corners per scan), pos=position within scan.
                    # Physical: scan determines X, pos determines Y.
                    # Pattern_3d: row determines Y, col determines X.
                    # Conversion: row=pos, col=scan, pattern_idx=row*cols_x+col
                    cols_x = squares_x - 1  # cols in pattern_3d
                    cols_per_scan = chessboard_size[0]  # corners per scan from patternSize
                    corners_row_major = np.zeros_like(corners_refined)
                    for det_idx in range(len(corners_refined)):
                        scan = det_idx // cols_per_scan  # horizontal scan index
                        pos = det_idx % cols_per_scan    # vertical position within scan
                        # Map to pattern_3d row-major index
                        # pattern_3d: X=(col+1)*len, Y=(row+1)*len
                        # Detection: X=(scan+1)*len, Y=(pos+1)*len
                        # So: col=scan, row=pos
                        pattern_idx = pos * cols_x + scan
                        if pattern_idx < len(corners_row_major):
                            corners_row_major[pattern_idx] = corners_refined[det_idx]
                    
                    charuco_corners = corners_row_major.reshape(-1, 1, 2)
                    charuco_ids = np.arange(expected_corners).reshape(-1, 1)
                    use_chessboard = True
                else:
                    n_found = 0 if charuco_corners is None else len(charuco_corners)
                    print(f'insufficient corners in {fname} (charuco={n_found}, chessboard=not found)')
                    continue
            else:
                n_found = 0 if charuco_corners is None else len(charuco_corners)
                print(f'insufficient ChArUco corners in {fname} (found {n_found}, need >= {min_corners})')
                continue

        n_corners = len(charuco_corners)

        grid_2d = np.full((2, expected_corners), np.nan)
        for i, corner_id in enumerate(charuco_ids.flatten()):
            if corner_id < expected_corners:
                grid_2d[0, corner_id] = charuco_corners[i][0][0]
                grid_2d[1, corner_id] = charuco_corners[i][0][1]

        row = pose_data[idx]
        t = row[:3] * pose_unit_trans
        r = row[3:6]

        handpose = np.eye(4)
        handpose[:3, 3] = t
        handpose[:3, :3] = Rotation.from_euler('xyz', r, degrees=pose_unit_rot_deg).as_matrix()
        world2hand = np.linalg.inv(handpose)

        pattern_2ds.append((grid_2d, charuco_ids.flatten(), n_corners))
        world2hands.append(world2hand)
        # Convert grayscale to BGR for RGB visualization output
        vis_img_color = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
        detection_data.append((vis_img_color, marker_corners, marker_ids, charuco_corners, charuco_ids, use_chessboard))

        if visualize:
            if marker_corners is not None and len(marker_corners) > 0:
                cv2.aruco.drawDetectedMarkers(undistorted, marker_corners, marker_ids)
            if use_chessboard:
                cv2.drawChessboardCorners(undistorted, chessboard_size, charuco_corners, True)
            elif charuco_corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(undistorted, charuco_corners, charuco_ids)
            small = cv2.resize(undistorted, (undistorted.shape[1] // 2, undistorted.shape[0] // 2))
            cv2.imshow('charuco', small)
            cv2.waitKey(200)

    if visualize:
        cv2.destroyAllWindows()

    if save_detection_dir and len(detection_data) > 0:
        chessboard_size = (squares_x - 1, squares_y - 1)
        for det_idx, (vis_img, mc, mi, cc, ci, used_chess) in enumerate(detection_data):
            if mc is not None and len(mc) > 0:
                cv2.aruco.drawDetectedMarkers(vis_img, mc, mi, borderColor=(0, 255, 0))
            if cc is not None and len(cc) > 0:
                if used_chess:
                    cv2.drawChessboardCorners(vis_img, chessboard_size, cc, True)
                else:
                    cv2.aruco.drawDetectedCornersCharuco(vis_img, cc, ci, cornerColor=(0, 0, 255))
            out_path = os.path.join(save_detection_dir, f'detection_{det_idx:03d}.png')
            cv2.imwrite(out_path, vis_img)
        print(f'detection images saved: {save_detection_dir}/ ({len(detection_data)} images)')

    final_pattern_2ds = []
    final_world2hands = []
    final_pattern_3d = pattern_3d

    for i, (grid_2d, corner_ids, n_corners) in enumerate(pattern_2ds):
        detected_mask = ~np.isnan(grid_2d[0, :])
        detected_indices = np.where(detected_mask)[0]
        reduced_pattern_2d = grid_2d[:, detected_mask]
        final_pattern_2ds.append(reduced_pattern_2d)
        final_world2hands.append(world2hands[i])

    if len(final_pattern_2ds) > 0:
        first_grid_2d, first_corner_ids, _ = pattern_2ds[0]
        first_detected_mask = ~np.isnan(first_grid_2d[0, :])
        first_detected_indices = np.where(first_detected_mask)[0]
        final_pattern_3d = pattern_3d[:, first_detected_indices]

        common_pattern_2ds = []
        common_world2hands = []
        for i, (grid_2d, corner_ids, n_corners) in enumerate(pattern_2ds):
            detected_mask = ~np.isnan(grid_2d[0, :])
            detected_indices = set(np.where(detected_mask)[0])
            common_indices = sorted(set(first_detected_indices) & detected_indices)

            if len(common_indices) >= min_corners:
                common_mask = np.array([idx in common_indices for idx in range(expected_corners)])
                common_pattern_2d = grid_2d[:, common_mask]
                common_pattern_2ds.append(common_pattern_2d)
                common_world2hands.append(world2hands[i])

        final_pattern_2ds = common_pattern_2ds
        final_world2hands = common_world2hands

    print(f'num_images: {len(final_pattern_2ds)} / {len(image_files)}')
    if len(final_pattern_2ds) > 0:
        print(f'detected_corners_per_image: {final_pattern_2ds[0].shape[1]} / {expected_corners}')

    return {
        'camera_matrix': camera_matrix,
        'distortion': distortion,
        'pattern_3d': final_pattern_3d,
        'pattern_2ds': final_pattern_2ds,
        'world2hands': final_world2hands,
    }


# =============================================================================
# Spec-compliant function aliases (for new calibrator API)
# =============================================================================

def load_camera_params_yaml(filepath: str):
    """Load camera parameters from ROS YAML file. Spec-compliant alias."""
    return read_ros_camera_params(filepath)


def load_poses_csv(filepath: str, trans_unit: float = 0.001, rot_deg: bool = True,
                   rot_order: str = "xyz", invert: bool = False):
    """Load poses from CSV file. Each row: tx,ty,tz,rx,ry,rz.
    
    Args:
        filepath: CSV file path
        trans_unit: Multiplier to convert translation to meters (0.001 for mm)
        rot_deg: True if rotation values are in degrees
        rot_order: Euler rotation order passed to scipy Rotation.from_euler
        invert: True if CSV rows must be inverted before use
    
    Returns:
        List of 4x4 numpy arrays (flange-to-world transforms by default)
    """
    from scipy.spatial.transform import Rotation
    poses = []
    data = read_matrix_csv(filepath)
    for row in data:
        t = row[:3] * trans_unit
        r = row[3:6]
        pose = np.eye(4)
        pose[:3, 3] = t
        pose[:3, :3] = Rotation.from_euler(rot_order, r, degrees=rot_deg).as_matrix()
        poses.append(np.linalg.inv(pose) if invert else pose)
    return poses


def _matrix_block(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {m.shape}")
    return {
        "rows": 4,
        "cols": 4,
        "data": [float(v) for v in m.reshape(-1)],
    }


def _plain_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_value(v) for v in value]
    return value


def save_calibration_yaml(filepath: str, *args, setup=None, transforms=None,
                          metrics=None, num_images=None, num_images_used=None,
                          filtered_images=None, depth_used=False):
    """Save calibration results using the new explicit schema.

    Legacy positional calls are accepted for callers that still pass
    ``(T_C2F, T_O2W, metrics)``.
    """
    if transforms is None and len(args) >= 2:
        setup = setup or "eye-in-hand"
        transforms = {"T_C2F": args[0], "T_O2W": args[1]}
        if metrics is None and len(args) >= 3:
            metrics = args[2]

    if setup is None or transforms is None:
        raise ValueError("setup and transforms are required")

    setup_value = setup.value if hasattr(setup, "value") else str(setup)
    data = {
        "setup": setup_value,
        "transforms": {name: _matrix_block(T) for name, T in transforms.items()},
        "metrics": _plain_value(metrics or {}),
        "num_images": int(num_images) if num_images is not None else None,
        "num_images_used": int(num_images_used) if num_images_used is not None else None,
        "filtered_images": list(filtered_images or []),
        "depth_used": bool(depth_used),
    }

    import yaml
    with open(filepath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def find_image_files(image_dir: str):
    """Find calibration images in directory. Returns sorted list of paths."""
    files = []
    for f in sorted(os.listdir(image_dir)):
        if f.endswith('.png') and '_Color.' in f:
            files.append(os.path.join(image_dir, f))
    return files
