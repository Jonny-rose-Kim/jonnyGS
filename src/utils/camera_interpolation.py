"""
Camera interpolation utilities for 3D Gaussian Splatting.

Interpolates camera position and rotation between two cameras.
"""

import torch
import numpy as np
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def quaternion_slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If negative, negate one quaternion to take shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # SLERP
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)

    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

    return R


def interpolate_cameras(cam1, cam2, t=0.5):
    """
    Interpolate between two cameras.

    Args:
        cam1: First camera
        cam2: Second camera
        t: Interpolation parameter [0, 1], where 0 = cam1, 1 = cam2, 0.5 = midpoint

    Returns:
        Interpolated camera object
    """
    # Extract camera positions (in world coordinates)
    if isinstance(cam1.camera_center, torch.Tensor):
        pos1 = cam1.camera_center.cpu().numpy()
        pos2 = cam2.camera_center.cpu().numpy()
    else:
        pos1 = np.array(cam1.camera_center)
        pos2 = np.array(cam2.camera_center)

    # Linear interpolation of positions
    pos_interp = (1 - t) * pos1 + t * pos2

    # Extract rotation matrices from R (world to camera)
    if isinstance(cam1.R, torch.Tensor):
        R1 = cam1.R.T.cpu().numpy()  # Transpose to get camera to world
        R2 = cam2.R.T.cpu().numpy()
    else:
        R1 = cam1.R.T  # Already numpy, just transpose
        R2 = cam2.R.T

    # Convert to quaternions
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)

    # SLERP interpolation
    q_interp = quaternion_slerp(q1, q2, t)

    # Convert back to rotation matrix
    R_interp = quaternion_to_rotation_matrix(q_interp)
    R_interp = R_interp.T  # Transpose back to world to camera

    # Create a shallow copy of cam1 and update R, T
    # This avoids the complex Camera constructor
    import copy
    interp_cam = copy.copy(cam1)

    # Calculate new T
    T_interp = -R_interp @ pos_interp  # T = -R @ camera_center
    camera_center_interp = -np.dot(R_interp.T, T_interp)

    # Update interpolated parameters as torch tensors (required by renderer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interp_cam.R = R_interp  # Keep as numpy for internal use
    interp_cam.T = T_interp  # Keep as numpy for internal use
    interp_cam.camera_center = torch.tensor(camera_center_interp, dtype=torch.float32, device=device)

    interp_cam.uid = -1
    interp_cam.colmap_id = -1
    interp_cam.image_name = f"interpolated_{cam1.image_name}_{cam2.image_name}_{t:.2f}"

    # Update world_view_transform and other computed matrices
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    interp_cam.world_view_transform = torch.tensor(getWorld2View2(R_interp, T_interp), dtype=torch.float32).transpose(0, 1).cuda()
    interp_cam.projection_matrix = getProjectionMatrix(
        znear=interp_cam.znear,
        zfar=interp_cam.zfar,
        fovX=interp_cam.FoVx,
        fovY=interp_cam.FoVy
    ).transpose(0, 1).cuda()
    interp_cam.full_proj_transform = (interp_cam.world_view_transform.unsqueeze(0).bmm(
        interp_cam.projection_matrix.unsqueeze(0))).squeeze(0)

    return interp_cam


def find_adjacent_camera_pairs(cameras, max_pairs=None, spatial_threshold=None):
    """
    Find pairs of adjacent cameras.

    Args:
        cameras: List of camera objects
        max_pairs: Maximum number of pairs to return (None = all)
        spatial_threshold: Maximum distance between cameras to be considered adjacent

    Returns:
        List of (cam1, cam2, distance) tuples
    """
    pairs = []

    for i in range(len(cameras) - 1):
        cam1 = cameras[i]
        cam2 = cameras[i + 1]

        # Compute distance
        pos1 = cam1.camera_center.cpu().numpy()
        pos2 = cam2.camera_center.cpu().numpy()
        dist = np.linalg.norm(pos1 - pos2)

        # Filter by spatial threshold if provided
        if spatial_threshold is None or dist < spatial_threshold:
            pairs.append((cam1, cam2, dist))

    # Sort by distance (closest pairs first)
    pairs.sort(key=lambda x: x[2])

    # Limit number of pairs if requested
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    return pairs


def create_interpolated_views(cameras, num_interpolations=1, max_pairs=None):
    """
    Create interpolated views between adjacent cameras.

    Args:
        cameras: List of camera objects
        num_interpolations: Number of interpolated views per pair (e.g., 1 = midpoint only)
        max_pairs: Maximum number of camera pairs to interpolate

    Returns:
        List of (interp_cam, cam1, cam2, t) tuples
    """
    # Find adjacent pairs
    pairs = find_adjacent_camera_pairs(cameras, max_pairs=max_pairs)

    print(f"Found {len(pairs)} adjacent camera pairs")

    interpolated_views = []

    for cam1, cam2, dist in pairs:
        # Create interpolated views
        for i in range(num_interpolations):
            if num_interpolations == 1:
                t = 0.5  # Midpoint
            else:
                t = (i + 1) / (num_interpolations + 1)

            interp_cam = interpolate_cameras(cam1, cam2, t)
            interpolated_views.append((interp_cam, cam1, cam2, t))

    print(f"Created {len(interpolated_views)} interpolated views")

    return interpolated_views


def compute_look_at_rotation(camera_pos, target_pos, up=None):
    """
    Compute rotation matrix for camera looking at target.

    Args:
        camera_pos: Camera position [3]
        target_pos: Target position to look at [3]
        up: Up vector (if None, uses [0, -1, 0] which works for many scenes)

    Returns:
        R: 3x3 rotation matrix (world to camera)
    """
    if up is None:
        up = np.array([0, -1, 0])  # Y-down is common in many coordinate systems

    # Forward direction (camera to target)
    forward = target_pos - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # Right direction
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # forward is parallel to up, use alternative up
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right = right / (right_norm + 1e-8)

    # Recompute up to be orthogonal
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    # Camera convention: -Z is forward (looking direction)
    # R transforms world to camera coordinates
    R = np.array([
        right,        # X-axis (right)
        -up,          # Y-axis (down in image)
        -forward      # Z-axis (backward, opposite of viewing direction)
    ])

    return R


def estimate_scene_up_vector(cameras):
    """
    Estimate the scene's up vector from camera orientations.

    Args:
        cameras: List of camera objects

    Returns:
        up: Estimated up vector [3]
    """
    up_vectors = []
    for cam in cameras:
        R = cam.R if isinstance(cam.R, np.ndarray) else cam.R.cpu().numpy()
        # Camera's up direction is -R[1,:] (negative Y-axis in camera space)
        up = -R[1, :]
        up_vectors.append(up)

    up_vectors = np.array(up_vectors)
    avg_up = up_vectors.mean(axis=0)
    avg_up = avg_up / (np.linalg.norm(avg_up) + 1e-8)

    return avg_up


def interpolate_cameras_look_at(cam1, cam2, scene_center, t=0.5, arc_interpolation=True, world_up=None):
    """
    Interpolate camera position while always looking at scene center.
    This is better for detecting floating noise as it keeps the scene in view.

    Args:
        cam1: First camera
        cam2: Second camera
        scene_center: 3D point to look at [3] (e.g., gaussian centroid)
        t: Interpolation parameter [0, 1]
        arc_interpolation: If True, interpolate along arc around scene center
                          If False, linear interpolation of position
        world_up: Scene's world up vector [3] (should be pre-computed from all cameras)

    Returns:
        Interpolated camera object looking at scene_center
    """
    import copy

    # Extract camera positions
    if isinstance(cam1.camera_center, torch.Tensor):
        pos1 = cam1.camera_center.cpu().numpy()
        pos2 = cam2.camera_center.cpu().numpy()
    else:
        pos1 = np.array(cam1.camera_center)
        pos2 = np.array(cam2.camera_center)

    scene_center = np.array(scene_center)

    # Use provided world_up or default to Y-down
    if world_up is None:
        world_up = np.array([0, -1, 0])
    else:
        world_up = np.array(world_up)

    if arc_interpolation:
        # Arc interpolation: move along a circular arc around scene center
        # This creates a more natural orbit-like camera motion

        # Vectors from scene center to cameras
        v1 = pos1 - scene_center
        v2 = pos2 - scene_center

        r1 = np.linalg.norm(v1)
        r2 = np.linalg.norm(v2)

        if r1 < 1e-6 or r2 < 1e-6:
            # Fallback to linear if camera is at scene center
            pos_interp = (1 - t) * pos1 + t * pos2
        else:
            # Normalize directions
            d1 = v1 / r1
            d2 = v2 / r2

            # Interpolate radius
            r_interp = (1 - t) * r1 + t * r2

            # SLERP for direction (on unit sphere)
            # Convert to quaternion representation for interpolation
            dot = np.dot(d1, d2)
            dot = np.clip(dot, -1.0, 1.0)

            if abs(dot) > 0.9999:
                # Almost parallel, use linear interpolation
                d_interp = (1 - t) * d1 + t * d2
            else:
                theta = np.arccos(dot)
                sin_theta = np.sin(theta)
                w1 = np.sin((1 - t) * theta) / sin_theta
                w2 = np.sin(t * theta) / sin_theta
                d_interp = w1 * d1 + w2 * d2

            d_interp = d_interp / (np.linalg.norm(d_interp) + 1e-8)

            # Final position
            pos_interp = scene_center + r_interp * d_interp
    else:
        # Linear interpolation of position
        pos_interp = (1 - t) * pos1 + t * pos2

    # Compute look-at rotation toward scene center
    R_interp = compute_look_at_rotation(pos_interp, scene_center, up=world_up)

    # Create camera copy
    interp_cam = copy.copy(cam1)

    # Calculate new T
    T_interp = -R_interp @ pos_interp

    # Update parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interp_cam.R = R_interp
    interp_cam.T = T_interp
    interp_cam.camera_center = torch.tensor(pos_interp, dtype=torch.float32, device=device)

    interp_cam.uid = -1
    interp_cam.colmap_id = -1
    interp_cam.image_name = f"lookat_{cam1.image_name}_{cam2.image_name}_{t:.2f}"

    # Update transforms
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    interp_cam.world_view_transform = torch.tensor(
        getWorld2View2(R_interp, T_interp), dtype=torch.float32
    ).transpose(0, 1).cuda()
    interp_cam.projection_matrix = getProjectionMatrix(
        znear=interp_cam.znear,
        zfar=interp_cam.zfar,
        fovX=interp_cam.FoVx,
        fovY=interp_cam.FoVy
    ).transpose(0, 1).cuda()
    interp_cam.full_proj_transform = (
        interp_cam.world_view_transform.unsqueeze(0).bmm(
            interp_cam.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)

    return interp_cam


def compute_scene_center(gaussians):
    """
    Compute scene center from gaussian positions.
    Uses opacity-weighted centroid for better center estimation.

    Args:
        gaussians: GaussianModel object

    Returns:
        scene_center: [3] numpy array
    """
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().squeeze(-1)

    # Use opacity as weight (more opaque = more important)
    weights = torch.sigmoid(opacity)  # Normalize to [0, 1]
    weights = weights / (weights.sum() + 1e-8)

    # Weighted centroid
    center = (xyz * weights.unsqueeze(-1)).sum(dim=0)

    return center.cpu().numpy()


def compute_scene_center_robust(gaussians, percentile=90):
    """
    Compute robust scene center by excluding outliers.
    Useful when there are floaters far from the main scene.

    Args:
        gaussians: GaussianModel object
        percentile: Keep only gaussians within this percentile of distances

    Returns:
        scene_center: [3] numpy array
    """
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)

    # Initial center estimate (simple mean)
    initial_center = xyz.mean(axis=0)

    # Compute distances from center
    distances = np.linalg.norm(xyz - initial_center, axis=1)

    # Keep only points within percentile
    threshold = np.percentile(distances, percentile)
    mask = distances <= threshold

    # Opacity-weighted center of remaining points
    weights = 1.0 / (1.0 + np.exp(-opacity[mask]))  # sigmoid
    weights = weights / (weights.sum() + 1e-8)

    center = (xyz[mask] * weights[:, np.newaxis]).sum(axis=0)

    return center


def compute_scene_center_from_cameras(cameras):
    """
    Compute scene center from camera positions.
    For 360-degree scenes, the camera centroid is typically the object of interest.

    Args:
        cameras: List of camera objects

    Returns:
        scene_center: [3] numpy array
    """
    positions = []
    for cam in cameras:
        if isinstance(cam.camera_center, torch.Tensor):
            pos = cam.camera_center.cpu().numpy()
        else:
            pos = np.array(cam.camera_center)
        positions.append(pos)

    positions = np.array(positions)
    return positions.mean(axis=0)


def compute_scene_center_from_gaze(cameras, distance=2.0):
    """
    Compute scene center from camera gaze directions.
    Takes the camera centroid and moves along average gaze direction.

    Args:
        cameras: List of camera objects
        distance: Distance to move along gaze direction (meters)

    Returns:
        scene_center: [3] numpy array
    """
    positions = []
    look_dirs = []

    for cam in cameras:
        if isinstance(cam.camera_center, torch.Tensor):
            pos = cam.camera_center.cpu().numpy()
        else:
            pos = np.array(cam.camera_center)
        positions.append(pos)

        # Extract look direction from rotation matrix
        R = cam.R if isinstance(cam.R, np.ndarray) else cam.R.cpu().numpy()
        # Camera looks along -Z in camera space
        forward = -R[2, :]
        look_dirs.append(forward)

    positions = np.array(positions)
    look_dirs = np.array(look_dirs)

    cam_center = positions.mean(axis=0)
    avg_look = look_dirs.mean(axis=0)
    avg_look = avg_look / (np.linalg.norm(avg_look) + 1e-8)

    # Scene center = camera center + distance along average gaze
    scene_center = cam_center + avg_look * distance

    return scene_center


def compute_gaze_intersection(cam1, cam2):
    """
    두 카메라의 시선(gaze) 방향의 교차점 또는 최근접점을 계산합니다.

    두 3D 직선이 정확히 교차하지 않을 수 있으므로,
    두 직선의 최근접점(closest points)의 중점을 반환합니다.

    Args:
        cam1: 첫 번째 카메라
        cam2: 두 번째 카메라

    Returns:
        intersection: 교차점/최근접점의 중점 [3] numpy array
        distance: 두 직선 사이의 최소 거리 (교차하면 0에 가까움)
    """
    # 카메라 위치 추출
    if isinstance(cam1.camera_center, torch.Tensor):
        P1 = cam1.camera_center.cpu().numpy()
        P2 = cam2.camera_center.cpu().numpy()
    else:
        P1 = np.array(cam1.camera_center)
        P2 = np.array(cam2.camera_center)

    # 시선 방향 추출 (카메라는 -Z 방향을 바라봄)
    R1 = cam1.R if isinstance(cam1.R, np.ndarray) else cam1.R.cpu().numpy()
    R2 = cam2.R if isinstance(cam2.R, np.ndarray) else cam2.R.cpu().numpy()

    D1 = -R1[2, :]  # -Z direction in world space
    D2 = -R2[2, :]  # -Z direction in world space

    # Normalize
    D1 = D1 / (np.linalg.norm(D1) + 1e-8)
    D2 = D2 / (np.linalg.norm(D2) + 1e-8)

    # 두 직선의 최근접점 계산
    # Ray1: P1 + t1 * D1
    # Ray2: P2 + t2 * D2
    # 최근접점을 찾기 위해 선형 시스템 풀기:
    # (P1 + t1*D1 - P2 - t2*D2) · D1 = 0
    # (P1 + t1*D1 - P2 - t2*D2) · D2 = 0

    # w = P1 - P2
    w = P1 - P2

    a = np.dot(D1, D1)  # always >= 0
    b = np.dot(D1, D2)
    c = np.dot(D2, D2)  # always >= 0
    d = np.dot(D1, w)
    e = np.dot(D2, w)

    denom = a * c - b * b

    if abs(denom) < 1e-8:
        # 두 직선이 평행함 - 중점 방향으로 적당한 거리에 교차점 설정
        t1 = 0.0
        t2 = d / b if abs(b) > 1e-8 else 0.0
    else:
        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom

    # 카메라 앞쪽만 유효 (t > 0)
    t1 = max(t1, 0.1)  # 최소 거리 확보
    t2 = max(t2, 0.1)

    # 두 직선 위의 최근접점
    closest1 = P1 + t1 * D1
    closest2 = P2 + t2 * D2

    # 최근접점의 중점 = 교차점
    intersection = (closest1 + closest2) / 2.0

    # 두 직선 사이의 거리
    distance = np.linalg.norm(closest1 - closest2)

    return intersection, distance


def interpolate_cameras_pairwise_lookat(cam1, cam2, t=0.5, arc_interpolation=True, world_up=None):
    """
    두 카메라의 시선 교차점을 바라보도록 보간합니다.

    기존 interpolate_cameras_look_at과 달리, 전역 scene_center가 아닌
    각 카메라 쌍의 시선 교차점을 look-at target으로 사용합니다.

    Args:
        cam1: 첫 번째 카메라
        cam2: 두 번째 카메라
        t: 보간 파라미터 [0, 1]
        arc_interpolation: True면 교차점 주위를 도는 arc 경로 사용
        world_up: Scene의 world up vector [3]

    Returns:
        보간된 카메라 객체 (시선 교차점을 바라봄)
    """
    import copy

    # 두 카메라의 시선 교차점 계산
    look_at_target, dist = compute_gaze_intersection(cam1, cam2)

    # 카메라 위치 추출
    if isinstance(cam1.camera_center, torch.Tensor):
        pos1 = cam1.camera_center.cpu().numpy()
        pos2 = cam2.camera_center.cpu().numpy()
    else:
        pos1 = np.array(cam1.camera_center)
        pos2 = np.array(cam2.camera_center)

    # Use provided world_up or default to Y-down
    if world_up is None:
        world_up = np.array([0, -1, 0])
    else:
        world_up = np.array(world_up)

    if arc_interpolation:
        # Arc interpolation: look_at_target 주위를 도는 호 경로
        v1 = pos1 - look_at_target
        v2 = pos2 - look_at_target

        r1 = np.linalg.norm(v1)
        r2 = np.linalg.norm(v2)

        if r1 < 1e-6 or r2 < 1e-6:
            pos_interp = (1 - t) * pos1 + t * pos2
        else:
            d1 = v1 / r1
            d2 = v2 / r2

            r_interp = (1 - t) * r1 + t * r2

            dot = np.dot(d1, d2)
            dot = np.clip(dot, -1.0, 1.0)

            if abs(dot) > 0.9999:
                d_interp = (1 - t) * d1 + t * d2
            else:
                theta = np.arccos(dot)
                sin_theta = np.sin(theta)
                w1 = np.sin((1 - t) * theta) / sin_theta
                w2 = np.sin(t * theta) / sin_theta
                d_interp = w1 * d1 + w2 * d2

            d_interp = d_interp / (np.linalg.norm(d_interp) + 1e-8)
            pos_interp = look_at_target + r_interp * d_interp
    else:
        pos_interp = (1 - t) * pos1 + t * pos2

    # 교차점을 바라보는 rotation 계산
    R_interp = compute_look_at_rotation(pos_interp, look_at_target, up=world_up)

    # 카메라 복사 및 업데이트
    interp_cam = copy.copy(cam1)

    T_interp = -R_interp @ pos_interp

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interp_cam.R = R_interp
    interp_cam.T = T_interp
    interp_cam.camera_center = torch.tensor(pos_interp, dtype=torch.float32, device=device)

    interp_cam.uid = -1
    interp_cam.colmap_id = -1
    interp_cam.image_name = f"pairwise_{cam1.image_name}_{cam2.image_name}_{t:.2f}"

    # Transform 업데이트
    interp_cam.world_view_transform = torch.tensor(
        getWorld2View2(R_interp, T_interp), dtype=torch.float32
    ).transpose(0, 1).cuda()
    interp_cam.projection_matrix = getProjectionMatrix(
        znear=interp_cam.znear,
        zfar=interp_cam.zfar,
        fovX=interp_cam.FoVx,
        fovY=interp_cam.FoVy
    ).transpose(0, 1).cuda()
    interp_cam.full_proj_transform = (
        interp_cam.world_view_transform.unsqueeze(0).bmm(
            interp_cam.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)

    return interp_cam


def verify_view_overlap(rendered_interp, rendered_cam1, rendered_cam2,
                         min_valid_ratio=0.3, min_overlap_score=0.3):
    """
    보간 뷰가 원본 뷰들과 충분히 겹치는지 검증합니다.

    두 가지 기준:
    1. 유효 픽셀 비율 (검정/빈 영역이 너무 많으면 reject)
    2. 원본 뷰와의 구조적 유사도 (cosine similarity)

    Args:
        rendered_interp: 보간 뷰 렌더링 [3, H, W]
        rendered_cam1: cam1 렌더링 [3, H, W]
        rendered_cam2: cam2 렌더링 [3, H, W]
        min_valid_ratio: 최소 유효 픽셀 비율 (default 0.3)
        min_overlap_score: 최소 유사도 (default 0.3)

    Returns:
        is_valid: bool
        info: dict with valid_ratio, sim1, sim2
    """
    import torch.nn.functional as F

    # 1. 유효 픽셀 비율
    brightness = rendered_interp.mean(dim=0)  # [H, W]
    valid_ratio = (brightness > 0.01).float().mean().item()

    if valid_ratio < min_valid_ratio:
        return False, {'valid_ratio': valid_ratio, 'sim1': 0.0, 'sim2': 0.0,
                       'reason': 'low_valid_ratio'}

    # 2. 원본 뷰와의 cosine similarity
    flat_interp = rendered_interp.flatten().unsqueeze(0)
    flat_cam1 = rendered_cam1.flatten().unsqueeze(0)
    flat_cam2 = rendered_cam2.flatten().unsqueeze(0)

    sim1 = F.cosine_similarity(flat_interp, flat_cam1).item()
    sim2 = F.cosine_similarity(flat_interp, flat_cam2).item()

    min_sim = min(sim1, sim2)
    is_valid = min_sim >= min_overlap_score

    info = {
        'valid_ratio': valid_ratio,
        'sim1': sim1,
        'sim2': sim2,
        'reason': 'ok' if is_valid else 'low_overlap'
    }
    return is_valid, info
