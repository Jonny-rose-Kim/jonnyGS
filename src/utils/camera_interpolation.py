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


def compute_look_at_rotation(camera_pos, target_pos, up=np.array([0, 0, 1])):
    """
    Compute rotation matrix for camera looking at target.

    Args:
        camera_pos: Camera position [3]
        target_pos: Target position to look at [3]
        up: Up vector (default: Z-up)

    Returns:
        R: 3x3 rotation matrix (world to camera)
    """
    # Forward direction (camera to target)
    forward = target_pos - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # Right direction
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # forward is parallel to up, use alternative up
        up = np.array([0, 1, 0])
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


def interpolate_cameras_look_at(cam1, cam2, scene_center, t=0.5, arc_interpolation=True):
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
    R_interp = compute_look_at_rotation(pos_interp, scene_center)

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
