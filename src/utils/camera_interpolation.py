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
