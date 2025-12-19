"""
Ray-Gaussian Intersection utilities for Step 7.

2D 노이즈 마스크의 픽셀을 카메라 광선으로 변환하고,
해당 광선과 교차하는 3D 가우시안을 찾습니다.
"""

import torch
import numpy as np
from typing import Set, List, Tuple, Optional
from tqdm import tqdm


def pixel_to_ray(u: torch.Tensor, v: torch.Tensor, camera) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    픽셀 좌표를 카메라 광선으로 변환

    Args:
        u: 픽셀 x 좌표 [N]
        v: 픽셀 y 좌표 [N]
        camera: 카메라 객체 (FoVx, FoVy, image_width, image_height, R, T 필요)

    Returns:
        ray_origins: 광선 시작점 [N, 3] (카메라 위치)
        ray_directions: 광선 방향 [N, 3] (정규화됨)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 카메라 내부 파라미터
    W = camera.image_width
    H = camera.image_height
    fovx = camera.FoVx
    fovy = camera.FoVy

    # focal length 계산
    fx = W / (2 * np.tan(fovx / 2))
    fy = H / (2 * np.tan(fovy / 2))

    # principal point (이미지 중심)
    cx = W / 2
    cy = H / 2

    # 픽셀 좌표를 카메라 좌표계로 변환 (정규화된 좌표)
    # (u, v) -> (x, y, z) in camera frame
    x = (u.float() - cx) / fx
    y = (v.float() - cy) / fy
    z = torch.ones_like(x)

    # 카메라 좌표계에서의 광선 방향 [N, 3]
    ray_dirs_cam = torch.stack([x, y, z], dim=-1).to(device)

    # 카메라 회전 행렬 (world to camera -> camera to world)
    if isinstance(camera.R, torch.Tensor):
        R = camera.R.to(device)
    else:
        R = torch.tensor(camera.R, dtype=torch.float32, device=device)

    # R은 world_to_camera이므로 transpose하여 camera_to_world
    R_c2w = R.T

    # 광선 방향을 월드 좌표계로 변환
    ray_dirs_world = torch.matmul(ray_dirs_cam, R_c2w.T)  # [N, 3]

    # 정규화
    ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True)

    # 광선 시작점 (카메라 위치)
    if isinstance(camera.camera_center, torch.Tensor):
        ray_origin = camera.camera_center.to(device)
    else:
        ray_origin = torch.tensor(camera.camera_center, dtype=torch.float32, device=device)

    # 모든 광선이 동일한 시작점
    ray_origins = ray_origin.unsqueeze(0).expand(len(u), -1)

    return ray_origins, ray_dirs_world


def build_covariance_3d(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    스케일과 회전으로부터 3D 공분산 행렬 계산

    Args:
        scaling: 스케일 [N, 3]
        rotation: 쿼터니언 [N, 4] (w, x, y, z)

    Returns:
        covariance: 공분산 행렬 [N, 3, 3]
    """
    # 쿼터니언 정규화
    rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)

    # 쿼터니언 -> 회전 행렬
    w, x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

    R = torch.zeros((rotation.shape[0], 3, 3), device=rotation.device)
    R[:, 0, 0] = 1 - 2*y*y - 2*z*z
    R[:, 0, 1] = 2*x*y - 2*w*z
    R[:, 0, 2] = 2*x*z + 2*w*y
    R[:, 1, 0] = 2*x*y + 2*w*z
    R[:, 1, 1] = 1 - 2*x*x - 2*z*z
    R[:, 1, 2] = 2*y*z - 2*w*x
    R[:, 2, 0] = 2*x*z - 2*w*y
    R[:, 2, 1] = 2*y*z + 2*w*x
    R[:, 2, 2] = 1 - 2*x*x - 2*y*y

    # 스케일 행렬
    S = torch.diag_embed(scaling)  # [N, 3, 3]

    # 공분산 = R @ S @ S^T @ R^T = R @ S^2 @ R^T
    RS = torch.bmm(R, S)  # [N, 3, 3]
    covariance = torch.bmm(RS, RS.transpose(1, 2))  # [N, 3, 3]

    return covariance


def ray_point_distance(ray_origin: torch.Tensor, ray_dir: torch.Tensor,
                       point: torch.Tensor) -> torch.Tensor:
    """
    광선과 점 사이의 최단 거리 계산

    Args:
        ray_origin: 광선 시작점 [3] 또는 [N, 3]
        ray_dir: 광선 방향 (정규화) [3] 또는 [N, 3]
        point: 점 좌표 [M, 3]

    Returns:
        distances: 거리 [N, M] 또는 [M]
    """
    # Broadcasting을 위해 차원 조정
    if ray_origin.dim() == 1:
        ray_origin = ray_origin.unsqueeze(0)  # [1, 3]
        ray_dir = ray_dir.unsqueeze(0)  # [1, 3]

    # ray_origin: [N, 3], ray_dir: [N, 3], point: [M, 3]
    # 각 광선과 각 점 사이의 거리 계산

    # v = point - ray_origin: [N, M, 3]
    v = point.unsqueeze(0) - ray_origin.unsqueeze(1)  # [N, 1, 3] - [N, M, 3]

    # t = v . ray_dir: [N, M]
    ray_dir_expanded = ray_dir.unsqueeze(1)  # [N, 1, 3]
    t = torch.sum(v * ray_dir_expanded, dim=-1)  # [N, M]

    # 가장 가까운 점: ray_origin + t * ray_dir
    # 거리 = |v - t * ray_dir|
    closest = ray_dir_expanded * t.unsqueeze(-1)  # [N, M, 3]
    diff = v - closest  # [N, M, 3]
    distances = torch.norm(diff, dim=-1)  # [N, M]

    return distances.squeeze(0) if distances.shape[0] == 1 else distances


def ray_ellipsoid_intersection_batch(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    centers: torch.Tensor,
    covariances: torch.Tensor,
    threshold: float = 3.0
) -> torch.Tensor:
    """
    배치 광선-타원체 교차 판정 (마할라노비스 거리 기반)

    광선 위의 모든 점 중 가우시안 중심과의 마할라노비스 거리가
    threshold 이하인 점이 존재하면 교차로 판정

    Args:
        ray_origins: 광선 시작점 [N_rays, 3]
        ray_dirs: 광선 방향 [N_rays, 3]
        centers: 가우시안 중심 [N_gaussians, 3]
        covariances: 공분산 행렬 [N_gaussians, 3, 3]
        threshold: 마할라노비스 거리 임계값 (3.0 = 99.7%)

    Returns:
        intersects: 교차 여부 [N_rays, N_gaussians]
    """
    device = ray_origins.device
    N_rays = ray_origins.shape[0]
    N_gaussians = centers.shape[0]

    # 역공분산 계산 (수치 안정성을 위해 정규화)
    try:
        # 작은 값 추가하여 특이행렬 방지
        eye = torch.eye(3, device=device).unsqueeze(0) * 1e-6
        inv_cov = torch.inverse(covariances + eye)  # [N_gaussians, 3, 3]
    except:
        # 역행렬 계산 실패 시 pseudo-inverse 사용
        inv_cov = torch.linalg.pinv(covariances)

    # 광선 위의 가장 가까운 점에서의 마할라노비스 거리 계산
    # 최적화: 광선 위의 점 p(t) = o + t*d에서
    # 마할라노비스 거리 = (p - c)^T @ inv_cov @ (p - c)
    # 이를 최소화하는 t를 찾고, 그 때의 거리가 threshold 이하인지 확인

    intersects = torch.zeros((N_rays, N_gaussians), dtype=torch.bool, device=device)

    # 메모리 효율을 위해 배치 처리
    batch_size = 1000
    for g_start in range(0, N_gaussians, batch_size):
        g_end = min(g_start + batch_size, N_gaussians)
        batch_centers = centers[g_start:g_end]  # [B, 3]
        batch_inv_cov = inv_cov[g_start:g_end]  # [B, 3, 3]
        B = batch_centers.shape[0]

        for r_start in range(0, N_rays, batch_size):
            r_end = min(r_start + batch_size, N_rays)
            batch_origins = ray_origins[r_start:r_end]  # [R, 3]
            batch_dirs = ray_dirs[r_start:r_end]  # [R, 3]
            R = batch_origins.shape[0]

            # diff = origin - center: [R, B, 3]
            diff = batch_origins.unsqueeze(1) - batch_centers.unsqueeze(0)

            # 각 (ray, gaussian) 쌍에 대해 최소 마할라노비스 거리 계산
            # d^T @ inv_cov @ d, d^T @ inv_cov @ (o-c), (o-c)^T @ inv_cov @ (o-c)
            # 이차 방정식: A*t^2 + B*t + C에서 최소값

            # dirs: [R, 3], inv_cov: [B, 3, 3]
            # A = d^T @ inv_cov @ d: [R, B]
            d_expanded = batch_dirs.unsqueeze(1).unsqueeze(-1)  # [R, 1, 3, 1]
            inv_cov_expanded = batch_inv_cov.unsqueeze(0)  # [1, B, 3, 3]

            # d^T @ inv_cov: [R, B, 1, 3]
            d_T_inv_cov = torch.matmul(d_expanded.transpose(-1, -2), inv_cov_expanded)
            # A = d^T @ inv_cov @ d: [R, B]
            A = torch.matmul(d_T_inv_cov, d_expanded).squeeze(-1).squeeze(-1)

            # diff: [R, B, 3]
            diff_expanded = diff.unsqueeze(-1)  # [R, B, 3, 1]

            # B_coef = 2 * d^T @ inv_cov @ (o-c): [R, B]
            B_coef = 2 * torch.matmul(d_T_inv_cov, diff_expanded).squeeze(-1).squeeze(-1)

            # C = (o-c)^T @ inv_cov @ (o-c): [R, B]
            diff_T_inv_cov = torch.matmul(diff_expanded.transpose(-1, -2), inv_cov_expanded)
            C = torch.matmul(diff_T_inv_cov, diff_expanded).squeeze(-1).squeeze(-1)

            # 최소값을 주는 t: t_min = -B / (2A)
            # 최소 마할라노비스 거리 제곱: C - B^2 / (4A)
            # t >= 0 조건 확인 (광선 앞쪽)

            t_min = -B_coef / (2 * A + 1e-10)
            t_min = torch.clamp(t_min, min=0)  # t >= 0

            # t_min에서의 마할라노비스 거리 제곱
            mahal_sq = A * t_min * t_min + B_coef * t_min + C

            # threshold 이하면 교차
            batch_intersects = mahal_sq <= (threshold ** 2)
            intersects[r_start:r_end, g_start:g_end] = batch_intersects

    return intersects


def find_intersecting_gaussians(
    camera,
    mask: torch.Tensor,
    gaussians,
    threshold: float = 3.0,
    sample_ratio: float = 1.0,
    min_opacity: float = 0.01,
    verbose: bool = True,
    max_rays: int = 500,
    distance_prefilter: float = 5.0
) -> Set[int]:
    """
    마스크 영역의 픽셀들에 대해 교차하는 가우시안 찾기 (메모리 효율적)

    Args:
        camera: 카메라 객체
        mask: 2D 노이즈 마스크 [H, W] 또는 [1, H, W]
        gaussians: GaussianModel 객체
        threshold: 마할라노비스 거리 임계값
        sample_ratio: 마스크 픽셀 샘플링 비율 (1.0 = 전체)
        min_opacity: 최소 불투명도 (낮은 불투명도 가우시안 무시)
        verbose: 진행 상황 출력
        max_rays: 최대 광선 수 (메모리 제한)
        distance_prefilter: 유클리드 거리 사전 필터링 임계값

    Returns:
        noise_candidates: 노이즈 후보 가우시안 인덱스 집합
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 마스크 차원 조정
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    if mask.dim() == 3:  # 여전히 3D면 [1, H, W]
        mask = mask.squeeze(0)
    mask = mask.to(device)

    # 마스크에서 흰색 픽셀 좌표 추출
    white_pixels = torch.where(mask > 0.5)
    v_coords, u_coords = white_pixels[0], white_pixels[1]

    if len(u_coords) == 0:
        if verbose:
            print("  마스크에 노이즈 픽셀 없음")
        return set()

    # 샘플링 - 더 공격적으로
    n_pixels = len(u_coords)
    n_samples = min(max_rays, max(1, int(n_pixels * sample_ratio)))

    if n_samples < n_pixels:
        indices = torch.randperm(n_pixels, device=device)[:n_samples]
        u_coords = u_coords[indices]
        v_coords = v_coords[indices]

    if verbose:
        print(f"  노이즈 픽셀: {n_pixels} → 샘플링: {len(u_coords)}")

    # 픽셀 -> 광선 변환
    ray_origins, ray_dirs = pixel_to_ray(u_coords, v_coords, camera)
    ray_origin = ray_origins[0]  # 모든 광선이 같은 시작점

    # 가우시안 정보 추출
    all_centers = gaussians.get_xyz  # [N, 3]
    all_opacity = gaussians.get_opacity.squeeze(-1)  # [N]

    # 1단계: 불투명도 필터링
    opacity_mask = all_opacity > min_opacity

    # 2단계: Frustum 기반 사전 필터링 (카메라 근처 가우시안만)
    # 카메라에서 가우시안까지의 거리 계산
    camera_pos = ray_origin.to(all_centers.device)
    distances_to_camera = torch.norm(all_centers - camera_pos.unsqueeze(0), dim=1)

    # 거리 기반 필터 (scene extent 기반)
    distance_threshold = torch.quantile(distances_to_camera, 0.95)  # 상위 95% 내
    distance_mask = distances_to_camera < distance_threshold

    # 결합 마스크
    valid_mask = opacity_mask & distance_mask
    valid_indices = torch.where(valid_mask)[0]

    if verbose:
        print(f"  유효 가우시안: {len(valid_indices)} / {len(all_opacity)} (opacity + distance filter)")

    if len(valid_indices) == 0:
        return set()

    # 유효 가우시안만 추출
    centers = all_centers[valid_mask].to(device)
    scaling = gaussians.get_scaling[valid_mask].to(device)
    rotation = gaussians.get_rotation[valid_mask].to(device)

    # 3단계: 광선별로 처리 (메모리 효율)
    noise_candidates = set()
    ray_batch_size = 50  # 한 번에 처리할 광선 수
    gaussian_batch_size = 50000  # 한 번에 처리할 가우시안 수

    n_rays = len(ray_dirs)
    n_gaussians = len(centers)

    for ray_start in range(0, n_rays, ray_batch_size):
        ray_end = min(ray_start + ray_batch_size, n_rays)
        batch_ray_dirs = ray_dirs[ray_start:ray_end]
        batch_ray_origins = ray_origins[ray_start:ray_end]

        for g_start in range(0, n_gaussians, gaussian_batch_size):
            g_end = min(g_start + gaussian_batch_size, n_gaussians)

            batch_centers = centers[g_start:g_end]
            batch_scaling = scaling[g_start:g_end]
            batch_rotation = rotation[g_start:g_end]

            # 공분산 계산
            batch_cov = build_covariance_3d(batch_scaling, batch_rotation)

            # 교차 판정
            try:
                intersects = ray_ellipsoid_intersection_batch(
                    batch_ray_origins, batch_ray_dirs,
                    batch_centers, batch_cov, threshold
                )

                # 교차하는 가우시안 인덱스 추출
                any_intersect = intersects.any(dim=0)
                local_intersecting = torch.where(any_intersect)[0]

                # 전역 인덱스로 변환
                for local_idx in local_intersecting.cpu().numpy():
                    global_idx = valid_indices[g_start + local_idx].item()
                    noise_candidates.add(global_idx)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if verbose:
                        print(f"  메모리 부족, 배치 크기 줄임")
                    continue
                raise

            # 메모리 정리
            del batch_cov
            torch.cuda.empty_cache()

    if verbose:
        print(f"  교차 가우시안 수: {len(noise_candidates)}")

    return noise_candidates


def find_visible_gaussians(camera, gaussians, min_opacity: float = 0.01) -> Set[int]:
    """
    특정 카메라에서 보이는 가우시안 찾기 (frustum culling + opacity)

    Args:
        camera: 카메라 객체
        gaussians: GaussianModel 객체
        min_opacity: 최소 불투명도

    Returns:
        visible_indices: 보이는 가우시안 인덱스 집합
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    centers = gaussians.get_xyz.to(device)
    opacity = gaussians.get_opacity.squeeze(-1).to(device)

    # 불투명도 필터
    opacity_mask = opacity > min_opacity

    # 카메라 매트릭스
    if hasattr(camera, 'full_proj_transform'):
        proj = camera.full_proj_transform.to(device)
    else:
        # world_view_transform @ projection_matrix
        proj = camera.world_view_transform @ camera.projection_matrix
        proj = proj.to(device)

    # 동차 좌표로 변환
    ones = torch.ones((centers.shape[0], 1), device=device)
    centers_homo = torch.cat([centers, ones], dim=-1)  # [N, 4]

    # 프로젝션
    projected = torch.matmul(centers_homo, proj)  # [N, 4]

    # NDC 좌표
    w = projected[:, 3:4]
    ndc = projected[:, :3] / (w + 1e-10)

    # Frustum culling: NDC 범위 내 (-1, 1) 그리고 z > 0
    in_frustum = (
        (ndc[:, 0] >= -1.2) & (ndc[:, 0] <= 1.2) &
        (ndc[:, 1] >= -1.2) & (ndc[:, 1] <= 1.2) &
        (w.squeeze() > 0)
    )

    visible_mask = opacity_mask & in_frustum
    visible_indices = torch.where(visible_mask)[0]

    return set(visible_indices.cpu().numpy().tolist())
