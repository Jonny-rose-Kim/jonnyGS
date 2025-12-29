#!/usr/bin/env python3
"""
Step 7: 3D Noise Gaussian Identification (클러스터별 투표 방식)

2D 노이즈 마스크를 3D 공간의 노이즈 가우시안으로 역투영합니다.

프로세스:
1. Parent 3DGS 모델 로드
2. 카메라 클러스터링 (밀집 영역 선택)
3. 클러스터 내 novel view 생성 및 렌더링
4. 노이즈 마스크 생성 (학습된 detector 사용)
5. Ray-Gaussian intersection으로 노이즈 후보 추출
6. **클러스터별** 투표로 노이즈 가우시안 결정
7. 클러스터별 + Union 결과 저장
8. 카메라 클러스터 시각화

입력:
  - output/{scene}/ (Parent 3DGS 모델)
  - experiments/{scene}/checkpoints/best_model.pth (노이즈 감지 모델)

출력:
  - data/processed/{scene}/noise_gaussians/
    ├── noise_gaussians.json           # 전체 통계 및 클러스터별 결과
    ├── union_noise_gaussians.ply      # 모든 클러스터 Union 노이즈
    ├── union_clean_gaussians.ply      # 모든 클러스터 Union 클린
    ├── cluster_0/
    │   ├── noise_gaussians.ply        # 클러스터 0 노이즈
    │   └── clean_gaussians.ply        # 클러스터 0 클린
    ├── cluster_1/
    │   └── ...
    └── visualization/
        ├── camera_clusters.png        # 카메라 클러스터 시각화
        └── cluster{N}_view{M}.png     # 렌더링 + 마스크
"""

import torch
import sys
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gaussian_renderer import render, GaussianModel
from scene import Scene
from src.utils.camera_interpolation import (
    interpolate_cameras, find_adjacent_camera_pairs,
    interpolate_cameras_look_at, compute_scene_center_robust
)
from src.utils.camera_clustering import cluster_and_select_cameras, find_adjacent_pairs_in_cluster, extract_camera_positions
from src.utils.ray_gaussian_intersection import find_intersecting_gaussians, find_visible_gaussians
from src.utils.detector_inference import load_detector


def load_model_and_cameras(model_path, source_path, sh_degree=3):
    """
    3DGS 모델과 카메라 로드

    Args:
        model_path: 학습된 3DGS 모델 경로
        source_path: 원본 데이터 경로 (카메라 정보)
        sh_degree: Spherical harmonics degree

    Returns:
        (gaussians, cameras, scene)
    """
    print("Loading 3DGS model and cameras...")

    class Args:
        def __init__(self):
            self.model_path = str(model_path)
            self.source_path = str(source_path)
            self.sh_degree = sh_degree
            self.white_background = False
            self.images = "images"
            self.resolution = -1
            self.data_device = "cuda"
            self.eval = False
            self.depths = ""
            self.train_test_exp = False

    args = Args()

    gaussians = GaussianModel(sh_degree)
    scene = Scene(args, gaussians, load_iteration=30000, shuffle=False)
    cameras = scene.getTrainCameras()

    print(f"  Loaded {gaussians.get_xyz.shape[0]} gaussians")
    print(f"  Loaded {len(cameras)} cameras")

    return gaussians, cameras, scene


def render_view(gaussians, camera, background):
    """
    단일 뷰 렌더링

    Args:
        gaussians: GaussianModel
        camera: 카메라 객체
        background: 배경 텐서

    Returns:
        rendered: 렌더링된 이미지 [3, H, W]
    """
    class SimplePipeline:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.antialiasing = False

    pipeline = SimplePipeline()

    with torch.no_grad():
        rendering = render(camera, gaussians, pipeline, background)["render"]

    return rendering


def find_nearest_neighbor_pairs(cluster):
    """
    각 카메라에 대해 가장 가까운 이웃 카메라를 찾아 쌍 생성
    이미 쌍이 된 카메라는 skip

    Args:
        cluster: 카메라 리스트

    Returns:
        pairs: List[(cam, nearest_cam, distance)] - 카메라 쌍 (각 카메라는 한 번만 사용)
    """
    if len(cluster) < 2:
        return []

    positions = extract_camera_positions(cluster)
    n = len(cluster)

    # 이미 쌍이 된 카메라 인덱스 추적
    used = set()
    pairs = []

    for i in range(n):
        # 이미 쌍이 된 카메라는 skip
        if i in used:
            continue

        # 사용되지 않은 카메라 중 가장 가까운 이웃 찾기
        min_dist = float('inf')
        nearest_idx = -1

        for j in range(n):
            if i == j or j in used:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = j

        if nearest_idx >= 0:
            pairs.append((cluster[i], cluster[nearest_idx], min_dist))
            used.add(i)
            used.add(nearest_idx)

    return pairs


def create_cluster_novel_views(cluster, max_views_per_cluster=None, num_interpolations=5,
                                scene_center=None, use_look_at=True, arc_interpolation=True):
    """
    클러스터 내에서 novel view 생성
    각 카메라 쌍에서 균등한 간격으로 여러 개의 novel view 생성

    scene_center가 주어지면 look-at 기반 보간을 사용하여
    항상 scene center를 바라보는 카메라 생성 (floating noise 검출에 효과적)

    Args:
        cluster: 카메라 리스트
        max_views_per_cluster: 최대 novel view 수 (None이면 제한 없음)
        num_interpolations: 각 쌍에서 생성할 novel view 수 (기본값 5)
        scene_center: Scene 중심점 [3] (None이면 기존 방식 사용)
        use_look_at: True면 scene_center를 바라보는 look-at 보간 사용
        arc_interpolation: True면 scene 주위를 도는 arc 경로 사용

    Returns:
        novel_views: List[(interp_cam, cam1, cam2, t)]
    """
    # 각 카메라의 최근접 이웃 쌍 찾기
    pairs = find_nearest_neighbor_pairs(cluster)

    novel_views = []
    for cam1, cam2, dist in pairs:
        # 균등한 간격으로 여러 지점에서 보간
        # num_interpolations=5면 t = 1/6, 2/6, 3/6, 4/6, 5/6
        for i in range(num_interpolations):
            t = (i + 1) / (num_interpolations + 1)

            if scene_center is not None and use_look_at:
                # Look-at 기반 보간: 항상 scene center를 바라봄
                interp_cam = interpolate_cameras_look_at(
                    cam1, cam2, scene_center, t=t,
                    arc_interpolation=arc_interpolation
                )
            else:
                # 기존 방식: 단순 카메라 보간
                interp_cam = interpolate_cameras(cam1, cam2, t=t)

            novel_views.append((interp_cam, cam1, cam2, t))

    # max_views_per_cluster가 지정되면 제한
    if max_views_per_cluster is not None:
        novel_views = novel_views[:max_views_per_cluster]

    return novel_views


def check_view_quality(rendered, min_valid_ratio=0.1):
    """
    렌더링된 뷰의 품질 검사 - 검은색(빈 공간) 비율이 너무 높으면 건너뜀

    Args:
        rendered: 렌더링된 이미지 [3, H, W]
        min_valid_ratio: 최소 유효 픽셀 비율 (기본 10%)

    Returns:
        is_valid: 뷰가 유효한지 여부
        valid_ratio: 유효 픽셀 비율
    """
    # 픽셀 밝기 계산 (RGB 평균)
    brightness = rendered.mean(dim=0)  # [H, W]

    # 너무 어두운 픽셀 = 가우시안이 없는 영역
    valid_pixels = brightness > 0.01  # threshold
    valid_ratio = valid_pixels.float().mean().item()

    return valid_ratio >= min_valid_ratio, valid_ratio


def visualize_camera_clusters(clusters, all_cameras, save_path, scene_name=""):
    """
    카메라 클러스터를 3D로 시각화하고 저장

    Args:
        clusters: 선택된 클러스터 리스트 (List[List[Camera]])
        all_cameras: 전체 카메라 리스트
        save_path: 저장 경로
        scene_name: Scene 이름 (제목용)
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 전체 카메라 위치 (회색으로 배경 표시)
    all_positions = extract_camera_positions(all_cameras)
    ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2],
               c='lightgray', alpha=0.3, s=20, label='All cameras (unused)')

    # 클러스터별 색상
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    cluster_info = []
    for idx, (cluster, color) in enumerate(zip(clusters, colors)):
        positions = extract_camera_positions(cluster)

        # 카메라 위치 표시
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=[color], s=80, label=f'Cluster {idx} ({len(cluster)} cams)',
                   edgecolors='black', linewidths=0.5)

        # 클러스터 중심 표시
        centroid = np.mean(positions, axis=0)
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                   c=[color], s=200, marker='*', edgecolors='black', linewidths=1)

        # 카메라 이름 annotation (일부만)
        for i, cam in enumerate(cluster):
            if i % max(1, len(cluster) // 5) == 0:  # 5개 정도만 표시
                ax.text(positions[i, 0], positions[i, 1], positions[i, 2],
                        f'  {cam.image_name}', fontsize=6, alpha=0.7)

        cluster_info.append({
            'cluster_idx': idx,
            'num_cameras': len(cluster),
            'centroid': centroid.tolist(),
            'camera_names': [cam.image_name for cam in cluster]
        })

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f'Camera Clusters - {scene_name}\n'
                 f'Total: {len(all_cameras)} cameras, '
                 f'Selected: {sum(len(c) for c in clusters)} in {len(clusters)} clusters')

    # 여러 각도에서 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # 추가 각도 저장
    base_path = Path(save_path)
    for angle, elev in [(45, 20), (135, 20), (225, 20), (315, 20)]:
        ax.view_init(elev=elev, azim=angle)
        angle_path = base_path.parent / f"{base_path.stem}_angle{angle}{base_path.suffix}"
        plt.savefig(angle_path, dpi=150, bbox_inches='tight')

    plt.close()

    return cluster_info


def compute_noise_gaussians_voting(
    all_view_candidates: list,
    all_visible_gaussians: list,
    vote_threshold: float = 0.5
) -> set:
    """
    투표 기반 노이즈 가우시안 결정

    Args:
        all_view_candidates: List[Set[int]] - 각 뷰별 노이즈 후보 인덱스
        all_visible_gaussians: List[Set[int]] - 각 뷰별 가시 가우시안 인덱스
        vote_threshold: 투표 임계값 (0.5 = 50%)

    Returns:
        noise_gaussians: 최종 노이즈 가우시안 인덱스 집합
    """
    # 가우시안별 투표 수 및 가시 횟수 계산
    vote_count = Counter()
    visible_count = Counter()

    for candidates, visible in zip(all_view_candidates, all_visible_gaussians):
        for g_idx in candidates:
            vote_count[g_idx] += 1

        for g_idx in visible:
            visible_count[g_idx] += 1

    # 투표 비율 기준 필터링
    noise_gaussians = set()
    for g_idx, votes in vote_count.items():
        visible_views = visible_count.get(g_idx, 1)  # 최소 1
        if votes / visible_views >= vote_threshold:
            noise_gaussians.add(g_idx)

    return noise_gaussians


def save_noise_gaussians_ply(gaussians, noise_indices, output_path, save_noise=True):
    """
    노이즈 또는 클린 가우시안을 PLY로 저장

    Args:
        gaussians: GaussianModel
        noise_indices: 노이즈 가우시안 인덱스 집합
        output_path: 출력 경로
        save_noise: True면 노이즈만, False면 클린만 저장
    """
    from plyfile import PlyData, PlyElement

    # 인덱스 마스크 생성
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    mask = torch.zeros(N, dtype=torch.bool, device=device)

    # 인덱스를 텐서로 변환하여 한 번에 설정
    if len(noise_indices) > 0:
        indices_tensor = torch.tensor(list(noise_indices), dtype=torch.long, device=device)
        valid_indices = indices_tensor[indices_tensor < N]
        mask[valid_indices] = True

    if not save_noise:
        mask = ~mask  # 클린 가우시안

    print(f"    Saving {mask.sum().item()} gaussians (save_noise={save_noise})")

    # 데이터 추출
    xyz = gaussians.get_xyz[mask].detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    f_dc = gaussians._features_dc[mask].detach().cpu().numpy()
    f_dc = f_dc.transpose(0, 2, 1).reshape(f_dc.shape[0], -1)

    f_rest = gaussians._features_rest[mask].detach().cpu().numpy()
    f_rest = f_rest.transpose(0, 2, 1).reshape(f_rest.shape[0], -1)

    opacities = gaussians._opacity[mask].detach().cpu().numpy()
    scale = gaussians._scaling[mask].detach().cpu().numpy()
    rotation = gaussians._rotation[mask].detach().cpu().numpy()

    # PLY 속성 구성
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
    ]

    for i in range(f_dc.shape[1]):
        dtype_full.append((f'f_dc_{i}', 'f4'))
    for i in range(f_rest.shape[1]):
        dtype_full.append((f'f_rest_{i}', 'f4'))

    dtype_full.append(('opacity', 'f4'))
    for i in range(scale.shape[1]):
        dtype_full.append((f'scale_{i}', 'f4'))
    for i in range(rotation.shape[1]):
        dtype_full.append((f'rot_{i}', 'f4'))

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scale, rotation], axis=1)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(str(output_path))


def main():
    parser = argparse.ArgumentParser(description='Step 7: 3D Noise Gaussian Identification (클러스터별 투표)')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--model_path', default=None,
                       help='Path to parent 3DGS model (default: output/{scene})')
    parser.add_argument('--detector_checkpoint', default=None,
                       help='Path to detector checkpoint')
    parser.add_argument('--data_root', default='./data/360_v2', help='Data root directory')
    parser.add_argument('--output_root', default='./data/processed', help='Output root directory')

    # 클러스터링 파라미터
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of camera clusters')
    parser.add_argument('--min_cameras_per_cluster', type=int, default=5,
                       help='Minimum cameras per cluster')
    parser.add_argument('--top_k_clusters', type=int, default=None,
                       help='Number of top clusters to use (default: all clusters)')
    parser.add_argument('--views_per_cluster', type=int, default=None,
                       help='Max novel views per cluster (default: no limit)')
    parser.add_argument('--num_interpolations', type=int, default=5,
                       help='Number of novel views per camera pair (default: 5)')

    # Look-at 기반 카메라 보간 (floating noise 검출 개선)
    parser.add_argument('--use_look_at', action='store_true', default=True,
                       help='Use look-at interpolation (always look at scene center)')
    parser.add_argument('--no_look_at', dest='use_look_at', action='store_false',
                       help='Disable look-at interpolation (use original method)')
    parser.add_argument('--arc_interpolation', action='store_true', default=True,
                       help='Use arc interpolation around scene center')
    parser.add_argument('--no_arc', dest='arc_interpolation', action='store_false',
                       help='Use linear interpolation instead of arc')
    parser.add_argument('--min_valid_ratio', type=float, default=0.1,
                       help='Minimum valid pixel ratio (skip views with too much black)')

    # Ray-Gaussian intersection 파라미터
    parser.add_argument('--mahalanobis_threshold', type=float, default=3.0,
                       help='Mahalanobis distance threshold')
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                       help='Mask pixel sampling ratio (1.0 = all)')
    parser.add_argument('--min_opacity', type=float, default=0.01,
                       help='Minimum gaussian opacity')

    # 투표 파라미터
    parser.add_argument('--vote_threshold', type=float, default=0.5,
                       help='Vote threshold for noise classification (per cluster)')

    # 기타
    parser.add_argument('--image_size', type=int, default=256,
                       help='Detector input image size')
    parser.add_argument('--sh_degree', type=int, default=3,
                       help='Spherical harmonics degree')
    parser.add_argument('--save_visualization', action='store_true',
                       help='Save visualization images')

    args = parser.parse_args()

    print("=" * 80)
    print("Step 7: 3D Noise Gaussian Identification (클러스터별 투표)")
    print("=" * 80)
    print(f"Scene: {args.scene}")
    top_k_str = "all" if args.top_k_clusters is None else args.top_k_clusters
    print(f"Clusters: {args.n_clusters} → use {top_k_str}")
    views_str = "no limit" if args.views_per_cluster is None else args.views_per_cluster
    print(f"Max views per cluster: {views_str}")
    print(f"Interpolations per pair: {args.num_interpolations}")
    print(f"Vote threshold (per cluster): {args.vote_threshold}")
    print(f"Mahalanobis threshold: {args.mahalanobis_threshold}")
    print(f"Look-at interpolation: {args.use_look_at}")
    print(f"Arc interpolation: {args.arc_interpolation}")
    print(f"Min valid pixel ratio: {args.min_valid_ratio}")
    print("=" * 80 + "\n")

    # 경로 설정
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path("output") / args.scene

    if args.detector_checkpoint:
        detector_path = Path(args.detector_checkpoint)
    else:
        detector_path = Path("experiments") / args.scene / "checkpoints" / "best_model.pth"

    source_path = Path(args.data_root) / args.scene
    output_dir = Path(args.output_root) / args.scene / "noise_gaussians"
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(exist_ok=True)

    # 검증
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector not found: {detector_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_path}")

    # 1. 모델 및 카메라 로드
    gaussians, cameras, scene = load_model_and_cameras(
        model_path, source_path, sh_degree=args.sh_degree
    )

    # 2. 노이즈 감지기 로드
    print("\nLoading noise detector...")
    detector = load_detector(detector_path, device='cuda', image_size=args.image_size)

    # 3. 카메라 클러스터링
    print("\nClustering cameras...")
    clusters, cluster_stats = cluster_and_select_cameras(
        cameras,
        method='kmeans',
        n_clusters=args.n_clusters,
        min_cameras_per_cluster=args.min_cameras_per_cluster,
        top_k_clusters=args.top_k_clusters
    )

    if len(clusters) == 0:
        raise ValueError("No valid clusters found. Try lowering min_cameras_per_cluster.")

    # 4. 카메라 클러스터 시각화
    print("\nVisualizing camera clusters...")
    cluster_vis_path = vis_dir / "camera_clusters.png"
    cluster_info = visualize_camera_clusters(clusters, cameras, cluster_vis_path, args.scene)
    print(f"  Saved cluster visualization to {cluster_vis_path}")

    # 5. Scene center 계산 (look-at 보간 사용 시)
    scene_center = None
    if args.use_look_at:
        print("\nComputing scene center (robust, 90th percentile)...")
        scene_center = compute_scene_center_robust(gaussians, percentile=90)
        print(f"  Scene center: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")

    # 6. 클러스터별 Novel view 생성 및 노이즈 후보 수집
    print("\nGenerating novel views and collecting noise candidates (per cluster)...")
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    total_gaussians = gaussians.get_xyz.shape[0]

    # 클러스터별 데이터 저장
    cluster_results = []
    union_noise_gaussians = set()  # 전체 Union

    # total_views 계산: 각 클러스터의 쌍 수 * num_interpolations
    def estimate_views_for_cluster(cluster_size):
        num_pairs = cluster_size // 2  # 각 카메라는 한 번만 사용
        num_views = num_pairs * args.num_interpolations
        if args.views_per_cluster is not None:
            num_views = min(num_views, args.views_per_cluster)
        return num_views

    total_views = sum(estimate_views_for_cluster(len(c)) for c in clusters)

    with tqdm(total=total_views, desc="Processing views") as pbar:
        for cluster_idx, cluster in enumerate(clusters):
            print(f"\n  Processing Cluster {cluster_idx} ({len(cluster)} cameras)...")

            # 클러스터별 출력 디렉토리
            cluster_dir = output_dir / f"cluster_{cluster_idx}"
            cluster_dir.mkdir(exist_ok=True)

            # 클러스터 내 후보 수집
            cluster_view_candidates = []
            cluster_visible_gaussians = []
            cluster_view_metadata = []

            # 클러스터 내 novel view 생성 (look-at 기반)
            novel_views = create_cluster_novel_views(
                cluster,
                max_views_per_cluster=args.views_per_cluster,
                num_interpolations=args.num_interpolations,
                scene_center=scene_center,
                use_look_at=args.use_look_at,
                arc_interpolation=args.arc_interpolation
            )

            skipped_views = 0
            for view_idx, (interp_cam, cam1, cam2, t) in enumerate(novel_views):
                # 렌더링
                rendered = render_view(gaussians, interp_cam, background)

                # View quality check - 검은색 영역이 너무 많으면 건너뜀
                is_valid, valid_ratio = check_view_quality(rendered, args.min_valid_ratio)
                if not is_valid:
                    skipped_views += 1
                    pbar.update(1)
                    continue

                # 노이즈 마스크 예측
                mask = detector.predict_from_tensor(rendered)

                # 시각화 저장
                if args.save_visualization:
                    vis_path = vis_dir / f"cluster{cluster_idx}_view{view_idx}.png"
                    # 마스크를 [3, H, W] 형태로 변환
                    if mask.dim() == 2:
                        mask_vis = mask.unsqueeze(0).repeat(3, 1, 1)
                    elif mask.dim() == 3 and mask.shape[0] == 1:
                        mask_vis = mask.repeat(3, 1, 1)
                    elif mask.dim() == 4:
                        mask_vis = mask.squeeze(0).repeat(3, 1, 1) if mask.shape[1] == 1 else mask.squeeze(0)
                    else:
                        mask_vis = mask

                    # 크기 맞추기
                    target_h, target_w = rendered.shape[1], rendered.shape[2]
                    if mask_vis.shape[1] != target_h or mask_vis.shape[2] != target_w:
                        mask_vis = torch.nn.functional.interpolate(
                            mask_vis.unsqueeze(0),
                            size=(target_h, target_w),
                            mode='nearest'
                        ).squeeze(0)

                    save_combined = torch.cat([rendered, mask_vis], dim=2)
                    torchvision.utils.save_image(save_combined, str(vis_path))

                # Ray-Gaussian intersection
                candidates = find_intersecting_gaussians(
                    interp_cam, mask, gaussians,
                    threshold=args.mahalanobis_threshold,
                    sample_ratio=args.sample_ratio,
                    min_opacity=args.min_opacity,
                    verbose=False
                )

                # 가시 가우시안
                visible = find_visible_gaussians(interp_cam, gaussians, args.min_opacity)

                cluster_view_candidates.append(candidates)
                cluster_visible_gaussians.append(visible)

                cluster_view_metadata.append({
                    'view_idx': view_idx,
                    'cam1': cam1.image_name,
                    'cam2': cam2.image_name,
                    'interpolation_t': t,
                    'noise_candidates': len(candidates),
                    'visible_gaussians': len(visible),
                    'mask_noise_ratio': float(mask.mean().item()),
                    'valid_pixel_ratio': valid_ratio
                })

                pbar.update(1)

            if skipped_views > 0:
                print(f"    Skipped {skipped_views} views with too much black (< {args.min_valid_ratio*100:.0f}% valid pixels)")

            # 클러스터별 투표 (유효한 뷰가 있을 때만)
            print(f"    Computing noise gaussians for cluster {cluster_idx}...")
            cluster_noise_gaussians = compute_noise_gaussians_voting(
                cluster_view_candidates,
                cluster_visible_gaussians,
                vote_threshold=args.vote_threshold
            )

            cluster_noise_ratio = len(cluster_noise_gaussians) / total_gaussians if total_gaussians > 0 else 0
            print(f"    Cluster {cluster_idx}: {len(cluster_noise_gaussians)} noise gaussians ({cluster_noise_ratio*100:.2f}%)")

            # Union에 추가
            union_noise_gaussians.update(cluster_noise_gaussians)

            # 클러스터별 PLY 저장
            print(f"    Saving cluster {cluster_idx} PLY files...")
            save_noise_gaussians_ply(
                gaussians, cluster_noise_gaussians,
                cluster_dir / "noise_gaussians.ply",
                save_noise=True
            )
            save_noise_gaussians_ply(
                gaussians, cluster_noise_gaussians,
                cluster_dir / "clean_gaussians.ply",
                save_noise=False
            )

            # 클러스터 결과 저장
            cluster_results.append({
                'cluster_idx': cluster_idx,
                'num_cameras': len(cluster),
                'num_views': len(novel_views),
                'num_valid_views': len(novel_views) - skipped_views,
                'num_skipped_views': skipped_views,
                'camera_names': [cam.image_name for cam in cluster],
                'noise_gaussians_count': len(cluster_noise_gaussians),
                'noise_ratio': cluster_noise_ratio,
                'noise_gaussian_indices': sorted(list(cluster_noise_gaussians)),
                'view_metadata': cluster_view_metadata
            })

    # 7. 최종 결과 저장 (모든 클러스터 Union)
    print("\n" + "-" * 40)
    print("Saving final results (all clusters combined)...")
    final_noise_ratio = len(union_noise_gaussians) / total_gaussians if total_gaussians > 0 else 0

    print(f"  Final noise gaussians: {len(union_noise_gaussians)} ({final_noise_ratio*100:.2f}%)")

    save_noise_gaussians_ply(
        gaussians, union_noise_gaussians,
        output_dir / "noise_gaussians.ply",
        save_noise=True
    )
    save_noise_gaussians_ply(
        gaussians, union_noise_gaussians,
        output_dir / "clean_gaussians.ply",
        save_noise=False
    )

    # 8. JSON 결과 저장
    print("\nSaving JSON results...")

    def convert_to_serializable(obj):
        """numpy/torch 타입을 JSON 직렬화 가능한 타입으로 변환"""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    result = {
        'scene': args.scene,
        'model_path': str(model_path),
        'total_gaussians': int(total_gaussians),
        'noise_gaussians_count': len(union_noise_gaussians),
        'noise_ratio': final_noise_ratio,
        'clean_gaussians_count': total_gaussians - len(union_noise_gaussians),
        'parameters': {
            'n_clusters': args.n_clusters,
            'top_k_clusters': args.top_k_clusters,
            'views_per_cluster': args.views_per_cluster,
            'mahalanobis_threshold': args.mahalanobis_threshold,
            'sample_ratio': args.sample_ratio,
            'vote_threshold': args.vote_threshold,
            'min_opacity': args.min_opacity,
            'use_look_at': args.use_look_at,
            'arc_interpolation': args.arc_interpolation,
            'min_valid_ratio': args.min_valid_ratio,
        },
        'scene_center': scene_center.tolist() if scene_center is not None else None,
        'cluster_statistics': convert_to_serializable(cluster_stats),
        'cluster_info': convert_to_serializable(cluster_info),
        'cluster_results': convert_to_serializable(cluster_results),
        'noise_gaussian_indices': sorted(list(union_noise_gaussians))
    }

    with open(output_dir / "noise_gaussians.json", 'w') as f:
        json.dump(result, f, indent=2)

    # 완료 요약
    print("\n" + "=" * 80)
    print("Step 7 Complete! (클러스터별 투표 → 전체 Union)")
    print("=" * 80)
    print(f"Total gaussians: {total_gaussians}")
    print(f"\nCluster-wise Voting Results:")
    for cr in cluster_results:
        print(f"  Cluster {cr['cluster_idx']}: {cr['noise_gaussians_count']} noise ({cr['noise_ratio']*100:.2f}%)")
    print(f"\nFinal Result (Union of all clusters):")
    print(f"  Noise gaussians: {len(union_noise_gaussians)} ({final_noise_ratio*100:.2f}%)")
    print(f"  Clean gaussians: {total_gaussians - len(union_noise_gaussians)}")
    print(f"\nOutputs saved to: {output_dir}")
    print("  - noise_gaussians.ply (최종 노이즈)")
    print("  - clean_gaussians.ply (노이즈 제거된 모델)")
    print("  - noise_gaussians.json (전체 결과)")
    for i in range(len(clusters)):
        print(f"  - cluster_{i}/ (클러스터별 결과)")
    print("  - visualization/camera_clusters.png")
    if args.save_visualization:
        print("  - visualization/cluster*_view*.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
