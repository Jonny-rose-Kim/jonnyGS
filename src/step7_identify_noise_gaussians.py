#!/usr/bin/env python3
"""
Step 7: 3D Noise Gaussian Identification

2D 노이즈 마스크를 3D 공간의 노이즈 가우시안으로 역투영합니다.

프로세스:
1. Parent 3DGS 모델 로드
2. 카메라 클러스터링 (밀집 영역 선택)
3. 클러스터 내 novel view 생성 및 렌더링
4. 노이즈 마스크 생성 (학습된 detector 사용)
5. Ray-Gaussian intersection으로 노이즈 후보 추출
6. 여러 뷰에서 교집합 계산 (투표 방식)
7. 최종 노이즈 가우시안 저장

입력:
  - output/{scene}/ (Parent 3DGS 모델)
  - experiments/{scene}/checkpoints/best_model.pth (노이즈 감지 모델)

출력:
  - data/processed/{scene}/noise_gaussians/
    ├── noise_gaussians.json      # 노이즈 가우시안 인덱스 및 통계
    ├── noise_gaussians.ply       # 노이즈 가우시안만 포함된 PLY
    ├── clean_gaussians.ply       # 노이즈 제거된 PLY
    └── visualization/            # 시각화 결과
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gaussian_renderer import render, GaussianModel
from scene import Scene
from src.utils.camera_interpolation import interpolate_cameras, find_adjacent_camera_pairs
from src.utils.camera_clustering import cluster_and_select_cameras, find_adjacent_pairs_in_cluster
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


def create_cluster_novel_views(cluster, max_views_per_cluster=5):
    """
    클러스터 내에서 novel view 생성

    Args:
        cluster: 카메라 리스트
        max_views_per_cluster: 클러스터당 최대 novel view 수

    Returns:
        novel_views: List[(interp_cam, cam1, cam2, t)]
    """
    # 인접 카메라 쌍 찾기
    pairs = find_adjacent_pairs_in_cluster(cluster, max_pairs=max_views_per_cluster * 2)

    novel_views = []
    for cam1, cam2, dist in pairs[:max_views_per_cluster]:
        # 중점에서 보간
        interp_cam = interpolate_cameras(cam1, cam2, t=0.5)
        novel_views.append((interp_cam, cam1, cam2, 0.5))

    return novel_views


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
    parser = argparse.ArgumentParser(description='Step 7: 3D Noise Gaussian Identification')
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
    parser.add_argument('--top_k_clusters', type=int, default=3,
                       help='Number of top clusters to use')
    parser.add_argument('--views_per_cluster', type=int, default=5,
                       help='Novel views per cluster')

    # Ray-Gaussian intersection 파라미터
    parser.add_argument('--mahalanobis_threshold', type=float, default=3.0,
                       help='Mahalanobis distance threshold')
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                       help='Mask pixel sampling ratio (1.0 = all)')
    parser.add_argument('--min_opacity', type=float, default=0.01,
                       help='Minimum gaussian opacity')

    # 투표 파라미터
    parser.add_argument('--vote_threshold', type=float, default=0.5,
                       help='Vote threshold for noise classification')

    # 기타
    parser.add_argument('--image_size', type=int, default=256,
                       help='Detector input image size')
    parser.add_argument('--sh_degree', type=int, default=3,
                       help='Spherical harmonics degree')
    parser.add_argument('--save_visualization', action='store_true',
                       help='Save visualization images')

    args = parser.parse_args()

    print("=" * 80)
    print("Step 7: 3D Noise Gaussian Identification")
    print("=" * 80)
    print(f"Scene: {args.scene}")
    print(f"Clusters: {args.n_clusters} → top {args.top_k_clusters}")
    print(f"Views per cluster: {args.views_per_cluster}")
    print(f"Vote threshold: {args.vote_threshold}")
    print(f"Mahalanobis threshold: {args.mahalanobis_threshold}")
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

    if args.save_visualization:
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

    # 4. Novel view 생성 및 노이즈 후보 수집
    print("\nGenerating novel views and collecting noise candidates...")
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    all_view_candidates = []
    all_visible_gaussians = []
    view_metadata = []

    total_views = sum(min(args.views_per_cluster, len(c) * (len(c) - 1) // 2)
                      for c in clusters)

    with tqdm(total=total_views, desc="Processing views") as pbar:
        for cluster_idx, cluster in enumerate(clusters):
            # 클러스터 내 novel view 생성
            novel_views = create_cluster_novel_views(cluster, args.views_per_cluster)

            for view_idx, (interp_cam, cam1, cam2, t) in enumerate(novel_views):
                # 렌더링
                rendered = render_view(gaussians, interp_cam, background)

                # 노이즈 마스크 예측
                mask = detector.predict_from_tensor(rendered)

                # 시각화 저장
                if args.save_visualization:
                    vis_path = vis_dir / f"cluster{cluster_idx}_view{view_idx}.png"
                    # 마스크를 [3, H, W] 형태로 변환
                    if mask.dim() == 2:
                        # [H, W] -> [3, H, W]
                        mask_vis = mask.unsqueeze(0).repeat(3, 1, 1)
                    elif mask.dim() == 3 and mask.shape[0] == 1:
                        # [1, H, W] -> [3, H, W]
                        mask_vis = mask.repeat(3, 1, 1)
                    elif mask.dim() == 4:
                        # [1, 1, H, W] -> [3, H, W]
                        mask_vis = mask.squeeze(0).repeat(3, 1, 1) if mask.shape[1] == 1 else mask.squeeze(0)
                    else:
                        mask_vis = mask

                    # 크기 맞추기 (rendered와 mask 크기가 다를 수 있음)
                    target_h, target_w = rendered.shape[1], rendered.shape[2]
                    if mask_vis.shape[1] != target_h or mask_vis.shape[2] != target_w:
                        # [3, H, W] -> [1, 3, H, W] -> interpolate -> [3, H', W']
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

                all_view_candidates.append(candidates)
                all_visible_gaussians.append(visible)

                view_metadata.append({
                    'cluster_idx': cluster_idx,
                    'view_idx': view_idx,
                    'cam1': cam1.image_name,
                    'cam2': cam2.image_name,
                    'interpolation_t': t,
                    'noise_candidates': len(candidates),
                    'visible_gaussians': len(visible),
                    'mask_noise_ratio': float(mask.mean().item())
                })

                pbar.update(1)

    # 5. 투표 기반 노이즈 가우시안 결정
    print("\nComputing noise gaussians by voting...")
    noise_gaussians = compute_noise_gaussians_voting(
        all_view_candidates,
        all_visible_gaussians,
        vote_threshold=args.vote_threshold
    )

    total_gaussians = gaussians.get_xyz.shape[0]
    noise_ratio = len(noise_gaussians) / total_gaussians

    print(f"  Total gaussians: {total_gaussians}")
    print(f"  Noise gaussians: {len(noise_gaussians)} ({noise_ratio*100:.2f}%)")

    # 6. 결과 저장
    print("\nSaving results...")

    # JSON 저장을 위한 헬퍼 함수
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
        'noise_gaussians': len(noise_gaussians),
        'noise_ratio': float(noise_ratio),
        'parameters': {
            'n_clusters': args.n_clusters,
            'top_k_clusters': args.top_k_clusters,
            'views_per_cluster': args.views_per_cluster,
            'mahalanobis_threshold': args.mahalanobis_threshold,
            'sample_ratio': args.sample_ratio,
            'vote_threshold': args.vote_threshold,
            'min_opacity': args.min_opacity,
        },
        'cluster_statistics': convert_to_serializable(cluster_stats),
        'view_metadata': convert_to_serializable(view_metadata),
        'noise_gaussian_indices': sorted(list(noise_gaussians))
    }

    with open(output_dir / "noise_gaussians.json", 'w') as f:
        json.dump(result, f, indent=2)

    # PLY 저장
    print("  Saving noise_gaussians.ply...")
    save_noise_gaussians_ply(
        gaussians, noise_gaussians,
        output_dir / "noise_gaussians.ply",
        save_noise=True
    )

    print("  Saving clean_gaussians.ply...")
    save_noise_gaussians_ply(
        gaussians, noise_gaussians,
        output_dir / "clean_gaussians.ply",
        save_noise=False
    )

    # 완료
    print("\n" + "=" * 80)
    print("Step 7 Complete!")
    print("=" * 80)
    print(f"Total gaussians: {total_gaussians}")
    print(f"Noise gaussians: {len(noise_gaussians)} ({noise_ratio*100:.2f}%)")
    print(f"Clean gaussians: {total_gaussians - len(noise_gaussians)}")
    print(f"\nOutputs saved to: {output_dir}")
    print("  - noise_gaussians.json")
    print("  - noise_gaussians.ply")
    print("  - clean_gaussians.ply")
    if args.save_visualization:
        print("  - visualization/")
    print("=" * 80)


if __name__ == '__main__':
    main()
