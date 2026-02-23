#!/usr/bin/env python3
"""
Step 9: Noise Contribution Map 생성

Full render (all gaussians) vs Clean render (clean gaussians only)의 차이로
각 training view에서 noise gaussian의 실제 pixel 기여도를 측정한다.

입력:
  - data/360_v2/{scene}/                (학습 데이터, 카메라 정보)
  - output/{baseline_model}/            (full model checkpoint)
  - clean_gaussians.ply                 (clean model)

출력:
  - {noise_data_dir}/contribution_masks/
    ├── {image_name}.png                (binary mask, 0=clean, 255=noise)
    ├── {image_name}_soft.png           (continuous weight map, 0~255)
    └── metadata.json                   (통계 정보)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams


def compute_contribution_maps(
    dataset: ModelParams,
    pipeline: PipelineParams,
    clean_ply: str,
    output_dir: str,
    full_iteration: int = -1,
    threshold: float = 0.02,
    train_test_exp: bool = False,
):
    """
    Full model과 clean model을 각 training camera에서 렌더링하여
    noise contribution map을 생성한다.

    Args:
        dataset: ModelParams (source_path, model_path 등)
        pipeline: PipelineParams
        clean_ply: clean_gaussians.ply 경로
        output_dir: contribution_masks 저장 경로
        full_iteration: full model의 iteration (-1 = 최신)
        threshold: binary mask 변환 기준값
        train_test_exp: exposure correction 사용 여부
    """
    with torch.no_grad():
        # 1. Full model 로드 (Scene이 checkpoint에서 자동 로드)
        print("=" * 60)
        print("Full model 로드 중...")
        gaussians_full = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians_full, load_iteration=full_iteration, shuffle=False)
        print(f"  Full model: {gaussians_full.get_xyz.shape[0]:,} gaussians")
        print(f"  SH degree: {gaussians_full.active_sh_degree}")

        # 2. Clean model 로드 (PLY에서 직접)
        print(f"\nClean model 로드 중: {clean_ply}")
        gaussians_clean = GaussianModel(dataset.sh_degree)
        gaussians_clean.load_ply(clean_ply, train_test_exp)
        # SH degree를 full model과 맞춤
        gaussians_clean.active_sh_degree = gaussians_full.active_sh_degree
        print(f"  Clean model: {gaussians_clean.get_xyz.shape[0]:,} gaussians")
        print(f"  SH degree: {gaussians_clean.active_sh_degree}")

        # 3. 렌더링 설정
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_cameras = scene.getTrainCameras()
        print(f"\nTraining views: {len(train_cameras)}개")

        # 4. 출력 디렉토리
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 5. 각 training view에 대해 contribution map 생성
        stats = {
            "threshold": threshold,
            "full_model_gaussians": gaussians_full.get_xyz.shape[0],
            "clean_model_gaussians": gaussians_clean.get_xyz.shape[0],
            "noise_gaussians": gaussians_full.get_xyz.shape[0] - gaussians_clean.get_xyz.shape[0],
            "num_views": len(train_cameras),
            "per_view": {},
        }

        coverage_list = []

        for idx, camera in enumerate(tqdm(train_cameras, desc="Contribution map 생성")):
            image_name = camera.image_name

            # Full render
            full_image = render(
                camera, gaussians_full, pipeline, background,
                use_trained_exp=train_test_exp
            )["render"]  # [3, H, W]

            # Clean render
            clean_image = render(
                camera, gaussians_clean, pipeline, background,
                use_trained_exp=train_test_exp
            )["render"]  # [3, H, W]

            # Per-pixel noise contribution (RGB 채널 평균)
            contribution = (full_image - clean_image).abs()  # [3, H, W]
            contribution_gray = contribution.mean(dim=0)      # [H, W]

            # Binary mask
            mask = (contribution_gray > threshold).float()

            # Soft weight map: 1.0 = clean, 0.0 = noise
            # contribution_gray를 [0, 1]로 정규화 후 반전
            max_val = contribution_gray.max()
            if max_val > 1e-7:
                soft_noise = (contribution_gray / max_val).clamp(0, 1)
            else:
                soft_noise = torch.zeros_like(contribution_gray)
            soft_weight = 1.0 - soft_noise  # 1.0=clean, 0.0=max noise

            # 통계
            coverage = mask.mean().item()
            coverage_list.append(coverage)
            mean_contribution = contribution_gray.mean().item()
            max_contribution = contribution_gray.max().item()

            stats["per_view"][image_name] = {
                "coverage": coverage,
                "mean_contribution": mean_contribution,
                "max_contribution": max_contribution,
            }

            # 파일명: 확장자 제거 (image_name = "_DSC9214.JPG" → "_DSC9214")
            base_name = Path(image_name).stem

            # 저장: binary mask (0=clean, 255=noise)
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_np).save(output_dir / f"{base_name}.png")

            # 저장: soft weight map (0=noise, 255=clean)
            soft_np = (soft_weight.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(soft_np).save(output_dir / f"{base_name}_soft.png")

        # 전체 통계
        coverages = np.array(coverage_list)
        stats["summary"] = {
            "mean_coverage": float(coverages.mean()),
            "min_coverage": float(coverages.min()),
            "max_coverage": float(coverages.max()),
            "std_coverage": float(coverages.std()),
            "median_coverage": float(np.median(coverages)),
        }

        # metadata 저장
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(stats, f, indent=2)

        print("\n" + "=" * 60)
        print("Noise Contribution Map 생성 완료")
        print(f"  저장 위치: {output_dir}")
        print(f"  처리된 view: {len(train_cameras)}개")
        print(f"  평균 noise 커버리지: {coverages.mean():.4f} ({coverages.mean()*100:.2f}%)")
        print(f"  최소/최대 커버리지: {coverages.min():.4f} / {coverages.max():.4f}")
        print(f"  Threshold: {threshold}")
        print("=" * 60)


def main():
    parser = ArgumentParser(description="Step 9: Noise Contribution Map Generation")

    # ModelParams를 위한 기본 인자 (source_path, model_path 등)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--clean_ply", required=True, type=str,
                        help="Path to clean_gaussians.ply")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory for contribution masks")
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Full model iteration to load (-1 = latest)")
    parser.add_argument("--threshold", default=0.02, type=float,
                        help="Binary mask threshold for noise contribution")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Safe state
    from utils.general_utils import safe_state
    safe_state(args.quiet)

    dataset = lp.extract(args)
    pipeline = pp.extract(args)

    print("=" * 60)
    print("Step 9: Noise Contribution Map Generation")
    print("=" * 60)
    print(f"Source: {dataset.source_path}")
    print(f"Full model: {dataset.model_path}")
    print(f"Clean PLY: {args.clean_ply}")
    print(f"Output: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    compute_contribution_maps(
        dataset=dataset,
        pipeline=pipeline,
        clean_ply=args.clean_ply,
        output_dir=args.output_dir,
        full_iteration=args.iteration,
        threshold=args.threshold,
        train_test_exp=dataset.train_test_exp,
    )


if __name__ == "__main__":
    main()