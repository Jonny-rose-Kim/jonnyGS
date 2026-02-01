#!/usr/bin/env python3
"""
Step 8 Evaluation: Compare Baseline vs Retrained Models

두 모델(Baseline, Retrained)을 비교 평가합니다.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio) - 높을수록 좋음
- SSIM (Structural Similarity Index) - 높을수록 좋음
- LPIPS (Learned Perceptual Image Patch Similarity) - 낮을수록 좋음

Usage:
    # 기본 비교 (test 이미지가 이미 렌더링된 경우)
    python src/step8_evaluate.py \\
        --baseline output/stump \\
        --retrained output/stump_retrained_15k

    # 렌더링 + 평가 (처음 실행 시)
    python src/step8_evaluate.py \\
        --baseline output/stump \\
        --retrained output/stump_retrained_15k \\
        --render

    # 여러 retrained 모델 비교
    python src/step8_evaluate.py \\
        --baseline output/stump \\
        --retrained output/stump_retrained_10k output/stump_retrained_15k output/stump_retrained_30k
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.loss_utils import ssim as compute_ssim
from utils.image_utils import psnr as compute_psnr

# LPIPS 로드 시도
try:
    from lpipsPyTorch import lpips as compute_lpips
    LPIPS_AVAILABLE = True
except ImportError:
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        def compute_lpips(img1, img2, net_type='vgg'):
            return lpips_fn(img1, img2).item()
        LPIPS_AVAILABLE = True
    except ImportError:
        LPIPS_AVAILABLE = False
        print("WARNING: LPIPS not available. Install with: pip install lpips")


def find_latest_iteration(model_path: Path) -> int:
    """모델의 최신 iteration 찾기"""
    test_dir = model_path / "test"
    if not test_dir.exists():
        # point_cloud 디렉토리에서 찾기
        pc_dir = model_path / "point_cloud"
        if pc_dir.exists():
            iterations = []
            for d in pc_dir.iterdir():
                if d.is_dir() and d.name.startswith("iteration_"):
                    try:
                        iterations.append(int(d.name.split("_")[1]))
                    except ValueError:
                        pass
            if iterations:
                return max(iterations)
        return -1

    iterations = []
    for d in test_dir.iterdir():
        if d.is_dir() and d.name.startswith("ours_"):
            try:
                iterations.append(int(d.name.split("_")[1]))
            except ValueError:
                pass

    return max(iterations) if iterations else -1


def run_render(model_path: Path, skip_train: bool = True) -> bool:
    """render.py 실행"""
    print(f"  Rendering model: {model_path}")

    cmd = [
        "python", "render.py",
        "-m", str(model_path),
    ]

    if skip_train:
        cmd.append("--skip_train")

    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Rendering failed for {model_path}")
        return False


def load_images(renders_dir: Path, gt_dir: Path) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """렌더링된 이미지와 GT 이미지 로드"""
    renders = []
    gts = []
    image_names = []

    if not renders_dir.exists() or not gt_dir.exists():
        return renders, gts, image_names

    for fname in sorted(os.listdir(renders_dir)):
        if not fname.endswith(('.png', '.jpg', '.jpeg')):
            continue

        render_path = renders_dir / fname
        gt_path = gt_dir / fname

        if not gt_path.exists():
            continue

        try:
            render = Image.open(render_path)
            gt = Image.open(gt_path)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
        except Exception as e:
            print(f"  Warning: Failed to load {fname}: {e}")

    return renders, gts, image_names


def evaluate_model(model_path: Path, iteration: int = -1) -> Dict:
    """단일 모델 평가"""
    results = {
        "model_path": str(model_path),
        "iteration": iteration,
        "metrics": {},
        "per_view": {},
        "num_views": 0,
    }

    # iteration 자동 탐지
    if iteration == -1:
        iteration = find_latest_iteration(model_path)
        results["iteration"] = iteration

    if iteration == -1:
        print(f"  ERROR: No iterations found for {model_path}")
        return results

    # 테스트 디렉토리 찾기
    test_dir = model_path / "test" / f"ours_{iteration}"
    if not test_dir.exists():
        print(f"  ERROR: Test directory not found: {test_dir}")
        print(f"  Run: python render.py -m {model_path} --skip_train")
        return results

    renders_dir = test_dir / "renders"
    gt_dir = test_dir / "gt"

    # 이미지 로드
    renders, gts, image_names = load_images(renders_dir, gt_dir)

    if len(renders) == 0:
        print(f"  ERROR: No images found in {renders_dir}")
        return results

    results["num_views"] = len(renders)
    print(f"  Evaluating {len(renders)} test views...")

    # 메트릭 계산
    psnrs = []
    ssims = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="  Computing metrics", leave=False):
        psnr_val = compute_psnr(renders[idx], gts[idx]).item()
        ssim_val = compute_ssim(renders[idx], gts[idx]).item()
        psnrs.append(psnr_val)
        ssims.append(ssim_val)

        if LPIPS_AVAILABLE:
            lpips_val = compute_lpips(renders[idx], gts[idx], net_type='vgg')
            if isinstance(lpips_val, torch.Tensor):
                lpips_val = lpips_val.item()
            lpipss.append(lpips_val)

        results["per_view"][image_names[idx]] = {
            "PSNR": psnr_val,
            "SSIM": ssim_val,
        }
        if LPIPS_AVAILABLE:
            results["per_view"][image_names[idx]]["LPIPS"] = lpips_val

    # 평균 계산
    results["metrics"]["PSNR"] = sum(psnrs) / len(psnrs)
    results["metrics"]["SSIM"] = sum(ssims) / len(ssims)
    if LPIPS_AVAILABLE and lpipss:
        results["metrics"]["LPIPS"] = sum(lpipss) / len(lpipss)

    return results


def get_gaussian_count(model_path: Path, iteration: int = -1) -> int:
    """PLY 파일에서 gaussian 수 확인"""
    if iteration == -1:
        iteration = find_latest_iteration(model_path)

    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    if not ply_path.exists():
        return -1

    try:
        with open(ply_path, 'rb') as f:
            for _ in range(20):  # 헤더 내에서 찾기
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('element vertex'):
                    return int(line.split()[-1])
    except:
        pass

    return -1


def print_comparison_table(baseline_results: Dict, retrained_results: List[Dict]):
    """비교 테이블 출력"""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # 헤더
    metrics = ["PSNR", "SSIM"]
    if LPIPS_AVAILABLE:
        metrics.append("LPIPS")

    # 테이블 헤더
    header = f"{'Model':<35} {'Iter':>8} {'#Gauss':>10}"
    for m in metrics:
        header += f" {m:>10}"
    print(header)
    print("-" * 80)

    # Baseline
    baseline_name = Path(baseline_results["model_path"]).name
    baseline_iter = baseline_results["iteration"]
    baseline_gaussians = get_gaussian_count(Path(baseline_results["model_path"]), baseline_iter)

    row = f"{baseline_name + ' (baseline)':<35} {baseline_iter:>8} {baseline_gaussians:>10}"
    for m in metrics:
        val = baseline_results["metrics"].get(m, 0)
        row += f" {val:>10.4f}"
    print(row)

    # Retrained models
    for retrained in retrained_results:
        retrained_name = Path(retrained["model_path"]).name
        retrained_iter = retrained["iteration"]
        retrained_gaussians = get_gaussian_count(Path(retrained["model_path"]), retrained_iter)

        row = f"{retrained_name:<35} {retrained_iter:>8} {retrained_gaussians:>10}"
        for m in metrics:
            val = retrained["metrics"].get(m, 0)
            baseline_val = baseline_results["metrics"].get(m, 0)
            delta = val - baseline_val

            # LPIPS는 낮을수록 좋음
            if m == "LPIPS":
                better = delta < 0
            else:
                better = delta > 0

            indicator = "+" if delta >= 0 else ""
            color_indicator = " ^" if better else " v" if delta != 0 else "  "
            row += f" {val:>10.4f}"
        print(row)

    print("-" * 80)

    # Delta 테이블
    print("\nDELTA (vs Baseline):")
    print("-" * 80)

    header = f"{'Model':<35}"
    for m in metrics:
        header += f" {m:>10}"
    header += "  Better?"
    print(header)
    print("-" * 80)

    for retrained in retrained_results:
        retrained_name = Path(retrained["model_path"]).name

        row = f"{retrained_name:<35}"
        better_count = 0
        total_count = 0

        for m in metrics:
            val = retrained["metrics"].get(m, 0)
            baseline_val = baseline_results["metrics"].get(m, 0)
            delta = val - baseline_val

            # LPIPS는 낮을수록 좋음
            if m == "LPIPS":
                is_better = delta < 0
            else:
                is_better = delta > 0

            if is_better:
                better_count += 1
            total_count += 1

            sign = "+" if delta >= 0 else ""
            row += f" {sign}{delta:>9.4f}"

        verdict = "YES" if better_count > total_count / 2 else "NO"
        row += f"  {verdict} ({better_count}/{total_count})"
        print(row)

    print("=" * 80)


def save_results(output_path: Path, baseline_results: Dict, retrained_results: List[Dict]):
    """결과를 JSON으로 저장"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline_results,
        "retrained": retrained_results,
        "comparison": []
    }

    metrics = ["PSNR", "SSIM"]
    if LPIPS_AVAILABLE:
        metrics.append("LPIPS")

    for retrained in retrained_results:
        comparison = {
            "model": Path(retrained["model_path"]).name,
            "deltas": {}
        }
        for m in metrics:
            baseline_val = baseline_results["metrics"].get(m, 0)
            retrained_val = retrained["metrics"].get(m, 0)
            comparison["deltas"][m] = retrained_val - baseline_val
        results["comparison"].append(comparison)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Step 8 Evaluation: Compare Baseline vs Retrained Models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline model (e.g., output/stump)')
    parser.add_argument('--retrained', type=str, nargs='+', required=True,
                       help='Path(s) to retrained model(s)')
    parser.add_argument('--iteration', type=int, default=-1,
                       help='Specific iteration to evaluate (-1 for latest)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON path (default: auto)')
    parser.add_argument('--render', action='store_true',
                       help='Run render.py before evaluation')
    parser.add_argument('--skip_train', action='store_true', default=True,
                       help='Skip training views when rendering (default: True)')

    args = parser.parse_args()

    # 경로 변환
    baseline_path = Path(args.baseline).resolve()
    retrained_paths = [Path(p).resolve() for p in args.retrained]

    print("=" * 80)
    print("Step 8 Evaluation: Baseline vs Retrained Model Comparison")
    print("=" * 80)
    print(f"Baseline:  {baseline_path}")
    for i, p in enumerate(retrained_paths):
        print(f"Retrained[{i}]: {p}")
    print("=" * 80)

    # 경로 검증
    if not baseline_path.exists():
        print(f"ERROR: Baseline model not found: {baseline_path}")
        sys.exit(1)

    for p in retrained_paths:
        if not p.exists():
            print(f"ERROR: Retrained model not found: {p}")
            sys.exit(1)

    # 렌더링 (옵션)
    if args.render:
        print("\n[1/3] Rendering models...")
        all_paths = [baseline_path] + retrained_paths

        for path in all_paths:
            if not run_render(path, args.skip_train):
                print(f"  WARNING: Rendering failed for {path}")
    else:
        print("\n[1/3] Skipping rendering (use --render to render first)")

    # Baseline 평가
    print("\n[2/3] Evaluating baseline model...")
    baseline_results = evaluate_model(baseline_path, args.iteration)

    if not baseline_results["metrics"]:
        print("ERROR: Failed to evaluate baseline model")
        print(f"  Check if test images exist: {baseline_path}/test/")
        print(f"  Run: python render.py -m {baseline_path} --skip_train")
        sys.exit(1)

    print(f"  PSNR: {baseline_results['metrics'].get('PSNR', 0):.4f}")
    print(f"  SSIM: {baseline_results['metrics'].get('SSIM', 0):.4f}")
    if LPIPS_AVAILABLE:
        print(f"  LPIPS: {baseline_results['metrics'].get('LPIPS', 0):.4f}")

    # Retrained 모델 평가
    print("\n[3/3] Evaluating retrained model(s)...")
    retrained_results = []

    for path in retrained_paths:
        print(f"\n  Model: {path.name}")
        results = evaluate_model(path, args.iteration)

        if results["metrics"]:
            print(f"  PSNR: {results['metrics'].get('PSNR', 0):.4f}")
            print(f"  SSIM: {results['metrics'].get('SSIM', 0):.4f}")
            if LPIPS_AVAILABLE:
                print(f"  LPIPS: {results['metrics'].get('LPIPS', 0):.4f}")
            retrained_results.append(results)
        else:
            print(f"  WARNING: Failed to evaluate {path}")

    if not retrained_results:
        print("\nERROR: No retrained models could be evaluated")
        sys.exit(1)

    # 비교 테이블 출력
    print_comparison_table(baseline_results, retrained_results)

    # 결과 저장
    if args.output:
        output_path = Path(args.output)
    else:
        # 자동 경로 생성
        output_dir = retrained_paths[0].parent / "evaluation"
        output_path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_results(output_path, baseline_results, retrained_results)

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
