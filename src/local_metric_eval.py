#!/usr/bin/env python3
"""
Local Metric Evaluation: noise 영역 vs clean 영역 분리 평가

Contribution mask를 이용해 이미지를 noise/clean 영역으로 나누고,
각 영역별로 PSNR과 SSIM을 따로 계산한다.
"""

import os
import sys
import csv
import json
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim


def load_image(path: str) -> np.ndarray:
    """이미지를 [H, W, 3] float32 (0~1)로 로드."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def load_mask(path: str, threshold: float) -> np.ndarray:
    """
    Mask 파일을 로드하여 binary mask로 변환.
    Returns: [H, W] bool, True=noise
    """
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        mask = np.load(path).astype(np.float32)
    else:  # .png, .jpg 등
        mask = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return mask >= threshold


def masked_psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """
    mask=True인 픽셀에 대해서만 PSNR 계산.
    img1, img2: [H, W, 3], mask: [H, W] bool
    """
    if mask.sum() == 0:
        return float("nan")
    diff = (img1 - img2) ** 2  # [H, W, 3]
    # mask 영역의 모든 채널에 대해 MSE
    mse = diff[mask].mean()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def masked_ssim(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """
    mask=True인 영역의 bounding box를 crop하여 SSIM 계산.
    """
    if mask.sum() == 0:
        return float("nan")

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # bounding box crop
    crop1 = img1[rmin:rmax+1, cmin:cmax+1]
    crop2 = img2[rmin:rmax+1, cmin:cmax+1]

    # crop 영역이 너무 작으면 skip (SSIM win_size=7 최소 필요)
    if crop1.shape[0] < 7 or crop1.shape[1] < 7:
        return float("nan")

    return compare_ssim(crop1, crop2, channel_axis=2, data_range=1.0)


def global_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = ((img1 - img2) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def global_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return compare_ssim(img1, img2, channel_axis=2, data_range=1.0)


def build_index_to_name_mapping(cameras_json_path: str, split: str = "train") -> dict:
    """
    cameras.json + LLFF hold (every 8th → test)로 index→image_name 매핑 생성.
    render.py는 enumerate(views) 순서로 00000.png, 00001.png ... 저장.
    Scene은 shuffle=False, LLFF hold 기준으로 정렬.
    """
    with open(cameras_json_path) as f:
        cameras = json.load(f)

    cam_names = sorted([c["img_name"] for c in cameras])

    if split == "train":
        names = [n for i, n in enumerate(cam_names) if i % 8 != 0]
    elif split == "test":
        names = [n for i, n in enumerate(cam_names) if i % 8 == 0]
    else:
        names = cam_names

    return {i: n for i, n in enumerate(names)}


def find_mask_file(mask_dir: Path, image_name: str) -> str:
    """
    image_name (e.g., "_DSC9214.JPG") 에 해당하는 mask 파일 찾기.
    contribution mask는 stem 기준으로 저장됨 (e.g., "_DSC9214.png").
    _soft 파일은 제외.
    """
    stem = Path(image_name).stem  # "_DSC9214"

    for ext in [".png", ".npy", ".jpg"]:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def evaluate(args):
    gt_dir = Path(args.gt_dir)
    baseline_dir = Path(args.baseline_render_dir)
    noise_aware_dir = Path(args.noise_aware_render_dir)
    mask_dir = Path(args.mask_dir)

    # index → image_name 매핑
    idx_to_name = None
    if args.cameras_json:
        idx_to_name = build_index_to_name_mapping(args.cameras_json, args.split)
        print(f"[INFO] cameras.json 로드: {len(idx_to_name)} {args.split} views")

    # GT 파일 목록 (정렬)
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])
    baseline_files = sorted([f for f in os.listdir(baseline_dir) if f.endswith(".png")])
    noise_aware_files = sorted([f for f in os.listdir(noise_aware_dir) if f.endswith(".png")])

    if len(gt_files) != len(baseline_files) or len(gt_files) != len(noise_aware_files):
        print(f"[WARNING] 파일 수 불일치: GT={len(gt_files)}, "
              f"baseline={len(baseline_files)}, noise_aware={len(noise_aware_files)}")

    n_files = min(len(gt_files), len(baseline_files), len(noise_aware_files))

    # 결과 수집
    results = {
        "baseline": {"global_psnr": [], "global_ssim": [],
                     "noise_psnr": [], "noise_ssim": [],
                     "clean_psnr": [], "clean_ssim": []},
        "noise_aware": {"global_psnr": [], "global_ssim": [],
                        "noise_psnr": [], "noise_ssim": [],
                        "clean_psnr": [], "clean_ssim": []},
    }
    coverages = []
    per_view_results = []
    skipped = 0

    for i in range(n_files):
        gt_path = gt_dir / gt_files[i]
        baseline_path = baseline_dir / baseline_files[i]
        noise_aware_path = noise_aware_dir / noise_aware_files[i]

        # mask 찾기: index→name 매핑 사용
        mask_path = None
        image_name = None
        if idx_to_name and i in idx_to_name:
            image_name = idx_to_name[i]
            mask_path = find_mask_file(mask_dir, image_name)

        # fallback: GT 파일명으로 직접 시도
        if mask_path is None:
            mask_path = find_mask_file(mask_dir, gt_files[i])
            image_name = gt_files[i]

        if mask_path is None:
            skipped += 1
            continue

        # 이미지 로드
        gt = load_image(str(gt_path))
        baseline = load_image(str(baseline_path))
        noise_aware = load_image(str(noise_aware_path))

        # mask 로드
        noise_mask = load_mask(mask_path, args.threshold)

        # 해상도 맞추기
        if noise_mask.shape[0] != gt.shape[0] or noise_mask.shape[1] != gt.shape[1]:
            print(f"[INFO] Resizing mask {noise_mask.shape} → {gt.shape[:2]} for {image_name}")
            mask_pil = Image.fromarray((noise_mask.astype(np.float32) * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            noise_mask = np.array(mask_pil, dtype=np.float32) / 255.0 >= args.threshold

        clean_mask = ~noise_mask
        coverage = noise_mask.mean()
        coverages.append(coverage)

        view_result = {"index": i, "image_name": image_name, "coverage": coverage}

        for method_name, render_img in [("baseline", baseline), ("noise_aware", noise_aware)]:
            gp = global_psnr(gt, render_img)
            gs = global_ssim(gt, render_img)
            np_ = masked_psnr(gt, render_img, noise_mask)
            ns = masked_ssim(gt, render_img, noise_mask)
            cp = masked_psnr(gt, render_img, clean_mask)
            cs = masked_ssim(gt, render_img, clean_mask)

            results[method_name]["global_psnr"].append(gp)
            results[method_name]["global_ssim"].append(gs)
            results[method_name]["noise_psnr"].append(np_)
            results[method_name]["noise_ssim"].append(ns)
            results[method_name]["clean_psnr"].append(cp)
            results[method_name]["clean_ssim"].append(cs)

            view_result[f"{method_name}_global_psnr"] = gp
            view_result[f"{method_name}_noise_psnr"] = np_
            view_result[f"{method_name}_clean_psnr"] = cp

        per_view_results.append(view_result)

    # 평균 계산 (nan 제외)
    def nanmean(lst):
        arr = np.array(lst)
        valid = arr[~np.isnan(arr)]
        return valid.mean() if len(valid) > 0 else float("nan")

    n_evaluated = len(coverages)
    avg_coverage = np.mean(coverages) if coverages else 0

    print()
    print("=" * 72)
    print("Local Metric Evaluation Results")
    print("=" * 72)
    print(f"  Views evaluated: {n_evaluated} (skipped: {skipped})")
    print(f"  Average noise coverage: {avg_coverage*100:.2f}%")
    print(f"  Mask threshold: {args.threshold}")
    print()

    # PSNR 테이블
    bm = results["baseline"]
    nm = results["noise_aware"]

    b_np = nanmean(bm["noise_psnr"])
    b_cp = nanmean(bm["clean_psnr"])
    b_gp = nanmean(bm["global_psnr"])
    n_np = nanmean(nm["noise_psnr"])
    n_cp = nanmean(nm["clean_psnr"])
    n_gp = nanmean(nm["global_psnr"])

    print("PSNR (dB)")
    print("-" * 72)
    print(f"{'':20s} | {'Noise Region':>14s} | {'Clean Region':>14s} | {'Global':>14s}")
    print("-" * 72)
    print(f"{'Baseline':20s} | {b_np:>14.4f} | {b_cp:>14.4f} | {b_gp:>14.4f}")
    print(f"{'Noise-aware':20s} | {n_np:>14.4f} | {n_cp:>14.4f} | {n_gp:>14.4f}")
    d_np = n_np - b_np
    d_cp = n_cp - b_cp
    d_gp = n_gp - b_gp
    print(f"{'Delta':20s} | {d_np:>+14.4f} | {d_cp:>+14.4f} | {d_gp:>+14.4f}")
    print()

    # SSIM 테이블
    b_ns = nanmean(bm["noise_ssim"])
    b_cs = nanmean(bm["clean_ssim"])
    b_gs = nanmean(bm["global_ssim"])
    n_ns = nanmean(nm["noise_ssim"])
    n_cs = nanmean(nm["clean_ssim"])
    n_gs = nanmean(nm["global_ssim"])

    print("SSIM")
    print("-" * 72)
    print(f"{'':20s} | {'Noise Region':>14s} | {'Clean Region':>14s} | {'Global':>14s}")
    print("-" * 72)
    print(f"{'Baseline':20s} | {b_ns:>14.4f} | {b_cs:>14.4f} | {b_gs:>14.4f}")
    print(f"{'Noise-aware':20s} | {n_ns:>14.4f} | {n_cs:>14.4f} | {n_gs:>14.4f}")
    d_ns = n_ns - b_ns
    d_cs = n_cs - b_cs
    d_gs = n_gs - b_gs
    print(f"{'Delta':20s} | {d_ns:>+14.4f} | {d_cs:>+14.4f} | {d_gs:>+14.4f}")
    print()
    print("=" * 72)

    # CSV 저장
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index", "image_name", "coverage",
                "baseline_global_psnr", "baseline_noise_psnr", "baseline_clean_psnr",
                "noise_aware_global_psnr", "noise_aware_noise_psnr", "noise_aware_clean_psnr",
            ])
            for r in per_view_results:
                writer.writerow([
                    r["index"], r["image_name"], f"{r['coverage']:.6f}",
                    f"{r['baseline_global_psnr']:.4f}",
                    f"{r['baseline_noise_psnr']:.4f}",
                    f"{r['baseline_clean_psnr']:.4f}",
                    f"{r['noise_aware_global_psnr']:.4f}",
                    f"{r['noise_aware_noise_psnr']:.4f}",
                    f"{r['noise_aware_clean_psnr']:.4f}",
                ])
        print(f"Per-view results saved to: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Local Metric Evaluation: noise vs clean region")
    parser.add_argument("--gt_dir", required=True, type=str,
                        help="GT 이미지 디렉토리 (e.g., output/.../train/ours_10000/gt)")
    parser.add_argument("--baseline_render_dir", required=True, type=str,
                        help="Baseline 렌더링 디렉토리 (e.g., output/.../train/ours_10000/renders)")
    parser.add_argument("--noise_aware_render_dir", required=True, type=str,
                        help="Noise-aware 렌더링 디렉토리")
    parser.add_argument("--mask_dir", required=True, type=str,
                        help="Contribution mask 디렉토리")
    parser.add_argument("--cameras_json", type=str, default=None,
                        help="cameras.json 경로 (index→image_name 매핑용)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"],
                        help="train/test split (LLFF hold 기준, default=train)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Mask 이진화 threshold (default=0.5)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Per-view 결과 CSV 저장 경로")
    args = parser.parse_args()
    evaluate(args)
