#!/usr/bin/env python3
"""
실제 에러 분포 확인 스크립트 (matplotlib 없이)
"""

import numpy as np
from PIL import Image
from pathlib import Path

# 샘플 이미지 로드
pair_dir = Path("data/processed/stump/rendered_views")
rendered_files = sorted(pair_dir.glob("render_*.png"))

print("="*60)
print("에러 분포 진단")
print("="*60)

if len(rendered_files) == 0:
    print("렌더링 이미지를 찾을 수 없습니다!")
    exit(1)

# 첫 5개 이미지 분석
for i, render_path in enumerate(rendered_files[:5]):
    idx_str = render_path.stem.split('_')[1]
    gt_path = pair_dir / f"gt_{idx_str}.png"

    if not gt_path.exists():
        continue

    rendered = np.array(Image.open(render_path).convert('RGB'))
    gt = np.array(Image.open(gt_path).convert('RGB'))

    # L1 error 계산 (정규화 전)
    pixel_error = np.abs(rendered.astype(np.float32) - gt.astype(np.float32)).mean(axis=-1) / 255.0

    print(f"\n이미지 {idx_str}:")
    print(f"  Pixel error (정규화 전):")
    print(f"    최소값: {pixel_error.min():.6f}")
    print(f"    최대값: {pixel_error.max():.6f}")
    print(f"    평균:   {pixel_error.mean():.6f}")
    print(f"    중앙값: {np.median(pixel_error):.6f}")
    print(f"    표준편차: {pixel_error.std():.6f}")

    # Percentiles
    p95 = np.percentile(pixel_error, 95)
    p99 = np.percentile(pixel_error, 99)
    print(f"    95 percentile: {p95:.6f}")
    print(f"    99 percentile: {p99:.6f}")

    # Normalize 후
    arr_min = pixel_error.min()
    arr_max = pixel_error.max()
    if arr_max - arr_min > 1e-7:
        pixel_error_norm = (pixel_error - arr_min) / (arr_max - arr_min)
    else:
        pixel_error_norm = np.zeros_like(pixel_error)

    print(f"  Pixel error (정규화 후):")
    print(f"    최소값: {pixel_error_norm.min():.6f}")
    print(f"    최대값: {pixel_error_norm.max():.6f}")
    print(f"    평균:   {pixel_error_norm.mean():.6f}")

    # Threshold별 노이즈 비율
    print(f"  Threshold별 노이즈 비율 (정규화 후):")
    for thresh in [0.1, 0.15, 0.2, 0.3, 0.5]:
        ratio = (pixel_error_norm > thresh).mean()
        print(f"    threshold={thresh:.2f} → {ratio*100:.2f}%")

print("\n" + "="*60)
print("진단 완료!")
print("="*60)
