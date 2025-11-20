#!/usr/bin/env python3
"""
Threshold 분석 스크립트: combined_error의 분포를 확인
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 샘플 데이터 로드
pair_dir = Path("data/processed/stump/rendered_views")
rendered = np.array(Image.open(pair_dir / "render_0000.png").convert('RGB'))
gt = np.array(Image.open(pair_dir / "gt_0000.png").convert('RGB'))

# L1 error 계산
pixel_error = np.abs(rendered.astype(np.float32) - gt.astype(np.float32)).mean(axis=-1) / 255.0

# Normalize
def normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-7:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

pixel_error_norm = normalize(pixel_error)

# LPIPS는 생략 (L1만으로도 분포 확인 가능)
# combined_error ≈ pixel_error_norm (LPIPS가 균일한 값이므로)

# 통계
print("=" * 60)
print("Combined Error 분포 분석")
print("=" * 60)
print(f"최소값: {pixel_error_norm.min():.6f}")
print(f"최대값: {pixel_error_norm.max():.6f}")
print(f"평균:   {pixel_error_norm.mean():.6f}")
print(f"중앙값: {np.median(pixel_error_norm):.6f}")
print(f"표준편차: {pixel_error_norm.std():.6f}")
print()

# Percentile 분석
percentiles = [50, 75, 90, 95, 99]
print("Percentile 분석:")
for p in percentiles:
    val = np.percentile(pixel_error_norm, p)
    print(f"  {p}%: {val:.6f} (threshold={val:.3f}로 설정 시 상위 {100-p}%가 노이즈)")

# 다양한 threshold에서의 노이즈 비율
print("\nThreshold별 노이즈 비율:")
thresholds = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
for thresh in thresholds:
    noise_ratio = (pixel_error_norm > thresh).mean()
    print(f"  threshold={thresh:.2f} → 노이즈 비율: {noise_ratio*100:.2f}%")

# Histogram 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(pixel_error_norm.flatten(), bins=100, edgecolor='black', alpha=0.7)
plt.axvline(0.03, color='r', linestyle='--', label='Current threshold=0.03')
plt.axvline(0.15, color='g', linestyle='--', label='Suggested threshold=0.15')
plt.xlabel('Combined Error')
plt.ylabel('Pixel Count')
plt.title('Combined Error Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.imshow(pixel_error_norm, cmap='hot')
plt.colorbar(label='Combined Error')
plt.title('Error Map')

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150)
print("\n✓ 시각화 저장: threshold_analysis.png")
