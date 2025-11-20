#!/usr/bin/env python3
"""
Step 4: Error Map & Dataset Pairs Generation

입력:
  - data/processed/{scene_name}/rendered_views/ (rendered, gt 이미지)

출력:
  - data/processed/{scene_name}/dataset_pairs/
    ├── train/
    ├── val/
    └── test/
"""

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import argparse
import sys

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")
    print("Falling back to L1-only mode.")


class ErrorMapGenerator:
    """
    렌더링 이미지와 GT를 비교하여 error map (noise mask) 생성
    """

    def __init__(self, threshold=0.03, min_size=50, device='cuda', use_lpips=True):
        self.threshold = threshold
        self.min_size = min_size
        self.device = device
        self.use_lpips = use_lpips and LPIPS_AVAILABLE

        # LPIPS 모델 초기화 (spatial=True로 pixel-wise map 받기)
        if self.use_lpips:
            print(f"LPIPS 모델 초기화 중 (spatial mode, device: {device})...")
            self.lpips_model = lpips.LPIPS(net='alex', spatial=True).to(device)
            self.lpips_model.eval()
            print("✓ LPIPS 모델 로드 완료 (pixel-wise mode)")
        else:
            self.lpips_model = None
            print("✓ L1-only mode (LPIPS 비활성화)")

    def generate_mask(self, rendered, gt):
        """
        Error map 생성 (L1 + LPIPS 조합).

        Args:
            rendered: 렌더링 이미지 [H, W, 3], numpy, [0, 255]
            gt: Ground truth [H, W, 3], numpy, [0, 255]

        Returns:
            mask: Binary noise mask [H, W], float32, {0, 1}

        Note:
            Option A (LPIPS 없음): pixel_error만 사용
            Option B (LPIPS 있음): 0.5 * normalize(L1) + 0.5 * normalize(LPIPS)
        """
        # 1. Pixel-wise L1 error (RGB 채널 평균, 0~1 범위)
        pixel_error = np.abs(rendered.astype(np.float32) -
                           gt.astype(np.float32)).mean(axis=-1) / 255.0

        if self.use_lpips and self.lpips_model is not None:
            # 2. LPIPS perceptual error (pixel-wise)
            lpips_error = self._compute_lpips_spatial(rendered, gt)

            # 3. 정규화 및 조합
            pixel_error_norm = self._normalize(pixel_error)
            lpips_error_norm = self._normalize(lpips_error)

            # Combined error (0.5 * L1 + 0.5 * LPIPS)
            combined_error = 0.5 * pixel_error_norm + 0.5 * lpips_error_norm

            # 4. Threshold 적용
            mask = (combined_error > self.threshold).astype(np.float32)
        else:
            # LPIPS 없이 L1만 사용
            mask = (pixel_error > self.threshold).astype(np.float32)

        # 5. 후처리 (작은 영역 제거)
        if self.min_size > 0:
            mask = self._remove_small_regions(mask)

        return mask

    def _compute_lpips_spatial(self, img1, img2):
        """
        Pixel-wise LPIPS 계산 (spatial map 반환).

        Args:
            img1, img2: [H, W, 3], numpy, [0, 255]

        Returns:
            lpips_map: [H, W], numpy, LPIPS error per pixel
        """
        def to_tensor(img):
            # [H, W, 3] -> [1, 3, H, W], normalized to [-1, 1]
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
            return img.to(self.device)

        t1 = to_tensor(img1)
        t2 = to_tensor(img2)

        with torch.no_grad():
            # spatial=True인 경우: [1, 1, H, W] 형태로 반환
            lpips_map = self.lpips_model(t1, t2)
            lpips_map = lpips_map.squeeze().cpu().numpy()  # [H, W]

        return lpips_map

    def _normalize(self, arr):
        """
        정규화 (0~1 범위로)

        Note: 이미지별로 정규화하므로 상대적인 에러 분포를 볼 수 있음
        """
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min < 1e-7:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    def _remove_small_regions(self, mask):
        """작은 영역 제거"""
        try:
            from scipy import ndimage
            labeled, _ = ndimage.label(mask)
            component_sizes = np.bincount(labeled.ravel())
            mask_sizes = component_sizes >= self.min_size
            mask_sizes[0] = 0  # Background
            cleaned_mask = mask_sizes[labeled]
            return cleaned_mask.astype(np.float32)
        except ImportError:
            print("Warning: scipy not available, skipping small region removal")
            return mask


def generate_all_pairs(rendered_dir, output_dir, error_generator,
                      train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    모든 데이터 쌍 생성 및 train/val/test split.

    Args:
        rendered_dir: 렌더링된 이미지 디렉토리
        output_dir: 출력 디렉토리
        error_generator: ErrorMapGenerator 인스턴스
        train_ratio: Train 비율
        val_ratio: Validation 비율
        test_ratio: Test 비율
        random_seed: Random seed
    """
    rendered_dir = Path(rendered_dir)

    # rendered 이미지 찾기
    rendered_files = sorted(rendered_dir.glob("render_*.png"))

    print(f"\n{len(rendered_files)}개 렌더링 이미지 발견")

    pairs = []

    for rendered_path in rendered_files:
        # 대응하는 GT 이미지 찾기
        idx_str = rendered_path.stem.split('_')[1]  # render_0000.png -> 0000
        gt_path = rendered_dir / f"gt_{idx_str}.png"

        if not gt_path.exists():
            print(f"  경고: GT 이미지 없음: {gt_path}")
            continue

        # 이미지 로드
        rendered = np.array(Image.open(rendered_path).convert('RGB'))
        gt = np.array(Image.open(gt_path).convert('RGB'))

        # Error map 생성
        mask = error_generator.generate_mask(rendered, gt)

        # 저장할 정보
        pair = {
            'view_index': int(idx_str),
            'rendered': rendered,
            'gt': gt,
            'mask': mask,
            'noise_ratio': float(mask.mean())
        }
        pairs.append(pair)

        if len(pairs) % 10 == 0:
            print(f"  [{len(pairs)}/{len(rendered_files)}] 처리 완료")

    print(f"\n✓ {len(pairs)}개 쌍 생성 완료")

    # Noise ratio 통계
    noise_ratios = [p['noise_ratio'] for p in pairs]
    print(f"\nNoise ratio 통계:")
    print(f"  평균: {np.mean(noise_ratios):.3f}")
    print(f"  최소: {np.min(noise_ratios):.3f}")
    print(f"  최대: {np.max(noise_ratios):.3f}")

    # Random split
    np.random.seed(random_seed)
    indices = np.random.permutation(len(pairs))

    n_train = int(len(pairs) * train_ratio)
    n_val = int(len(pairs) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # 저장
    save_pairs_with_split(pairs, train_indices, val_indices, test_indices, output_dir)


def save_pairs_with_split(pairs, train_indices, val_indices, test_indices, output_dir):
    """
    생성된 쌍을 train/val/test로 나누어 저장.

    Args:
        pairs: 데이터 쌍 리스트
        train_indices: Train 인덱스
        val_indices: Validation 인덱스
        test_indices: Test 인덱스
        output_dir: 출력 디렉토리
    """
    output_dir = Path(output_dir)

    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    for split_name, split_indices in splits.items():
        print(f"\n{split_name} split 저장 중 ({len(split_indices)}개)...")

        for i in split_indices:
            pair = pairs[i]

            # 저장 경로
            save_dir = output_dir / split_name / f"pair_{pair['view_index']:04d}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # 이미지 저장
            Image.fromarray(pair['rendered']).save(save_dir / "rendered.png")
            Image.fromarray(pair['gt']).save(save_dir / "gt.png")
            Image.fromarray((pair['mask'] * 255).astype(np.uint8)).save(
                save_dir / "mask.png"
            )

            # 메타데이터 저장
            metadata = {
                'view_index': pair['view_index'],
                'noise_ratio': pair['noise_ratio']
            }
            with open(save_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

    # 통계 출력
    print(f"\n저장 완료:")
    print(f"  Train: {len(train_indices)}개")
    print(f"  Val:   {len(val_indices)}개")
    print(f"  Test:  {len(test_indices)}개")
    print(f"  위치: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Step 4: Generate Dataset Pairs')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--output_root', default='./data/processed', help='Output root directory')
    parser.add_argument('--threshold', type=float, default=0.15,
                       help='Error threshold for noise detection (0~1 range, combined error)')
    parser.add_argument('--min_size', type=int, default=50,
                       help='Minimum noise region size (pixels)')
    parser.add_argument('--use_lpips', action='store_true', default=False,
                       help='Use LPIPS + L1 combined (default: L1 only)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: Error Map & Dataset Pairs Generation")
    print("=" * 60)
    print(f"Scene: {args.scene}")
    print(f"Mode: {'L1 + LPIPS (combined)' if args.use_lpips else 'L1 only'}")
    print(f"Threshold: {args.threshold}")
    print(f"Min size: {args.min_size}")
    print(f"Split: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print("=" * 60 + "\n")

    # 경로 설정
    rendered_dir = Path(args.output_root) / args.scene / "rendered_views"
    output_dir = Path(args.output_root) / args.scene / "dataset_pairs"

    # 검증
    if not rendered_dir.exists():
        raise FileNotFoundError(
            f"렌더링 이미지를 찾을 수 없습니다: {rendered_dir}\n"
            "먼저 Step 3을 실행하세요."
        )

    # Error map generator 생성
    error_generator = ErrorMapGenerator(
        threshold=args.threshold,
        min_size=args.min_size,
        device=args.device,
        use_lpips=args.use_lpips
    )

    # 데이터 쌍 생성
    generate_all_pairs(
        rendered_dir,
        output_dir,
        error_generator,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    print("\n" + "=" * 60)
    print("Step 4 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
