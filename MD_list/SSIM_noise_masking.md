# SSIM Loss의 노이즈 간섭 문제 및 해결

## 1. 문제 발견

Exp-B (noise-weighted loss, w=0.7)를 processed_eval_v5 (24,505 noise, 0.60%)로 학습한 결과:

| 모델 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| baseline_eval_10k | 26.820 | 0.7745 | 0.2338 |
| nwl_w07_v5 | 26.890 | 0.7774 | 0.2321 |

- 노이즈 영역: 표현력 향상 확인
- **디테일 영역: 기존보다 저하됨**
- 수치 개선폭 미미 (PSNR +0.07)

## 2. 원인 분석: SSIM Loss가 노이즈 마스킹 없이 동작

### 현재 Exp-B의 loss 구조

```python
# L1: noise-weighted ✓
weight_map = 1.0 - noise_mask * (1.0 - 0.7)  # noise pixel → 0.7
Ll1 = (torch.abs(image - gt_image) * weight_map).mean()

# SSIM: unmasked ✗
ssim_value = fused_ssim(image, gt_image)  # 노이즈 GT를 그대로 사용!

# 최종 loss (lambda_dssim=0.2)
loss = 0.8 * Ll1 + 0.2 * (1 - ssim_value)
```

L1 loss만 가중치 적용, **SSIM loss는 원본 GT를 그대로 사용**.

### SSIM의 11×11 윈도우 문제

SSIM은 per-pixel이 아니라 **11×11 gaussian window (sigma=1.5)**로 패치 통계를 계산:

```python
# utils/loss_utils.py
mu1 = F.conv2d(img1, window, padding=5, groups=channel)  # 11x11 블러
sigma12 = F.conv2d(img1 * img2, window, ...) - mu1_mu2    # 교차상관
ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / (...)
```

노이즈 픽셀이 패치 안에 있으면 mu, sigma, sigma12 모든 통계가 오염됨.

### 정량적 영향

| 거리 | Gaussian 가중치 | 영향도 |
|------|----------------|--------|
| 0 (중심) | 0.2660 (100%) | 매우 강함 |
| 1 | 0.2130 (80%) | 강함 |
| 2 | 0.1094 (41%) | 큼 |
| 3 | 0.0360 (13.5%) | **무시 불가** |
| 4 | 0.0076 (2.9%) | 미미 |
| 5 (경계) | 0.0010 (0.4%) | 무시 가능 |

- 단일 노이즈 픽셀(distance 3)의 SSIM loss 변화: **+0.949 (95% 증가)**
- 분산 변화: **+111,235,553%**

### 노이즈 영역에서의 loss 기여도

| Loss 구성 | 비중 | 마스킹 | 실제 기여 |
|-----------|------|--------|----------|
| L1 (masked) | 80% | O (w=0.7) | **17.4%** |
| SSIM (unmasked) | 20% | **X** | **82.6%** |

→ **노이즈 영역에서 loss의 82.6%가 SSIM을 통해 노이즈 재현을 강제**

### 디테일 저하 메커니즘

1. SSIM의 backward pass가 11×11 윈도우 내 **모든 121 픽셀에 gradient 전파**
2. 노이즈 근처의 clean 픽셀도 "노이즈 패턴을 재현하라"는 gradient를 받음
3. L1에서 noise를 억제해도 SSIM이 이를 상쇄
4. 결과: clean 영역의 디테일이 노이즈 방향으로 왜곡

## 3. 해결 방법: GT 마스킹

### 핵심 아이디어

SSIM 계산 전에 GT 이미지의 **노이즈 영역을 현재 렌더링 결과로 대체**:

```python
gt_for_ssim = gt_image * (1 - noise_mask) + image.detach() * noise_mask
ssim_value = fused_ssim(image, gt_for_ssim)
```

### 동작 원리

- **노이즈 영역**: GT = rendered (동일) → SSIM = 1.0 → gradient = 0 → 노이즈 학습 완전 차단
- **clean 영역**: GT 원본 유지 → 정상 SSIM 학습
- **경계 영역**: 11×11 윈도우가 걸쳐도 noise 쪽은 rendered=GT이므로 통계 오염 없음

### 장점

1. SSIM 내부 구현 수정 불필요 (fused_ssim, python ssim 모두 호환)
2. `image.detach()` 사용으로 노이즈 영역에서 gradient 차단
3. 구현 3줄 추가로 완료
4. `--noise_data_dir` 미지정 시 기존 동작과 100% 동일 (backward compatible)

### 수정 코드

```python
# train.py (exp/noise-weighted-loss)

if noise_mask is not None:
    noise_mask = noise_mask.to(image.device)
    # L1: 기존 noise-weighted 유지
    weight_map = 1.0 - noise_mask * (1.0 - noise_loss_weight)
    weight_map = weight_map.unsqueeze(0)
    Ll1 = (torch.abs(image - gt_image) * weight_map).mean()

    # [NEW] SSIM GT 마스킹
    noise_mask_3ch = noise_mask.unsqueeze(0)  # [1, H, W]
    gt_for_ssim = gt_image * (1 - noise_mask_3ch) + image.detach() * noise_mask_3ch
else:
    Ll1 = l1_loss(image, gt_image)
    gt_for_ssim = gt_image

# SSIM: gt_for_ssim 사용 (노이즈 영역 = rendered)
if FUSED_SSIM_AVAILABLE:
    ssim_value = fused_ssim(image.unsqueeze(0), gt_for_ssim.unsqueeze(0))
else:
    ssim_value = ssim(image, gt_for_ssim)
```

## 4. 검증 계획

### 비교 대상
1. `stump_baseline_eval_10k` - baseline (30K + 10K eval)
2. `stump_nwl_w07_v5` - L1만 masked (기존 exp-b)
3. `stump_nwl_w07_v5_ssim_fix` - L1 + SSIM masked (개선)

### 기대 결과
- 노이즈 영역 개선 유지 (L1 masking 효과)
- 디테일 영역 저하 해소 (SSIM 노이즈 gradient 차단)
- PSNR/SSIM/LPIPS 전반적 개선

### 실행 명령어
```bash
python train.py -s data/360_v2/stump \
    -m output/stump_nwl_w07_v5_ssim_fix \
    --iterations 10000 \
    --load_ply data/processed_eval_v5/stump/noise_gaussians/clean_gaussians.ply \
    --noise_data_dir data/processed_eval_v5/stump/noise_gaussians \
    --noise_loss_weight 0.7 \
    --test_iterations 7000 10000 \
    --save_iterations 7000 10000 \
    --eval
```
