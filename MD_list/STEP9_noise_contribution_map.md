# Step 9: Noise Contribution Map 생성 및 Noise-Aware Loss 적용

## 목표

3D noise gaussian 정보를 활용하여 각 training view에서 **noise gaussian이 실제로 pixel 색상에 기여하는 정도**를 정확히 측정하고, 이를 기반으로 noise-aware loss를 적용하여 재학습 품질을 개선한다.

---

## 배경: 기존 접근의 문제점

### 1. 과거 Reprojection (삭제됨) — 근본적 결함

noise gaussian만 별도로 렌더링하여 alpha 누적값을 mask로 사용:

```python
# 과거 방식 (잘못됨)
noise_image = render(noise_gaussians_only, camera, bg=black)
mask = noise_image > threshold
```

**문제**: clean gaussian이 없는 상태에서 렌더링하므로 **occlusion(가림)을 고려하지 않음**.
- 실제로는 clean gaussian 뒤에 가려져 기여가 없는 noise gaussian도 높은 alpha로 나옴
- transmittance T_i = Π_{j<i}(1-α_j)에서 clean gaussian이 빠지면 T_noise ≈ 1.0
- 결과: t=0.01 → 95%, t=0.7 → 20% 과도한 커버리지

### 2. dataset_pairs (step4) — noise gaussian 정보 미사용

```python
# step4 방식
mask = (0.5 * norm(L1_error) + 0.5 * norm(LPIPS_error)) > 0.15
```

- rendered vs GT 이미지 비교로 error map 생성
- noise gaussian의 3D 위치 정보와 무관
- "렌더링 에러가 큰 영역" ≠ "noise gaussian이 기여하는 영역"

### 3. SSIM gradient 혼합 문제

SSIM은 11×11 gaussian-weighted window (sigma=1.5) 기반:
- noise pixel이 window 내에 있으면 clean pixel의 gradient도 오염
- 반경 3px 내 영향: gaussian weight = 13.5% (무시 불가)
- 단순 `ssim_map * (1-mask)`로는 backward convolution gradient 전파를 막지 못함

---

## 새로운 접근: Full Render vs Clean Render 차분

### 핵심 아이디어

동일한 camera에서 **전체 모델**과 **clean 모델**을 각각 렌더링하여 차이를 구함.

```python
full_image  = render(all_gaussians, camera)    # noise + clean (baseline 30k)
clean_image = render(clean_gaussians, camera)  # clean only

noise_contribution = |full_image - clean_image|  # pixel별 noise 기여도
```

### 왜 정확한가

3DGS 렌더링은 front-to-back alpha compositing:

```
C(pixel) = Σ_i (color_i × α_i × T_i)
T_i = Π_{j<i} (1 - α_j)    ← 앞의 모든 gaussian에 의한 투과율
```

- **full render**: noise + clean 모든 gaussian 참여 → 실제 training GT와 유사
- **clean render**: clean gaussian만 참여 → noise 기여 없는 렌더링
- **차이**: noise gaussian이 **occlusion을 고려하여** 실제로 pixel에 기여한 양

예시:
- noise gaussian이 clean gaussian 뒤에 있음 → T_noise ≈ 0 → 차이 ≈ 0 (정확)
- noise gaussian이 앞에 있어 실제 pixel에 영향 → 차이가 큼 (정확)
- noise gaussian이 빈 공간에 떠있음 → 차이가 큼 (정확)

---

## 구현 계획

### Phase 1: Noise Contribution Map 생성 스크립트

**파일**: `src/step9_noise_contribution_map.py`

```
입력:
  - data/360_v2/{scene}/               (학습 데이터, 카메라 정보)
  - output/{scene}/point_cloud/iteration_30000/point_cloud.ply  (full model)
  - data/processed_eval_v5/{scene}/noise_gaussians/clean_gaussians.ply

출력:
  - data/processed_eval_v5/{scene}/noise_gaussians/contribution_masks/
    ├── {image_name}.png              (binary mask, 0=clean, 255=noise)
    ├── {image_name}_heatmap.png      (continuous heatmap, 시각화용)
    └── metadata.json                 (통계 정보)
```

#### 주요 로직

```python
def compute_noise_contribution_map(full_model, clean_model, camera, threshold):
    """
    Args:
        full_model: 전체 gaussian (baseline 30k)
        clean_model: clean gaussian만
        camera: training view camera
        threshold: binary mask 변환 기준

    Returns:
        contribution: [3, H, W] per-channel 차이
        mask: [H, W] binary noise mask
    """
    bg = torch.zeros(3, device="cuda")
    pipeline = SimplePipeline()

    with torch.no_grad():
        full_image = render(camera, full_model, pipeline, bg)["render"]
        clean_image = render(camera, clean_model, pipeline, bg)["render"]

    # Per-pixel noise contribution (RGB 채널 평균)
    contribution = (full_image - clean_image).abs()  # [3, H, W]
    contribution_gray = contribution.mean(dim=0)      # [H, W]

    # Binary mask
    mask = (contribution_gray > threshold).float()

    return contribution, mask
```

#### 고려사항

1. **SH degree 일치**: full model과 clean model의 SH degree가 같아야 함
2. **배경색 통일**: 둘 다 동일한 bg color 사용
3. **exposure correction**: train_test_exp 사용 시 둘 다 동일하게 적용
4. **threshold 선정**:
   - per-view adaptive threshold (mean + k*std) 권장
   - 또는 고정 threshold (실험적 결정, 예: 0.01~0.05)

### Phase 2: noise_loader.py에 contribution mask 로딩 추가

```python
# NoiseDataLoader에 contribution_masks/ 탐색 추가
self._contribution_masks_dir = self.noise_data_dir / "contribution_masks"

# load_noise_mask_for_view()에서 contribution mask 우선 로딩
# Priority: contribution_masks > dataset_pairs > on-the-fly detector
```

### Phase 3: fused-ssim CUDA 커널 수정 (weighted SSIM)

#### 현재 fused-ssim 구조 분석

**파일**: `submodules/fused-ssim/ssim.cu`

현재 SSIM은 고정 11×11 gaussian kernel로 separable convolution 수행:

```
Forward (fusedssimCUDA):
  μ₁ = G ⊛ img1                  ← separable conv (x → y)
  σ₁² = G ⊛ img1² - μ₁²
  μ₂ = G ⊛ img2
  σ₂² = G ⊛ img2² - μ₂²
  σ₁₂ = G ⊛ (img1·img2) - μ₁·μ₂
  ssim_map = (2μ₁μ₂+C1)(2σ₁₂+C2) / (μ₁²+μ₂²+C1)(σ₁²+σ₂²+C2)

Backward (fusedssim_backwardCUDA):
  dL/dimg1 += G ⊛ (dL/dmap · ∂m/∂μ₁)          ← mu1 gradient
  dL/dimg1 += 2·img1 · G ⊛ (dL/dmap · ∂m/∂σ₁²) ← sigma1_sq gradient
  dL/dimg1 += img2 · G ⊛ (dL/dmap · ∂m/∂σ₁₂)   ← sigma12 gradient
```

kernel weight G_00~G_10은 `#define`으로 고정 (하드코딩).

#### 수정 방향: per-pixel weight를 gaussian kernel에 곱하기

noise mask `w[y][x]`를 추가 입력으로 받아, convolution 시 gaussian kernel에 곱함:

```
현재:  μ₁(p) = Σ G(i,j) · img1(p+i,p+j)
수정:  μ₁(p) = Σ G(i,j) · w(p+i,p+j) · img1(p+i,p+j) / Σ G(i,j) · w(p+i,p+j)
```

noise pixel은 w=0.7, clean pixel은 w=1.0. 이렇게 하면:
- **clean pixel의 SSIM window**: noise pixel의 기여가 30% 감소 → gradient 오염 감소
- **noise pixel의 SSIM window**: noise pixel 자체 기여도 감소 → noise 학습 약화
- **순수 clean 영역**: w=1.0이므로 변화 없음

#### CUDA 수정 상세

**1. 함수 시그니처 변경**

```cuda
// Forward
__global__ void fusedssimCUDA(
  int H, int W, int CH,
  float C1, float C2,
  float* img1, float* img2,
  float* weight_map,        // NEW: [H, W] per-pixel weight (1.0=clean, 0.7=noise)
  float* ssim_map,
  float* dm_dmu1, float* dm_dsigma1_sq, float* dm_dsigma12
)

// Backward
__global__ void fusedssim_backwardCUDA(
  int H, int W, int CH,
  float C1, float C2,
  float* img1, float* img2,
  float* weight_map,        // NEW
  float* dL_dmap, float* dL_dimg1,
  float* dm_dmu1, float* dm_dsigma1_sq, float* dm_dsigma12
)
```

**2. Weighted separable convolution 구현**

weight_map을 shared memory에 로드하고, convolution 시 gaussian kernel에 곱함:

```cuda
// do_weighted_separable_conv_x: 기존 conv에 weight를 곱함
__device__ void do_weighted_separable_conv_x(
  float pixels[SY][SSX],
  float weights[SY][SSX],    // NEW: per-pixel weight in shared mem
  float opt[CY][CCX],
  float wsum[CY][CCX],       // NEW: weight sum for normalization
  int H, int W, bool sq = false
) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x + 5;
  float val = 0.0f;
  float ws = 0.0f;

  // 각 위치에서 G(k) * w(k) * pixel(k) 계산
  // 동시에 G(k) * w(k) 합산 (정규화용)
  if (sq) {
    val += G_00 * weights[local_y][local_x-5] * do_sq(pixels[local_y][local_x-5]);
    ws  += G_00 * weights[local_y][local_x-5];
    val += G_01 * weights[local_y][local_x-4] * do_sq(pixels[local_y][local_x-4]);
    ws  += G_01 * weights[local_y][local_x-4];
    // ... G_02 ~ G_10 동일 패턴
  } else {
    val += G_00 * weights[local_y][local_x-5] * pixels[local_y][local_x-5];
    ws  += G_00 * weights[local_y][local_x-5];
    val += G_01 * weights[local_y][local_x-4] * pixels[local_y][local_x-4];
    ws  += G_01 * weights[local_y][local_x-4];
    // ... G_02 ~ G_10 동일 패턴
  }
  opt[local_y][local_x] = val;
  wsum[local_y][local_x] = ws;

  // 2nd row (local_y + BY) 처리 동일
}
```

**주의**: separable convolution이므로 x방향 → y방향 순서로 수행.
weight도 2D separable로 분리 가능한지 확인 필요.
- weight_map은 임의 값이므로 separable하지 않음
- **대안**: weight_map을 x/y 각 방향에서 독립적으로 적용하거나,
  또는 2D convolution으로 변경 (성능 저하 감수)
- **실용적 대안**: separable 구조 유지하되, weight를 x conv 시에만 적용하고
  y conv는 기존 그대로. 이 경우 근사치이지만 성능 유지.

**3. 정규화 처리**

weighted mean이 올바르려면 weight sum으로 나누어야 함:

```cuda
// μ₁ = (G·w ⊛ img1) / (G·w ⊛ 1)
float mu1_unnorm = do_weighted_conv(img1, weight_map);
float w_sum = do_weighted_conv(ones, weight_map);  // 또는 weight sum만 계산
float mu1 = mu1_unnorm / (w_sum + 1e-7f);
```

**4. Backward도 동일하게 수정**

backward의 convolution도 weight_map을 곱한 형태로 수정:

```cuda
// gradient from mu1: 기존 G ⊛ (dL/dmap · ∂m/∂μ₁)
// 수정:              G·w ⊛ (dL/dmap · ∂m/∂μ₁) / w_sum
```

**5. Python 인터페이스 변경**

```python
# fused_ssim/__init__.py
class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, weight_map=None, padding="same", train=True):
        if weight_map is None:
            weight_map = torch.ones(1, 1, img1.shape[2], img1.shape[3], device=img1.device)
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(
            C1, C2, img1, img2, weight_map, train
        )
        ctx.save_for_backward(img1.detach(), img2, weight_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ...

def fused_ssim(img1, img2, weight_map=None, padding="same", train=True):
    """
    Args:
        weight_map: [1, 1, H, W] per-pixel weight. 1.0=clean, 0.7=noise.
                    None이면 기존 동작과 동일 (모든 pixel weight=1.0)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    map = FusedSSIMMap.apply(C1, C2, img1, img2, weight_map, padding, train)
    return map.mean()
```

#### 대안: Non-separable 2D convolution

separable convolution에서 per-pixel weight를 정확히 적용하는 것은 수학적으로 불가능
(weight가 separable하지 않으므로). 두 가지 선택:

| 방법 | 정확도 | 성능 | 난이도 |
|------|--------|------|--------|
| x-conv에만 weight 적용 (근사) | △ | 기존과 동일 | 중 |
| 2D convolution으로 변경 (정확) | ◎ | ~2x 느림 | 상 |
| weight를 sqrt 분해 후 각 방향 적용 | △ | 기존과 동일 | 중 |

**권장**: 우선 x-conv에만 weight 적용하는 근사 방식으로 구현 후 효과 확인.
효과가 있으면 2D convolution 정확 버전으로 업그레이드.

### Phase 4: train.py에 noise-aware loss 적용

```python
if noise_mask is not None:
    noise_mask = noise_mask.to(image.device)

    # L1: noise 영역 weight 감소 (pixel-wise, 혼합 문제 없음)
    weight_map = 1.0 - noise_mask * (1.0 - noise_loss_weight)
    Ll1 = (torch.abs(image - gt_image) * weight_map.unsqueeze(0)).mean()

    # SSIM: CUDA 커널에서 noise/clean 분리 계산
    # weight_map을 [1, 1, H, W]로 변환하여 fused_ssim에 전달
    ssim_weight = weight_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    ssim_value = fused_ssim(
        image.unsqueeze(0), gt_image.unsqueeze(0),
        weight_map=ssim_weight
    )
else:
    Ll1 = l1_loss(image, gt_image)
    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

loss = 0.8 * Ll1 + 0.2 * (1.0 - ssim_value)
```

#### 왜 CUDA 커널 수정이 GT 마스킹보다 우수한가

```
GT 마스킹 (우회):
  - noise GT를 rendered로 교체 → SSIM window 내 오염 간접 감소
  - 한계: 교체 경계에서 불연속 발생, noise 영역의 SSIM loss 기여 = 0 (조절 불가)

CUDA weighted SSIM (직접):
  - gaussian kernel 자체에 per-pixel weight 적용
  - noise pixel의 window 내 기여가 weight에 비례하여 정확히 감소
  - weight=0.7이면 noise pixel이 mean/variance/covariance에 70%만 기여
  - clean pixel의 gradient가 noise에 의해 오염되는 양도 정확히 70%로 감소
  - weight 값으로 연속적 조절 가능 (0.0~1.0)
```

---

## 실험 계획

### 실행 순서

```bash
# 1. Noise Contribution Map 생성
python src/step9_noise_contribution_map.py \
    --scene stump \
    --source_path data/360_v2/stump \
    --full_model output/stump_baseline_eval_10k \
    --clean_ply data/processed_eval_v5/stump/noise_gaussians/clean_gaussians.ply \
    --noise_data_dir data/processed_eval_v5/stump/noise_gaussians \
    --threshold 0.02 \
    --eval

# 2. 커버리지 확인 (적정 범위: 1~10%)
cat data/processed_eval_v5/stump/noise_gaussians/contribution_masks/metadata.json

# 3. Noise-aware retraining
python train.py -s data/360_v2/stump \
    -m output/stump_noise_aware_10k \
    --iterations 10000 \
    --load_ply data/processed_eval_v5/stump/noise_gaussians/clean_gaussians.ply \
    --noise_data_dir data/processed_eval_v5/stump/noise_gaussians \
    --noise_loss_weight 0.7 \
    --test_iterations 7000 10000 \
    --save_iterations 7000 10000 \
    --eval
```

### 비교 대상

| 실험 | 시작점 | Mask | Loss | 명칭 |
|------|--------|------|------|------|
| A | baseline 30k | 없음 | vanilla | stump_baseline_eval_10k (기존, PSNR 26.820) |
| B | clean_gaussians.ply | 없음 | vanilla | stump_nwl_w07_v5 (기존, PSNR 26.890) |
| **C** | **clean_gaussians.ply** | **contribution map** | **noise-aware** | **stump_noise_aware_10k (NEW)** |

### 기대 결과

- 실험 C > 실험 B: contribution map + noise-aware loss의 추가 효과 확인
- 실험 C > 실험 A: clean start + masking의 복합 효과 확인

### 주의: 평가 패러독스

- Test GT에도 noise 포함 → noise를 재현하지 않으면 metric이 하락할 수 있음
- 시각적 품질 vs metric 간 괴리 가능
- SIBR viewer에서의 시각적 비교가 중요

---

## 디렉토리 구조 (완료 후)

```
data/processed_eval_v5/stump/noise_gaussians/
├── noise_gaussians.json
├── noise_gaussians.ply
├── clean_gaussians.ply
├── cameras.json
├── contribution_masks/          ← NEW (Phase 1)
│   ├── _DSC9213.png
│   ├── _DSC9213_heatmap.png
│   ├── _DSC9214.png
│   ├── ...
│   └── metadata.json
├── cluster_0..9/
└── visualization/
```

---

## 체크리스트

### 구현
- [x] `src/step9_noise_contribution_map.py` 작성 (Phase 1)
- [x] `src/utils/noise_loader.py` 수정 — contribution_masks + ssim_weight 로딩 (Phase 2)
- [x] `submodules/fused-ssim/ssim.cu` 수정 — weighted SSIM forward/backward (Phase 3)
- [x] `submodules/fused-ssim/ssim.h` 수정 — weight_map 파라미터 추가 (Phase 3)
- [x] `submodules/fused-ssim/fused_ssim/__init__.py` 수정 — weight_map 인터페이스 (Phase 3)
- [ ] fused-ssim 재빌드 (`pip install submodules/fused-ssim`) — 서버에서 실행 필요
- [x] `train.py` 수정 — noise-aware loss + weighted SSIM 호출 (Phase 4)

### 실험
- [ ] 서버에 코드 동기화 (`rsync`)
- [ ] fused-ssim 재빌드 (서버)
- [ ] Contribution map 생성 및 커버리지 확인 (적정: 1~10%)
- [ ] Threshold 탐색 (0.01, 0.02, 0.05)
- [ ] Weighted SSIM 단위 테스트 (weight=1.0일 때 기존과 동일한지)
- [ ] Noise-aware retraining 실행
- [ ] render.py + metrics.py 평가
- [ ] SIBR viewer 시각적 비교

---

Last Updated: 2026-02-23
Status: 코드 구현 완료, 서버 빌드 및 실험 대기