# Step 8: Clean Initialization Retraining

## 개요

**목표**: Noise가 제거된 Clean PLY를 초기화로 사용하여 재학습함으로써 품질 개선 검증

**핵심 가설**: 
- Noise Gaussian이 제거된 상태에서 재학습하면, 빈 공간에 새로운 Gaussian이 올바른 위치에 생성됨
- Local minima에 갇혀있던 Noise Gaussian의 "예산"이 품질 개선에 재할당됨

---

## 현재 자산 (Step 1-7 완료)

```
output/stump/                           # Parent 모델 (125장 학습)
├── point_cloud/iteration_30000/
│   └── point_cloud.ply                 # 4,199,523 gaussians (994MB)

output/stump_clean_v2/                  # Noise 제거 버전
├── point_cloud/iteration_30000/
│   └── point_cloud.ply                 # 4,194,065 gaussians (992MB) - 5,458개 제거

output/stump_noise_only_v2/             # 제거된 Noise만
├── point_cloud/iteration_30000/
│   └── point_cloud.ply                 # 5,458 gaussians (1.3MB)

data/processed/stump/noise_gaussians/   # Step 7 최신 결과
├── clean_gaussians.ply                 # 4,196,740 gaussians (993MB) - 2,783개 제거
├── noise_gaussians.ply                 # 2,783 gaussians
└── noise_gaussians.json                # 통계 정보
```

---

## Step 8 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Step 8: Clean Retraining                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: 준비 (이미 완료)                                               │
│  ├── output/stump (Parent 30k) ─────────────────────┐                   │
│  │                                                   │                   │
│  Phase 2: Noise 제거 (이미 완료)                      │                   │
│  ├── Step 7 → clean_gaussians.ply ──────────────────┼──┐                │
│  │                                                   │  │                │
│  Phase 3: 재학습 (NEW)                               │  │                │
│  │   clean_gaussians.ply를 초기화로 사용              │  │                │
│  │   ├── Experiment A: 10k iterations               │  │                │
│  │   ├── Experiment B: 15k iterations               │  │                │
│  │   └── Experiment C: 30k iterations (full)        │  │                │
│  │                                                   │  │                │
│  Phase 4: 평가                                       │  │                │
│  │   ├── PSNR/SSIM/LPIPS on test views             │  │                │
│  │   ├── Novel view 렌더링 품질 비교                 │  │                │
│  │   └── Noise detection 재실행 (noise 감소 확인)    │  │                │
│  │                                                   │  │                │
└──┴───────────────────────────────────────────────────┴──┴────────────────┘
```

---

## 실험 설계

### 실험 그룹

| 실험 | 초기화 | Iterations | 목적 |
|------|--------|------------|------|
| Baseline | SfM → 30k | 30,000 | 기준선 (output/stump) |
| Exp A | Clean PLY → 10k | 10,000 | 최소 재학습 효과 |
| Exp B | Clean PLY → 15k | 15,000 | 중간 재학습 |
| Exp C | Clean PLY → 30k | 30,000 | 동등 조건 비교 |
| Exp D (공정비교) | SfM → 45k | 45,000 | 총 iteration 동등 |

### 핵심 비교

```
비교 1: Noise 제거 효과
  Baseline (30k) vs Exp A (Clean + 10k)
  → 적은 iteration으로도 품질 개선되는가?

비교 2: 동등 조건
  Baseline (30k) vs Exp C (Clean + 30k)
  → 같은 iteration에서 Clean 초기화가 더 나은가?

비교 3: 총 iteration 동등
  Exp C (30k + 30k = 60k total effort) vs Exp D (45k)
  → 추가 effort가 정당화되는가?
```

---

## 구현 계획

### 8.1 Clean PLY를 3DGS 초기화로 변환

**문제**: 3DGS train.py는 SfM point cloud 또는 random init만 지원

**해결책 A: 기존 checkpoint 구조 활용**
```python
# clean_gaussians.ply를 3DGS checkpoint 형식으로 변환
# output/stump_clean_init/ 생성

def create_clean_init_checkpoint(clean_ply_path, output_path):
    """
    Clean PLY를 3DGS가 로드할 수 있는 checkpoint 구조로 변환
    """
    # 1. PLY 로드
    gaussians = load_ply(clean_ply_path)
    
    # 2. checkpoint 구조 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 3. point_cloud/iteration_0/ 생성
    pc_dir = output_path / "point_cloud" / "iteration_0"
    pc_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. PLY 복사
    shutil.copy(clean_ply_path, pc_dir / "point_cloud.ply")
    
    # 5. cfg_args 생성 (원본에서 복사)
    # 6. cameras.json 복사 (원본에서)
```

**해결책 B: train.py 수정하여 PLY 초기화 지원**
```python
# train.py에 --init_ply 옵션 추가
parser.add_argument('--init_ply', type=str, default=None,
                    help='Initialize from existing PLY file')

# GaussianModel.create_from_ply() 호출
if args.init_ply:
    gaussians.load_ply(args.init_ply)
else:
    # 기존 SfM 초기화
```

### 8.2 재학습 스크립트

**파일**: `src/step8_clean_retraining.py`

```python
"""
Step 8: Clean Initialization Retraining

Clean PLY를 초기화로 사용하여 3DGS 재학습

Usage:
    python src/step8_clean_retraining.py \
        --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \
        --source_path data/360_v2/stump \
        --output_path output/stump_retrained \
        --iterations 15000
"""

import argparse
import subprocess
from pathlib import Path
import shutil

def prepare_init_checkpoint(clean_ply, source_model, output_dir):
    """
    Clean PLY를 초기 checkpoint로 준비
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # iteration_0 폴더 생성
    init_dir = output_dir / "point_cloud" / "iteration_0"
    init_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean PLY 복사
    shutil.copy(clean_ply, init_dir / "point_cloud.ply")
    
    # cfg_args 복사 (원본 모델에서)
    source_cfg = Path(source_model) / "cfg_args"
    if source_cfg.exists():
        shutil.copy(source_cfg, output_dir / "cfg_args")
    
    # cameras.json 복사
    source_cameras = Path(source_model) / "cameras.json"
    if source_cameras.exists():
        shutil.copy(source_cameras, output_dir / "cameras.json")
    
    return output_dir

def run_training(model_path, source_path, iterations, start_checkpoint=None):
    """
    3DGS 학습 실행
    """
    cmd = [
        "python", "train.py",
        "-s", str(source_path),
        "-m", str(model_path),
        "--iterations", str(iterations),
    ]
    
    if start_checkpoint:
        cmd.extend(["--start_checkpoint", str(start_checkpoint)])
    
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ply', required=True, help='Path to clean_gaussians.ply')
    parser.add_argument('--source_path', required=True, help='Path to source data (360_v2/stump)')
    parser.add_argument('--source_model', default='output/stump', help='Original trained model')
    parser.add_argument('--output_path', required=True, help='Output path for retrained model')
    parser.add_argument('--iterations', type=int, default=15000, help='Training iterations')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 8: Clean Initialization Retraining")
    print("=" * 60)
    print(f"Clean PLY: {args.clean_ply}")
    print(f"Source data: {args.source_path}")
    print(f"Output: {args.output_path}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60)
    
    # 1. 초기 checkpoint 준비
    print("\n[1/2] Preparing initial checkpoint from clean PLY...")
    init_checkpoint = prepare_init_checkpoint(
        args.clean_ply, 
        args.source_model,
        args.output_path
    )
    
    # 2. 재학습 실행
    print(f"\n[2/2] Starting retraining for {args.iterations} iterations...")
    run_training(
        args.output_path,
        args.source_path,
        args.iterations,
        start_checkpoint=init_checkpoint / "point_cloud" / "iteration_0" / "point_cloud.ply"
    )
    
    print("\n" + "=" * 60)
    print("Retraining completed!")
    print(f"Output saved to: {args.output_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

### 8.3 평가 스크립트

**파일**: `src/step8_evaluate.py`

```python
"""
Step 8 평가: Baseline vs Retrained 모델 비교

Metrics:
- PSNR, SSIM, LPIPS on test views
- Novel view quality comparison
- Noise detection re-run (noise reduction verification)
"""

import torch
from pathlib import Path
import json
from lpips import LPIPS
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_model(model_path, test_views, metrics=['psnr', 'ssim', 'lpips']):
    """
    모델 품질 평가
    """
    results = {m: [] for m in metrics}
    
    for view in test_views:
        rendered = render_view(model_path, view)
        gt = load_gt(view)
        
        if 'psnr' in metrics:
            results['psnr'].append(psnr(gt, rendered))
        if 'ssim' in metrics:
            results['ssim'].append(ssim(gt, rendered, multichannel=True))
        if 'lpips' in metrics:
            results['lpips'].append(lpips_fn(gt, rendered))
    
    return {m: np.mean(v) for m, v in results.items()}

def compare_models(baseline_path, retrained_path, test_views):
    """
    Baseline vs Retrained 비교
    """
    baseline_metrics = evaluate_model(baseline_path, test_views)
    retrained_metrics = evaluate_model(retrained_path, test_views)
    
    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    print(f"{'Metric':<10} {'Baseline':<15} {'Retrained':<15} {'Delta':<10}")
    print("-" * 60)
    
    for metric in baseline_metrics:
        baseline_val = baseline_metrics[metric]
        retrained_val = retrained_metrics[metric]
        delta = retrained_val - baseline_val
        
        # LPIPS는 낮을수록 좋음
        if metric == 'lpips':
            better = "↓" if delta < 0 else "↑"
        else:
            better = "↑" if delta > 0 else "↓"
        
        print(f"{metric:<10} {baseline_val:<15.4f} {retrained_val:<15.4f} {delta:+.4f} {better}")
    
    print("=" * 60)
```

---

## 실행 순서

### Phase 1: 환경 확인

```bash
# 1. Clean PLY 존재 확인
ls -lh data/processed/stump/noise_gaussians/clean_gaussians.ply

# 2. Gaussian 수 확인
head -5 data/processed/stump/noise_gaussians/clean_gaussians.ply
# element vertex 4196740 확인

# 3. 원본 모델 확인
ls -lh output/stump/point_cloud/iteration_30000/point_cloud.ply
```

### Phase 2: 재학습 실행

```bash
# Experiment A: 10k iterations
python src/step8_clean_retraining.py \
    --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \
    --source_path data/360_v2/stump \
    --output_path output/stump_retrained_10k \
    --iterations 10000

# Experiment B: 15k iterations
python src/step8_clean_retraining.py \
    --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \
    --source_path data/360_v2/stump \
    --output_path output/stump_retrained_15k \
    --iterations 15000

# Experiment C: 30k iterations (동등 조건)
python src/step8_clean_retraining.py \
    --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \
    --source_path data/360_v2/stump \
    --output_path output/stump_retrained_30k \
    --iterations 30000
```

### Phase 3: 평가

```bash
# 각 모델 평가
python src/step8_evaluate.py \
    --baseline output/stump \
    --retrained output/stump_retrained_15k \
    --test_views data/processed/stump/splits/test_indices.json

# Noise detection 재실행 (noise 감소 확인)
python src/step7_identify_noise_gaussians.py \
    --scene stump \
    --model_path output/stump_retrained_15k \
    --output_dir data/processed/stump_retrained/noise_gaussians
```

---

## 예상 결과

### 성공 시나리오

```
┌─────────────────────────────────────────────────────────────┐
│ Model               │ PSNR ↑  │ SSIM ↑  │ LPIPS ↓ │ Noise │
├─────────────────────┼─────────┼─────────┼─────────┼───────┤
│ Baseline (30k)      │ 26.50   │ 0.850   │ 0.120   │ 2783  │
│ Clean + 10k         │ 26.55   │ 0.852   │ 0.118   │ 2100  │
│ Clean + 15k         │ 26.70   │ 0.858   │ 0.115   │ 1500  │
│ Clean + 30k         │ 26.85   │ 0.862   │ 0.110   │ 1000  │
└─────────────────────┴─────────┴─────────┴─────────┴───────┘
```

**해석**: Clean 초기화로 noise가 감소하고, 더 적은 iteration으로도 품질 유지/개선

### 실패 시나리오

```
┌─────────────────────────────────────────────────────────────┐
│ Model               │ PSNR    │ SSIM    │ LPIPS   │ Noise │
├─────────────────────┼─────────┼─────────┼─────────┼───────┤
│ Baseline (30k)      │ 26.50   │ 0.850   │ 0.120   │ 2783  │
│ Clean + 15k         │ 26.30   │ 0.845   │ 0.125   │ 2500  │
└─────────────────────┴─────────┴─────────┴─────────┴───────┘
```

**해석**: Noise 제거로 인한 hole이 제대로 복구되지 않음 → 방향 수정 필요

---

## 잠재적 문제 및 대응

### 문제 1: 3DGS가 PLY 초기화를 지원하지 않음

**대응**: 
- train.py 수정하여 `--init_ply` 옵션 추가
- 또는 checkpoint 구조 직접 생성

### 문제 2: Hole이 제대로 채워지지 않음

**대응**:
- Noise 제거 전 depth prior 기반 hole filling 적용
- 또는 Noise Gaussian 제거 대신 opacity 감소로 변경

### 문제 3: 재학습 후 새로운 Noise 발생

**대응**:
- Iterative refinement: 재학습 → Noise 제거 → 재학습 반복
- 또는 Regularization 추가 (opacity, scale 제약)

---

## 다음 단계 (Step 8 완료 후)

### 성공 시
1. **논문 작성**: "Clean Initialization improves 3DGS quality"
2. **다른 scene 검증**: bicycle, garden 등에서 일반화 확인
3. **Iterative refinement**: 여러 번 반복하여 추가 개선

### 실패 시
1. **원인 분석**: Hole filling 문제인지, 다른 문제인지
2. **방향 수정**: 
   - 제거 대신 Regularization 방향
   - 또는 2D Denoising network 방향

---

## 파일 구조 (Step 8 완료 후)

```
output/
├── stump/                          # Baseline (기존)
├── stump_clean_v2/                 # Noise 제거만 (기존)
├── stump_retrained_10k/            # Clean + 10k (NEW)
├── stump_retrained_15k/            # Clean + 15k (NEW)
└── stump_retrained_30k/            # Clean + 30k (NEW)

data/processed/stump/
├── noise_gaussians/                # Step 7 결과 (기존)
└── evaluation/                     # Step 8 평가 결과 (NEW)
    ├── baseline_metrics.json
    ├── retrained_10k_metrics.json
    ├── retrained_15k_metrics.json
    ├── retrained_30k_metrics.json
    └── comparison_report.md
```

---

## 체크리스트

### 구현
- [ ] `src/step8_clean_retraining.py` 작성
- [ ] `src/step8_evaluate.py` 작성
- [ ] train.py에 `--init_ply` 옵션 추가 (필요시)

### 실험
- [ ] Experiment A: Clean + 10k 실행
- [ ] Experiment B: Clean + 15k 실행
- [ ] Experiment C: Clean + 30k 실행
- [ ] 각 모델 평가 (PSNR, SSIM, LPIPS)

### 검증
- [ ] Noise detection 재실행하여 noise 감소 확인
- [ ] Visual comparison (렌더링 이미지 비교)
- [ ] 다른 scene에서 일반화 테스트

---

Last Updated: 2025-02-02
Status: 계획 수립 완료, 구현 대기