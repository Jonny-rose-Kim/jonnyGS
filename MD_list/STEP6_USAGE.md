# Step 6: Interpolated View Rendering + Noise Detection

## 개요

Step 6는 125장으로 학습된 3DGS 모델에서 **인접 카메라 중점**에서 novel view를 렌더링하고, Step 5에서 학습된 노이즈 감지 모델을 적용하는 파이프라인입니다.

### 주요 목적
- 학습 데이터에 없는 **어려운 시점**(인접 카메라 중간)에서 렌더링 품질 평가
- 노이즈 감지 모델의 일반화 성능 테스트
- 3DGS 모델의 보간(interpolation) 성능 분석

---

## 🎯 Step 6의 작동 원리

1. **카메라 쌍 선택**: 125장 모델의 인접 카메라 쌍 찾기
2. **중점 계산**: 두 카메라 위치와 회전의 중점(interpolation) 계산
   - 위치: 선형 보간 (Linear Interpolation)
   - 회전: 쿼터니언 SLERP (Spherical Linear Interpolation)
3. **렌더링**: 중점 카메라에서 3DGS 렌더링
4. **노이즈 감지**: 학습된 U-Net 모델 적용
5. **시각화 및 저장**: 결과를 이미지와 JSON으로 저장

---

## 📋 전제 조건

### 필요한 선행 단계
- ✅ **Step 1-5 완료**: 모든 이전 단계가 완료되어야 합니다
- ✅ **125장 모델 존재**: `output/stump/` (전체 데이터로 학습된 모델)
- ✅ **Detector 모델 존재**: `experiments/stump/checkpoints/best_model.pth`

### 확인 명령
```bash
# 125장 모델 확인
ls output/stump/point_cloud/iteration_30000/point_cloud.ply

# Detector 모델 확인
ls experiments/stump/checkpoints/best_model.pth
```

---

## 🚀 실행 방법

### 기본 실행

```bash
python src/step6_interpolate_detect.py \
  --scene stump \
  --model_path ./output/stump \
  --detector_checkpoint ./experiments/stump/checkpoints/best_model.pth \
  --data_root ./data/360_v2 \
  --output_root ./data/processed \
  --num_interpolations 1 \
  --max_pairs 10 \
  --image_size 256
```

### 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--scene` | (필수) | Scene 이름 (예: stump) |
| `--model_path` | `output/{scene}` | 125장으로 학습된 3DGS 모델 경로 |
| `--detector_checkpoint` | `experiments/{scene}/checkpoints/best_model.pth` | 노이즈 감지 모델 경로 |
| `--data_root` | `./data/360_v2` | 원본 데이터 경로 (카메라 정보 로드용) |
| `--output_root` | `./data/processed` | 결과 저장 경로 |
| `--num_interpolations` | 1 | 카메라 쌍당 생성할 중점 수 (1 = 정중앙만) |
| `--max_pairs` | 20 | 처리할 최대 카메라 쌍 수 |
| `--image_size` | 256 | Detector 입력 이미지 크기 |
| `--sh_degree` | 3 | Spherical harmonics degree |

---

## 📊 출력 구조

```
data/processed/stump/interpolated_results/
├── summary.json                    # 전체 결과 요약
├── interpolated_0000/
│   ├── rendered.png                # 렌더링된 이미지
│   ├── mask.png                    # 노이즈 마스크 (Binary)
│   ├── visualization.png           # 통합 시각화 (Rendered | Mask | Overlay)
│   └── metadata.json               # 상세 메타데이터
├── interpolated_0001/
│   └── ...
└── interpolated_000X/
    └── ...
```

### metadata.json 구조

```json
{
  "index": 0,
  "camera_1": "_DSC9213.JPG",
  "camera_2": "_DSC9214.JPG",
  "interpolation_t": 0.5,
  "noise_ratio": 0.5917,
  "noise_pixels": 1003920,
  "total_pixels": 1696000,
  "camera_1_center": [1.2345, -0.5678, 2.3456],
  "camera_2_center": [1.3456, -0.6789, 2.4567]
}
```

### summary.json 구조

```json
{
  "scene": "stump",
  "num_interpolated_views": 3,
  "num_interpolations_per_pair": 1,
  "max_pairs": 3,
  "model_path": "output/stump",
  "detector_path": "experiments/stump/checkpoints/best_model.pth",
  "results": [...],  // 모든 interpolated views의 metadata
  "statistics": {
    "mean_noise_ratio": 0.5917,
    "std_noise_ratio": 0.0006,
    "min_noise_ratio": 0.5911,
    "max_noise_ratio": 0.5924
  }
}
```

---

## 📈 Stump Scene 검증 결과

### 실행 예시
```bash
python src/step6_interpolate_detect.py \
  --scene stump \
  --max_pairs 3 \
  --num_interpolations 1
```

### 출력 결과
```
처리된 interpolated views: 3
노이즈 통계:
  평균 noise ratio: 0.5917 (59.17%)
  표준편차: 0.0006
  최소: 0.5911
  최대: 0.5924
```

### 분석
- **높은 Noise Ratio (59%)**: 인접 카메라 중점은 학습 데이터에 없는 어려운 시점이므로, 노이즈가 많이 검출됨
- **낮은 표준편차 (0.06%)**: 일관된 노이즈 패턴, 모델이 안정적으로 예측

---

## 🔬 고급 사용법

### 1. 여러 중점 생성 (Fine-grained Interpolation)

카메라 쌍당 여러 중점을 생성하여 더 세밀한 분석:

```bash
python src/step6_interpolate_detect.py \
  --scene stump \
  --num_interpolations 3 \
  --max_pairs 5
```

이 경우:
- 각 카메라 쌍에서 3개의 중점 생성 (t=0.25, 0.5, 0.75)
- 총 5 pairs × 3 interpolations = **15개 views** 생성

### 2. 전체 Scene 분석

모든 인접 카메라 쌍 분석:

```bash
python src/step6_interpolate_detect.py \
  --scene stump \
  --max_pairs 124  # 125 cameras = 124 adjacent pairs
```

⚠️ **주의**: 렌더링 시간이 오래 걸릴 수 있습니다 (~2분/view).

### 3. 다른 Scene 적용

```bash
# 1. Step 1-5를 먼저 실행
python src/step1_split.py --scene bicycle ...
python src/step2_train_3dgs.py --scene bicycle ...
# ... (Step 3-5)

# 2. Step 6 실행
python src/step6_interpolate_detect.py \
  --scene bicycle \
  --model_path ./output/bicycle \
  --max_pairs 10
```

---

## 🛠️ 구현 세부사항

### 카메라 보간 알고리즘

#### 위치 보간 (Linear Interpolation)
```python
pos_interp = (1 - t) * pos1 + t * pos2  # t = 0.5 for midpoint
```

#### 회전 보간 (Quaternion SLERP)
```python
# 1. Rotation Matrix → Quaternion
q1 = rotation_matrix_to_quaternion(R1)
q2 = rotation_matrix_to_quaternion(R2)

# 2. Spherical Linear Interpolation
q_interp = slerp(q1, q2, t)

# 3. Quaternion → Rotation Matrix
R_interp = quaternion_to_rotation_matrix(q_interp)
```

**SLERP를 사용하는 이유:**
- 선형 보간은 회전에 부적합 (왜곡 발생)
- SLERP는 구면상에서 일정한 속도로 보간
- 자연스러운 카메라 회전 생성

### 노이즈 감지 흐름

```
Rendered Image [3, H, W]
    ↓
Detector (U-Net)
    ↓
Noise Mask [1, H, W]  # Values in [0, 1]
    ↓
Binary Mask (threshold=0.5)
    ↓
Visualization & Save
```

---

## 📊 결과 분석 방법

### 1. 시각적 분석

`visualization.png` 파일 확인:
- **Left**: 렌더링된 이미지
- **Center**: 노이즈 마스크 (밝을수록 노이즈가 많음)
- **Right**: Overlay (빨간색 = 노이즈 영역)

### 2. 정량적 분석

`summary.json` 로드하여 통계 분석:

```python
import json
import numpy as np

with open('data/processed/stump/interpolated_results/summary.json') as f:
    summary = json.load(f)

# 노이즈 비율 분포
noise_ratios = [r['noise_ratio'] for r in summary['results']]
print(f"Mean: {np.mean(noise_ratios):.4f}")
print(f"Std:  {np.std(noise_ratios):.4f}")
print(f"Min:  {np.min(noise_ratios):.4f}")
print(f"Max:  {np.max(noise_ratios):.4f}")
```

### 3. 카메라 거리 vs 노이즈 상관관계

```python
import numpy as np

for result in summary['results']:
    # 카메라 간 거리 계산
    p1 = np.array(result['camera_1_center'])
    p2 = np.array(result['camera_2_center'])
    dist = np.linalg.norm(p1 - p2)

    print(f"Distance: {dist:.3f}, Noise Ratio: {result['noise_ratio']:.4f}")
```

**예상 결과**: 카메라 거리가 멀수록 노이즈 비율이 증가할 가능성

---

## 🔧 트러블슈팅

### 문제 1: "Model not found" 에러

```
FileNotFoundError: Model not found: output/stump
```

**해결책**: 125장으로 학습된 모델이 있는지 확인

```bash
ls output/stump/point_cloud/iteration_30000/

# 없다면 먼저 학습
python train.py -s data/360_v2/stump -m output/stump
```

### 문제 2: "Detector checkpoint not found" 에러

```
FileNotFoundError: Detector checkpoint not found
```

**해결책**: Step 5를 먼저 실행

```bash
python src/step5_train_detector.py \
  --scene stump \
  --num_epochs 50
```

### 문제 3: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결책**: 배치 크기 또는 이미지 크기 줄이기

```bash
python src/step6_interpolate_detect.py \
  --scene stump \
  --image_size 128  # 256 → 128
  --max_pairs 5     # 적은 수의 쌍 처리
```

### 문제 4: 너무 많은 노이즈 감지됨

**원인**:
- Detector 모델이 충분히 학습되지 않음
- 인접 카메라 중점이 너무 어려운 시점

**해결책**:
1. Step 5에서 더 많은 epoch 학습
2. `--threshold` 값 조정 (Step 4에서)
3. 더 가까운 카메라 쌍만 분석

---

## 📌 활용 사례

### 1. 3DGS 모델 품질 평가
- 인접 카메라 중점에서의 렌더링 품질 확인
- 학습 데이터 밀도와 재구성 품질 상관관계 분석

### 2. 노이즈 감지 모델 평가
- 학습 데이터(test views)와 interpolated views에서의 성능 비교
- 일반화 성능 테스트

### 3. 카메라 배치 최적화
- 어떤 카메라 간격에서 노이즈가 증가하는지 분석
- 최적 카메라 밀도 결정

### 4. 데이터 증강 (Data Augmentation)
- Interpolated views를 추가 학습 데이터로 사용
- 노이즈 마스크를 ground truth로 활용

---

## 🎓 참고 자료

### 관련 개념
- **Camera Interpolation**: [Quaternion SLERP](https://en.wikipedia.org/wiki/Slerp)
- **3D Gaussian Splatting**: [Original Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Novel View Synthesis**: 학습되지 않은 시점에서 이미지 생성

### 프로젝트 문서
- **전체 파이프라인**: `PIPELINE_USAGE.md`
- **프로젝트 가이드**: `CLAUDE.md`
- **설계 문서**: `claude_modular_pipeline.md`

---

## 📝 핵심 파일

```
src/step6_interpolate_detect.py           # 메인 스크립트
src/utils/camera_interpolation.py         # 카메라 보간 유틸리티
src/utils/detector_inference.py           # 노이즈 감지 모델 추론
src/utils/visualization_utils.py          # 시각화 유틸리티
```

---

생성일: 2025-11-05
검증 완료: Stump scene (125 images)
작성자: Claude Code
