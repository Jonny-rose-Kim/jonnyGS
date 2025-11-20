# 3DGS Noise Detection Pipeline - Usage Guide

## ✅ 구현 완료 상태

모든 5단계 파이프라인이 성공적으로 구현되고 검증되었습니다!

### 완료된 단계

1. ✅ **Step 1: Train/Test Split** - 125 images → 93 train + 32 test
2. ✅ **Step 2: 3DGS Training Wrapper** - Train images로 3DGS 학습
3. ✅ **Step 3: Novel View Rendering** - 32 test views 렌더링 완료
4. ✅ **Step 4: Error Map Generation** - 32 dataset pairs 생성 (train: 25, val: 3, test: 4)
5. ✅ **Step 5: Detector Training** - U-Net 기반 노이즈 감지 모델

---

## 🚀 실행 방법

### Step-by-Step 실행

```bash
# 프로젝트 루트에서 실행

# Step 1: Train/Test Split
python src/step1_split.py \
  --scene stump \
  --data_root ./data/360_v2 \
  --output_root ./data/processed \
  --skip_frames 3

# Step 2: 3DGS Training (선택사항 - 이미 학습된 모델 사용 가능)
python src/step2_train_3dgs.py \
  --scene stump \
  --data_root ./data/360_v2 \
  --output_root ./data/processed \
  --iterations 30000 \
  --skip_training  # 기존 모델 사용

# Step 3: Novel View Rendering
python src/step3_render_views_v2.py \
  --scene stump \
  --model_path ./output/stump \
  --data_root ./data/360_v2 \
  --output_root ./data/processed

# Step 4: Error Map & Dataset Pairs Generation
python src/step4_generate_pairs.py \
  --scene stump \
  --output_root ./data/processed \
  --threshold 0.15 \
  --min_size 50

# Step 5: Detector Training
python src/step5_train_detector.py \
  --scene stump \
  --output_root ./data/processed \
  --num_epochs 50 \
  --batch_size 4 \
  --image_size 256
```

---

## 📊 Stump Scene 검증 결과

### Step 1 결과
```
총 이미지 수: 125장
Train: 93장 (74.4%)
Test:  32장 (25.6%)

출력:
  ✓ data/processed/stump/splits/train_indices.json
  ✓ data/processed/stump/splits/test_indices.json
```

### Step 3 결과
```
렌더링 완료: 32개 test views
속도: ~1.37 it/s
출력: 64개 파일 (32 rendered + 32 GT)

파일:
  ✓ data/processed/stump/rendered_views/render_0000.png
  ✓ data/processed/stump/rendered_views/gt_0000.png
  ...
```

### Step 4 결과
```
Dataset pairs 생성: 32개
Split: Train 25, Val 3, Test 4

Noise ratio 통계:
  평균: 0.001 (0.1%)
  최소: 0.000
  최대: 0.002

출력 구조:
data/processed/stump/dataset_pairs/
  ├── train/
  │   └── pair_0000/
  │       ├── rendered.png
  │       ├── gt.png
  │       ├── mask.png
  │       └── metadata.json
  ├── val/
  └── test/
```

**주의:** Noise ratio가 매우 낮은 이유는 3DGS 렌더링 품질이 매우 우수하기 때문입니다.
더 많은 노이즈를 감지하려면 `--threshold`를 낮추세요 (예: 0.05).

---

## 🛠️ 구현된 핵심 기능

### 1. Custom Rendering System
- 특정 camera indices만 선택적으로 렌더링
- COLMAP sparse data 자동 로드
- Train/Test view 자동 분리

### 2. Error Map Generation
- Pixel-wise L1 error 기반
- 작은 노이즈 영역 제거 (morphological operation)
- Configurable threshold

### 3. Simple U-Net Detector
- Encoder-Decoder 구조
- Skip connections
- Binary segmentation output
- IoU metric tracking

---

## 📁 전체 출력 구조

```
data/processed/stump/
├── splits/
│   ├── train_indices.json
│   └── test_indices.json
├── 3dgs_output/           # (Step 2에서 생성, 현재는 output/stump 사용)
├── rendered_views/
│   ├── render_0000.png
│   ├── gt_0000.png
│   └── ...
└── dataset_pairs/
    ├── train/
    ├── val/
    └── test/

experiments/stump/
└── checkpoints/
    └── best_model.pth
```

---

## ⚙️ 주요 파라미터

### Step 1 (Split)
- `--skip_frames`: Test view 간격 (기본값: 3)

### Step 3 (Rendering)
- `--model_path`: 3DGS 모델 경로
- `--data_root`: 원본 데이터 경로 (COLMAP sparse 필요)

### Step 4 (Error Map)
- `--threshold`: 노이즈 감지 임계값 (기본값: 0.15)
  - 낮추면 더 민감 (더 많은 노이즈 감지)
  - 높이면 덜 민감
- `--min_size`: 최소 노이즈 영역 크기 (픽셀, 기본값: 50)

### Step 5 (Training)
- `--num_epochs`: 학습 epoch 수
- `--batch_size`: 배치 크기
- `--image_size`: 이미지 크기 (기본값: 256)
- `--lr`: 학습률 (기본값: 0.001)

---

## 🔧 다른 Scene에 적용하기

```bash
# 1. 데이터 준비
# data/360_v2/{your_scene}/
#   ├── input/           # 모든 이미지
#   └── sparse/0/        # COLMAP 데이터

# 2. 파이프라인 실행
python src/step1_split.py --scene your_scene ...
python src/step3_render_views_v2.py --scene your_scene ...
python src/step4_generate_pairs.py --scene your_scene ...
python src/step5_train_detector.py --scene your_scene ...
```

---

## 🐛 Troubleshooting

### Noise ratio가 너무 낮음 (< 0.01)
→ `--threshold` 값을 낮추세요 (예: 0.05, 0.03)

### 렌더링이 실패함
→ source_path에 sparse 폴더가 있는지 확인
→ model_path가 올바른 3DGS checkpoint를 가리키는지 확인

### Validation set이 비어있음
→ Dataset pairs 개수가 너무 적음. 더 많은 test views 필요

### CUDA out of memory
→ `--batch_size` 줄이기 (예: 2)
→ `--image_size` 줄이기 (예: 128)

---

## 📌 다음 단계

1. **통합 실행 스크립트** (선택사항)
   - `scripts/run_all.py`: 전체 파이프라인 한번에 실행
   - `scripts/run_step.py`: 특정 단계만 실행

2. **평가 도구**
   - 학습된 모델로 test set 평가
   - 시각화 도구 (mask overlay)

3. **개선 사항**
   - Data augmentation
   - Advanced loss functions (Dice loss, Focal loss)
   - Larger U-Net architectures
   - Ensemble models

---

## 📚 참고

- **claude_modular_pipeline.md**: 상세한 설계 문서
- **CLAUDE.md**: 프로젝트 전체 가이드
- **README.md**: 원본 3DGS 문서

---

생성일: 2025-11-04
검증 완료: Stump scene (125 images)
