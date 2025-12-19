# CLAUDE.md - 3D Gaussian Splatting Noise Detection & Identification

## Project Overview

**3D Gaussian Splatting Noise Detection & Identification** - 3DGS 렌더링에서 노이즈를 감지하고, 이를 3D 공간의 가우시안으로 역추적하는 파이프라인

### Current Status (2024-11-28)

| Stage | Status | Description |
|-------|--------|-------------|
| Step 1-6 | ✅ 완료 | 2D 노이즈 마스크 감지 파이프라인 |
| Step 7 | 🔄 설계 완료 | 3D 노이즈 가우시안 식별 (구현 예정) |

---

## 완료된 파이프라인 (Step 1-6) 요약

### 핵심 성과
- 6단계 모듈러 파이프라인 구현 완료
- U-Net 기반 노이즈 감지 모델 학습 성공
- 최적 파라미터: `threshold=0.15, min_size=30` (L1+LPIPS mode)
- 검증 완료 씬: stump, bicycle

### 파이프라인 흐름

```
Step 1: Train/Test Split (125 → 93 train + 32 test)
    ↓
Step 2: 3DGS Training (93 images)
    ↓
Step 3: Novel View Rendering (test views)
    ↓
Step 4: Error Map Generation (L1 + LPIPS)
    ↓
Step 5: Detector Training (U-Net)
    ↓
Step 6: Interpolated View Detection (Validation)
```

### 빠른 실행
```bash
# 전체 파이프라인 (L1+LPIPS 모드)
python src/step1_split.py --scene stump --skip_frames 3
python src/step2_train_3dgs.py --scene stump --iterations 30000
python src/step3_render_views_v2.py --scene stump
python src/step4_generate_pairs.py --scene stump --threshold 0.15 --min_size 30
python src/step5_train_detector.py --scene stump --num_epochs 50
python src/step6_interpolate_detect.py --scene stump
```

### 주요 파일
- `src/step1_split.py` ~ `src/step6_interpolate_detect.py`: 각 단계 스크립트
- `src/models/simple_unet.py`: U-Net 아키텍처
- `src/utils/camera_interpolation.py`: 카메라 보간 유틸리티

---

## 새로운 목표: Step 7 - 3D Noise Gaussian Identification

### 핵심 아이디어

**2D 노이즈 마스크 → 3D 노이즈 가우시안 역투영**

```
[Parent 3DGS Model] → [Novel Views 렌더링] → [2D 노이즈 마스크]
                                                    ↓
                                          [Ray-Gaussian Intersection]
                                                    ↓
                                          [각 뷰별 노이즈 후보 가우시안]
                                                    ↓
                                          [교집합 (50% 투표)]
                                                    ↓
                                          [최종 노이즈 가우시안]
```

### 구현 전략

#### 방법 A: 타일 기반 근사 (코드 수정 없음)
- 장점: 기존 코드 활용
- 단점: 정확도 낮음

#### 방법 B: Ray-Gaussian Intersection (Python) ⬅️ **선택**
- 장점: 정확, 디버깅 용이, 유연함
- 단점: 계산량 (최적화 가능)

#### 방법 C: CUDA Rasterizer 수정
- 장점: 가장 효율적
- 단점: CUDA 코드 수정 필요, 메모리 관리 복잡

### Step 7 세부 단계

#### Phase 1: 카메라 클러스터링
- 카메라가 밀집된 영역에서 novel view 생성
- Hallucination 영역 방지
- K-means 또는 DBSCAN 클러스터링

#### Phase 2: Ray-Gaussian Intersection
- 2D 마스크 픽셀 → 카메라 광선 생성
- 광선과 가우시안(타원체) 교차 판정
- 마할라노비스 거리 기반 필터링

#### Phase 3: 교집합 계산
- 여러 뷰에서 투표 (50% 이상 = 노이즈)
- 최종 노이즈 가우시안 인덱스 집합 추출

### 예상 출력
```json
{
  "total_gaussians": 245892,
  "noise_gaussians": 3421,
  "noise_ratio": 0.0139,
  "noise_gaussian_indices": [123, 456, 789, ...]
}
```

### 활용 방안
1. **노이즈 가우시안 제거** → 품질 개선된 모델
2. **노이즈 패턴 분석** → floater, blur 등 분류
3. **재학습 regularization** → 노이즈 가우시안에 페널티

---

## 프로젝트 구조

```
GS_noise_masking/
├── MD_list/                              # 문서 디렉토리
│   ├── CLAUDE.md                         # 이 파일
│   ├── PIPELINE_USAGE.md                 # Step 1-6 사용 가이드
│   ├── STEP6_USAGE.md                    # Step 6 상세 문서
│   └── STEP7_NOISE_GAUSSIAN_IDENTIFICATION.md  # Step 7 설계 문서 ⬅️ 신규
│
├── src/
│   ├── step1_split.py ~ step6_*.py       # 기존 파이프라인
│   ├── step7_identify_noise_gaussians.py # Step 7 (구현 예정)
│   │
│   ├── utils/
│   │   ├── camera_interpolation.py       # 카메라 보간
│   │   ├── camera_clustering.py          # 카메라 클러스터링 (구현 예정)
│   │   └── ray_gaussian_intersection.py  # Ray-Gaussian 교차 (구현 예정)
│   │
│   └── models/
│       └── simple_unet.py                # 노이즈 감지 모델
│
├── scene/
│   └── gaussian_model.py                 # 가우시안 모델 (핵심)
│       # - _xyz: 가우시안 위치 [N, 3]
│       # - get_scaling: 스케일 [N, 3]
│       # - get_rotation: 회전 [N, 4]
│       # - get_covariance(): 공분산 [N, 6]
│
├── submodules/
│   └── diff-gaussian-rasterization/
│       └── cuda_rasterizer/
│           ├── forward.cu                # 렌더링 (인덱스 추적 가능)
│           └── rasterizer_impl.cu        # 타일/빈 관리
│
└── data/
    └── processed/{scene}/
        ├── interpolated_results/         # Step 6 출력
        └── noise_gaussians/              # Step 7 출력 (예정)
```

---

## 핵심 코드 참조

### 가우시안 모델 (scene/gaussian_model.py)
```python
class GaussianModel:
    def get_xyz(self):          # 위치 [N, 3]
    def get_scaling(self):      # 스케일 [N, 3]
    def get_rotation(self):     # 회전 쿼터니언 [N, 4]
    def get_covariance(self):   # 공분산 행렬 [N, 6] (상삼각)
    def prune_points(self, mask):  # 가우시안 제거
```

### CUDA Rasterizer (forward.cu:334)
```cuda
// 픽셀에 기여하는 가우시안 인덱스
int coll_id = point_list[range.x + progress];
collected_id[block.thread_rank()] = coll_id;  // 가우시안 인덱스
```

### Ray-Gaussian Intersection (구현 예정)
```python
def ray_ellipsoid_intersect(ray_origin, ray_dir, center, cov, threshold=3.0):
    """
    광선과 가우시안(타원체) 교차 판정
    마할라노비스 거리 <= threshold 이면 교차
    """
    inv_cov = torch.inverse(cov)
    # 이차 방정식 풀이
    # ...
```

---

## 개발 우선순위

### 현재 작업 (Step 7)
1. ⬜ 카메라 클러스터링 구현
2. ⬜ Ray-Gaussian Intersection 구현
3. ⬜ 교집합 계산 로직
4. ⬜ 통합 테스트

### 향후 작업
- Step 8: 노이즈 가우시안 자동 제거
- Step 9: 노이즈 패턴 분류
- Step 10: 학습 중 실시간 노이즈 감지

---

## 하드웨어 요구사항

| 작업 | GPU VRAM | 예상 시간 |
|------|----------|-----------|
| Step 1-4 | 8GB+ | ~30분 |
| Step 5 (학습) | 8GB+ | ~1시간 |
| Step 6 (검증) | 16GB+ | ~10분 |
| Step 7 (구현 예정) | 16GB+ | ~30분 |

---

## 참고 문서

- `PIPELINE_USAGE.md`: Step 1-6 상세 사용법
- `STEP6_USAGE.md`: 보간 뷰 감지 상세
- `STEP7_NOISE_GAUSSIAN_IDENTIFICATION.md`: Step 7 설계 문서

---

Last Updated: 2024-11-28
