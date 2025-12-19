# Step 7: 3D Noise Gaussian Identification

## 개요

Step 7은 2D 노이즈 마스크를 3D 공간의 **노이즈 가우시안**으로 역투영(back-projection)하는 단계입니다.

### 핵심 아이디어

```
[Parent 3DGS Model] → [Novel Views 렌더링] → [2D 노이즈 마스크 (Step 6)]
                                                      ↓
                                            [2D → 3D 역투영]
                                                      ↓
                                            [각 뷰별 노이즈 후보 가우시안]
                                                      ↓
                                            [교집합 = 진짜 노이즈 가우시안]
```

### 목표

- 2D 마스크의 흰색 영역 뒤에 있는 3D 가우시안 식별
- 여러 뷰에서 공통으로 노이즈로 판별된 가우시안만 추출
- **최종 결과**: Parent 3DGS 모델에서 "노이즈 가우시안"으로 라벨링된 가우시안 인덱스 집합

---

## 알고리즘 설계

### Phase 1: 카메라 클러스터링 (Camera Clustering)

**문제**: Novel view들이 서로 너무 다른 곳을 바라보면 교집합이 의미 없음

**해결책**: 카메라가 밀집된 영역에서 novel view 생성

```python
def cluster_cameras(cameras, n_clusters=5):
    """
    카메라를 위치 기반으로 클러스터링

    Args:
        cameras: 모든 카메라 리스트
        n_clusters: 클러스터 수

    Returns:
        clusters: List[List[Camera]] - 클러스터별 카메라 그룹
    """
    # 1. 카메라 위치 추출
    positions = [cam.camera_center.cpu().numpy() for cam in cameras]

    # 2. K-means 또는 DBSCAN 클러스터링
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(positions)

    # 3. 클러스터별 그룹화
    clusters = [[] for _ in range(n_clusters)]
    for cam, label in zip(cameras, labels):
        clusters[label].append(cam)

    return clusters
```

**클러스터 선택 기준**:
- 카메라 수가 많은 클러스터 우선
- 카메라 간 거리가 가까운 클러스터 우선
- 이 클러스터 내에서 novel view (중점) 생성

### Phase 2: Ray-Gaussian Intersection

**핵심**: 2D 마스크 픽셀 → 카메라 광선 → 해당 광선과 교차하는 가우시안 찾기

```python
def ray_gaussian_intersection(ray_origin, ray_direction, gaussian_center, gaussian_cov):
    """
    광선과 3D 가우시안의 교차 여부 판정

    가우시안은 타원체(ellipsoid)로 근사:
    (x - μ)^T Σ^(-1) (x - μ) <= threshold

    Args:
        ray_origin: 카메라 위치 [3]
        ray_direction: 광선 방향 (정규화) [3]
        gaussian_center: 가우시안 중심 μ [3]
        gaussian_cov: 가우시안 공분산 Σ [3, 3]

    Returns:
        bool: 교차 여부
        float: 교차점까지 거리 (t 값)
    """
    # 1. 광선-타원체 교차 방정식
    # ray: p(t) = o + t*d
    # ellipsoid: (p - μ)^T Σ^(-1) (p - μ) = 1 (마할라노비스 거리 = 1)

    # 2. 이차 방정식으로 변환하여 해 구하기
    # At^2 + Bt + C = 0

    inv_cov = torch.inverse(gaussian_cov)
    diff = ray_origin - gaussian_center

    A = ray_direction @ inv_cov @ ray_direction
    B = 2 * diff @ inv_cov @ ray_direction
    C = diff @ inv_cov @ diff - MAHALANOBIS_THRESHOLD**2

    discriminant = B**2 - 4*A*C

    if discriminant < 0:
        return False, float('inf')

    t = (-B - torch.sqrt(discriminant)) / (2*A)
    return t > 0, t.item()
```

**최적화 (Vectorized)**:
```python
def find_intersecting_gaussians_batch(camera, mask, gaussians, threshold=3.0):
    """
    마스크 영역의 모든 픽셀에 대해 교차 가우시안 찾기 (배치 처리)

    Args:
        camera: 카메라 객체
        mask: 2D 노이즈 마스크 [H, W]
        gaussians: GaussianModel
        threshold: 마할라노비스 거리 임계값 (3σ = 99.7%)

    Returns:
        noise_candidates: Set[int] - 노이즈 후보 가우시안 인덱스
    """
    # 1. 마스크에서 흰색 픽셀 좌표 추출
    white_pixels = torch.where(mask > 0.5)
    u, v = white_pixels[1], white_pixels[0]  # (x, y)

    # 2. 픽셀 → 광선 변환 (배치)
    rays = camera.pixels_to_rays(u, v)  # [N_pixels, 3]
    ray_origin = camera.camera_center   # [3]

    # 3. 모든 가우시안에 대해 거리 계산
    gaussian_centers = gaussians.get_xyz           # [N_gaussians, 3]
    gaussian_covs = gaussians.get_covariance()     # [N_gaussians, 3, 3]

    # 4. 마할라노비스 거리 기반 필터링
    noise_candidates = set()

    # 효율성을 위해 먼저 유클리드 거리로 필터링
    for ray_idx in range(len(rays)):
        ray_dir = rays[ray_idx]

        # 광선에 가까운 가우시안만 정밀 검사
        distances = compute_ray_point_distances(ray_origin, ray_dir, gaussian_centers)
        close_gaussians = torch.where(distances < DISTANCE_THRESHOLD)[0]

        for g_idx in close_gaussians:
            if ray_ellipsoid_intersect(ray_origin, ray_dir,
                                       gaussian_centers[g_idx],
                                       gaussian_covs[g_idx],
                                       threshold):
                noise_candidates.add(g_idx.item())

    return noise_candidates
```

### Phase 3: 교집합 계산 및 노이즈 가우시안 정의

```python
def identify_noise_gaussians(all_view_candidates, vote_threshold=0.5):
    """
    여러 뷰의 후보들에서 교집합 추출

    Args:
        all_view_candidates: List[Set[int]] - 각 뷰별 노이즈 후보 인덱스
        vote_threshold: 투표 임계값 (0.5 = 50% 이상의 뷰에서 노이즈로 판별)

    Returns:
        noise_gaussians: Set[int] - 최종 노이즈 가우시안 인덱스
    """
    # 가우시안별 투표 수 계산
    vote_count = Counter()
    view_count = Counter()  # 해당 가우시안이 보이는 뷰 수

    for view_idx, candidates in enumerate(all_view_candidates):
        for g_idx in candidates:
            vote_count[g_idx] += 1

        # 해당 뷰에서 "보이는" 가우시안들 기록
        visible_gaussians = get_visible_gaussians(view_idx)
        for g_idx in visible_gaussians:
            view_count[g_idx] += 1

    # 투표 비율 기준 필터링
    noise_gaussians = set()
    for g_idx, votes in vote_count.items():
        visible_views = view_count[g_idx]
        if visible_views > 0 and votes / visible_views >= vote_threshold:
            noise_gaussians.add(g_idx)

    return noise_gaussians
```

---

## 전체 파이프라인

```python
def step7_identify_noise_gaussians(
    model_path,           # Parent 3DGS 모델 경로
    source_path,          # 원본 데이터 경로
    detector_path,        # 노이즈 감지 모델 경로
    output_dir,           # 출력 디렉토리
    n_clusters=5,         # 카메라 클러스터 수
    views_per_cluster=5,  # 클러스터당 생성할 novel view 수
    vote_threshold=0.5,   # 투표 임계값
    mahalanobis_threshold=3.0  # 마할라노비스 거리 임계값
):
    """
    Step 7: 2D 노이즈 마스크 → 3D 노이즈 가우시안 식별
    """

    # 1. 모델 및 카메라 로드
    gaussians, cameras, scene = load_model_and_cameras(model_path, source_path)
    detector = load_detector(detector_path)

    # 2. 카메라 클러스터링
    clusters = cluster_cameras(cameras, n_clusters)
    print(f"카메라 클러스터링 완료: {[len(c) for c in clusters]} cameras per cluster")

    # 3. 각 클러스터에서 novel view 생성 및 노이즈 후보 수집
    all_view_candidates = []

    for cluster_idx, cluster_cameras in enumerate(clusters):
        # 클러스터 내 인접 카메라 쌍에서 novel view 생성
        novel_views = create_cluster_novel_views(cluster_cameras, views_per_cluster)

        for view in novel_views:
            # 렌더링
            rendered = render_view(gaussians, view)

            # 노이즈 마스크 예측
            mask = detector.predict(rendered)

            # Ray-Gaussian intersection으로 후보 추출
            candidates = find_intersecting_gaussians_batch(
                view, mask, gaussians, mahalanobis_threshold
            )
            all_view_candidates.append(candidates)

    # 4. 교집합 계산
    noise_gaussians = identify_noise_gaussians(all_view_candidates, vote_threshold)

    # 5. 결과 저장
    save_noise_gaussians(noise_gaussians, gaussians, output_dir)

    return noise_gaussians
```

---

## 출력 형식

### noise_gaussians.json
```json
{
  "scene": "stump",
  "model_path": "output/stump",
  "total_gaussians": 245892,
  "noise_gaussians": 3421,
  "noise_ratio": 0.0139,
  "parameters": {
    "n_clusters": 5,
    "views_per_cluster": 5,
    "vote_threshold": 0.5,
    "mahalanobis_threshold": 3.0
  },
  "noise_gaussian_indices": [123, 456, 789, ...],
  "per_view_statistics": [
    {"view_idx": 0, "candidates": 1245, "final_noise": 342},
    ...
  ]
}
```

### noise_gaussians.ply
노이즈 가우시안만 포함된 PLY 파일 (시각화용)

### clean_gaussians.ply
노이즈가 제거된 가우시안 PLY 파일

---

## 활용 방안

### 1. 노이즈 가우시안 제거
```python
# 노이즈 가우시안 제거 후 새 모델 저장
clean_mask = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool)
clean_mask[list(noise_gaussians)] = False
gaussians.prune_points(~clean_mask)
gaussians.save_ply("clean_model.ply")
```

### 2. 노이즈 가우시안 시각화
```python
# SIBR Viewer에서 노이즈 가우시안만 빨간색으로 표시
# 또는 별도 PLY로 저장하여 CloudCompare에서 확인
```

### 3. 재학습 파이프라인
```python
# 노이즈 가우시안에 더 낮은 학습률 적용
# 또는 노이즈 가우시안을 정규화 타겟으로 사용
```

---

## 예상 결과

| 항목 | 예상값 | 설명 |
|------|--------|------|
| 총 가우시안 수 | ~250,000 | 일반적인 3DGS 모델 |
| 노이즈 가우시안 수 | ~3,000-10,000 | 전체의 1-4% |
| 처리 시간 | ~10-30분 | 뷰 수와 가우시안 수에 비례 |

---

## 하이퍼파라미터 가이드

| 파라미터 | 기본값 | 범위 | 영향 |
|----------|--------|------|------|
| `n_clusters` | 5 | 3-10 | 많을수록 다양한 시점, 적을수록 집중적 |
| `views_per_cluster` | 5 | 3-10 | 많을수록 정확, 느림 |
| `vote_threshold` | 0.5 | 0.3-0.7 | 높을수록 엄격 (FP 감소, FN 증가) |
| `mahalanobis_threshold` | 3.0 | 2.0-4.0 | 높을수록 넓은 범위 포함 |

---

## 다음 단계 (Step 8 이후)

1. **노이즈 가우시안 분석**
   - 노이즈 가우시안의 특성 분석 (크기, 불투명도, 위치)
   - 노이즈 패턴 분류 (floater, blur, artifact type)

2. **자동 정제 (Auto-refinement)**
   - 노이즈 가우시안 자동 제거
   - 품질 개선된 모델 재생성

3. **학습 중 노이즈 감지**
   - 학습 과정에서 실시간 노이즈 감지
   - 노이즈 가우시안에 대한 regularization

---

## 참고 자료

- **마할라노비스 거리**: 공분산을 고려한 거리 측정
- **Ray-Ellipsoid Intersection**: [Real-Time Rendering 4th Ed.](http://www.realtimerendering.com/)
- **3DGS 가우시안 구조**: `scene/gaussian_model.py` 참조

---

작성일: 2024-11-28
상태: 설계 완료, 구현 예정
