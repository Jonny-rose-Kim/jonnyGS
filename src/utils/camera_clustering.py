"""
Camera clustering utilities for Step 7.

카메라들을 공간적으로 클러스터링하여 밀집된 영역에서 novel view를 생성합니다.
이를 통해 hallucination 영역을 피하고 신뢰할 수 있는 노이즈 가우시안 교집합을 찾습니다.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict


def extract_camera_positions(cameras) -> np.ndarray:
    """
    카메라 리스트에서 위치 벡터 추출

    Args:
        cameras: 카메라 객체 리스트

    Returns:
        positions: [N, 3] numpy array
    """
    positions = []
    for cam in cameras:
        if isinstance(cam.camera_center, torch.Tensor):
            pos = cam.camera_center.cpu().numpy()
        else:
            pos = np.array(cam.camera_center)
        positions.append(pos)
    return np.array(positions)


def cluster_cameras_kmeans(cameras, n_clusters: int = 5, random_state: int = 42) -> List[List]:
    """
    K-means 클러스터링으로 카메라 그룹화

    Args:
        cameras: 카메라 객체 리스트
        n_clusters: 클러스터 수
        random_state: 랜덤 시드

    Returns:
        clusters: List[List[Camera]] - 클러스터별 카메라 그룹
    """
    from sklearn.cluster import KMeans

    positions = extract_camera_positions(cameras)

    # 클러스터 수가 카메라 수보다 많으면 조정
    n_clusters = min(n_clusters, len(cameras))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(positions)

    # 클러스터별 그룹화
    clusters = [[] for _ in range(n_clusters)]
    for cam, label in zip(cameras, labels):
        clusters[label].append(cam)

    return clusters


def cluster_cameras_dbscan(cameras, eps: float = None, min_samples: int = 3) -> List[List]:
    """
    DBSCAN 클러스터링으로 카메라 그룹화 (밀도 기반)

    Args:
        cameras: 카메라 객체 리스트
        eps: 이웃 반경 (None이면 자동 계산)
        min_samples: 클러스터 최소 샘플 수

    Returns:
        clusters: List[List[Camera]] - 클러스터별 카메라 그룹 (노이즈 제외)
    """
    from sklearn.cluster import DBSCAN

    positions = extract_camera_positions(cameras)

    # eps 자동 계산: 평균 최근접 이웃 거리 사용
    if eps is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(positions)
        distances, _ = nn.kneighbors(positions)
        eps = np.mean(distances[:, -1]) * 1.5  # 마지막 이웃까지의 평균 거리 * 1.5

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(positions)

    # 클러스터별 그룹화 (노이즈 라벨 -1 제외)
    cluster_dict = defaultdict(list)
    for cam, label in zip(cameras, labels):
        if label >= 0:  # 노이즈 제외
            cluster_dict[label].append(cam)

    clusters = list(cluster_dict.values())

    return clusters


def select_dense_clusters(clusters, min_cameras: int = 5, top_k: int = None) -> List[List]:
    """
    밀집된 클러스터 선택

    Args:
        clusters: 클러스터 리스트
        min_cameras: 최소 카메라 수
        top_k: 상위 k개 클러스터만 선택 (None이면 전체)

    Returns:
        selected_clusters: 선택된 클러스터 리스트
    """
    # 최소 카메라 수 필터링
    valid_clusters = [c for c in clusters if len(c) >= min_cameras]

    # 카메라 수 기준 정렬 (내림차순)
    valid_clusters.sort(key=len, reverse=True)

    # 상위 k개 선택
    if top_k is not None:
        valid_clusters = valid_clusters[:top_k]

    return valid_clusters


def compute_cluster_density(cluster) -> float:
    """
    클러스터의 밀도 계산 (카메라 간 평균 거리의 역수)

    Args:
        cluster: 카메라 리스트

    Returns:
        density: 밀도 값 (높을수록 밀집)
    """
    if len(cluster) < 2:
        return 0.0

    positions = extract_camera_positions(cluster)

    # 모든 쌍의 거리 계산
    from scipy.spatial.distance import pdist
    distances = pdist(positions)

    if len(distances) == 0:
        return 0.0

    mean_distance = np.mean(distances)
    if mean_distance < 1e-6:
        return float('inf')

    return 1.0 / mean_distance


def compute_cluster_compactness(cluster) -> float:
    """
    클러스터의 컴팩트함 계산 (중심으로부터의 평균 거리)

    Args:
        cluster: 카메라 리스트

    Returns:
        compactness: 컴팩트함 값 (낮을수록 밀집)
    """
    if len(cluster) < 2:
        return float('inf')

    positions = extract_camera_positions(cluster)
    centroid = np.mean(positions, axis=0)

    distances = np.linalg.norm(positions - centroid, axis=1)
    return np.mean(distances)


def find_adjacent_pairs_in_cluster(cluster, max_pairs: int = None) -> List[Tuple]:
    """
    클러스터 내에서 인접한 카메라 쌍 찾기

    Args:
        cluster: 카메라 리스트
        max_pairs: 최대 쌍 수 (None이면 전체)

    Returns:
        pairs: List[(cam1, cam2, distance)] - 거리순 정렬된 카메라 쌍
    """
    if len(cluster) < 2:
        return []

    positions = extract_camera_positions(cluster)

    pairs = []
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            dist = np.linalg.norm(positions[i] - positions[j])
            pairs.append((cluster[i], cluster[j], dist))

    # 거리순 정렬 (가까운 쌍 우선)
    pairs.sort(key=lambda x: x[2])

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    return pairs


def get_cluster_statistics(clusters) -> dict:
    """
    클러스터들의 통계 정보 계산

    Args:
        clusters: 클러스터 리스트

    Returns:
        stats: 통계 딕셔너리
    """
    stats = {
        'num_clusters': len(clusters),
        'total_cameras': sum(len(c) for c in clusters),
        'cameras_per_cluster': [len(c) for c in clusters],
        'densities': [compute_cluster_density(c) for c in clusters],
        'compactness': [compute_cluster_compactness(c) for c in clusters],
    }

    if clusters:
        stats['mean_cameras'] = np.mean(stats['cameras_per_cluster'])
        stats['mean_density'] = np.mean([d for d in stats['densities'] if d != float('inf')])
        stats['mean_compactness'] = np.mean([c for c in stats['compactness'] if c != float('inf')])

    return stats


def visualize_clusters(clusters, save_path: str = None):
    """
    클러스터 시각화 (3D plot)

    Args:
        clusters: 클러스터 리스트
        save_path: 저장 경로 (None이면 화면에 표시)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    for idx, (cluster, color) in enumerate(zip(clusters, colors)):
        positions = extract_camera_positions(cluster)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c=[color], label=f'Cluster {idx} ({len(cluster)} cams)', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Camera Clusters')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Convenience function
def cluster_and_select_cameras(
    cameras,
    method: str = 'kmeans',
    n_clusters: int = 5,
    min_cameras_per_cluster: int = 5,
    top_k_clusters: int = 3,
    **kwargs
) -> Tuple[List[List], dict]:
    """
    카메라 클러스터링 및 선택 통합 함수

    Args:
        cameras: 카메라 객체 리스트
        method: 'kmeans' 또는 'dbscan'
        n_clusters: 클러스터 수 (kmeans용)
        min_cameras_per_cluster: 클러스터당 최소 카메라 수
        top_k_clusters: 선택할 상위 클러스터 수
        **kwargs: 클러스터링 알고리즘 추가 파라미터

    Returns:
        selected_clusters: 선택된 클러스터 리스트
        stats: 통계 정보
    """
    print(f"카메라 클러스터링 시작 (method={method}, n_cameras={len(cameras)})")

    # 클러스터링
    if method == 'kmeans':
        clusters = cluster_cameras_kmeans(cameras, n_clusters=n_clusters, **kwargs)
    elif method == 'dbscan':
        clusters = cluster_cameras_dbscan(cameras, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    print(f"  클러스터 수: {len(clusters)}")
    print(f"  클러스터별 카메라 수: {[len(c) for c in clusters]}")

    # 밀집 클러스터 선택
    selected = select_dense_clusters(
        clusters,
        min_cameras=min_cameras_per_cluster,
        top_k=top_k_clusters
    )

    print(f"  선택된 클러스터 수: {len(selected)}")
    print(f"  선택된 클러스터별 카메라 수: {[len(c) for c in selected]}")

    # 통계 계산
    stats = get_cluster_statistics(selected)

    return selected, stats
