#!/usr/bin/env python3
"""
Step 1: Train/Test Split & Data Preparation

입력:
  - data/360_v2/{scene_name}/input/ (원본 이미지)
  - data/360_v2/{scene_name}/sparse/ (COLMAP 정보)

출력:
  - data/processed/{scene_name}/images/ (train 이미지만 복사)
  - data/processed/{scene_name}/sparse/ (train 카메라만 포함된 COLMAP sparse)
  - data/processed/{scene_name}/splits/train_indices.json
  - data/processed/{scene_name}/splits/test_indices.json
  - data/processed/{scene_name}/splits/test_cameras.json (test 카메라 pose)
"""

import json
import shutil
from pathlib import Path
import argparse
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.read_write_model import read_model, write_model


def create_split(image_dir, skip_frames=3):
    """
    이미지를 train/test로 split.

    Args:
        image_dir: 이미지 디렉토리 경로
        skip_frames: Test로 사용할 frame 간격 (매 skip_frames+1번째가 test)

    Returns:
        train_indices: Train 이미지 인덱스 리스트
        test_indices: Test 이미지 인덱스 리스트
        image_names: 모든 이미지 파일명 리스트
    """
    # 모든 이미지 파일 찾기
    image_dir = Path(image_dir)
    image_files = (
        sorted(image_dir.glob("*.jpg")) +
        sorted(image_dir.glob("*.JPG")) +
        sorted(image_dir.glob("*.png")) +
        sorted(image_dir.glob("*.PNG"))
    )

    total_frames = len(image_files)
    print(f"총 이미지 수: {total_frames}장")

    if total_frames == 0:
        raise ValueError(f"이미지를 찾을 수 없습니다: {image_dir}")

    # Split 생성: 매 (skip_frames+1)번째가 test
    test_indices = list(range(0, total_frames, skip_frames + 1))
    train_indices = [i for i in range(total_frames) if i not in test_indices]

    print(f"Train: {len(train_indices)}장 ({len(train_indices)/total_frames*100:.1f}%)")
    print(f"Test:  {len(test_indices)}장 ({len(test_indices)/total_frames*100:.1f}%)")

    image_names = [f.name for f in image_files]

    return train_indices, test_indices, image_names


def copy_train_images(source_image_dir, train_indices, image_names, dest_image_dir):
    """
    Train 이미지만 복사.

    Args:
        source_image_dir: 원본 이미지 디렉토리
        train_indices: Train 인덱스 리스트
        image_names: 모든 이미지 파일명 리스트
        dest_image_dir: 목적지 디렉토리
    """
    source_image_dir = Path(source_image_dir)
    dest_image_dir = Path(dest_image_dir)
    dest_image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTrain 이미지 복사 중 ({len(train_indices)}장)...")

    copied_count = 0
    for idx in train_indices:
        src = source_image_dir / image_names[idx]
        dst = dest_image_dir / image_names[idx]
        shutil.copy2(src, dst)
        copied_count += 1

    print(f"✓ {copied_count}개 Train 이미지 복사 완료: {dest_image_dir}")


def filter_colmap_sparse(source_sparse_dir, dest_sparse_dir, train_image_names, test_image_names):
    """
    COLMAP sparse에서 train 이미지에 해당하는 카메라만 필터링.

    Args:
        source_sparse_dir: 원본 COLMAP sparse (125장)
        dest_sparse_dir: 출력 sparse (93장)
        train_image_names: Train 이미지 파일명 리스트
        test_image_names: Test 이미지 파일명 리스트

    Returns:
        test_cameras: Test 카메라 정보 딕셔너리 (Step 3에서 렌더링할 때 사용)
    """
    source_sparse_dir = Path(source_sparse_dir)
    dest_sparse_dir = Path(dest_sparse_dir)

    # 원본 COLMAP 모델 읽기
    print(f"\nCOLMAP sparse 읽는 중: {source_sparse_dir}")
    cameras, images, points3D = read_model(str(source_sparse_dir / "0"), ext=".bin")

    print(f"  원본: {len(cameras)} cameras, {len(images)} images, {len(points3D)} points3D")

    # Train 이미지 이름을 set으로 변환 (빠른 검색)
    train_image_set = set(train_image_names)
    test_image_set = set(test_image_names)

    # Train 이미지만 필터링
    filtered_images = {}
    test_cameras_info = {}

    for img_id, img in images.items():
        if img.name in train_image_set:
            filtered_images[img_id] = img
        elif img.name in test_image_set:
            # Test 카메라 정보 저장 (Step 3에서 렌더링할 때 사용)
            test_cameras_info[img.name] = {
                'image_id': int(img_id),
                'camera_id': int(img.camera_id),
                'qvec': img.qvec.tolist(),
                'tvec': img.tvec.tolist(),
                'name': img.name,
                # Camera intrinsics 추가
                'width': int(cameras[img.camera_id].width),
                'height': int(cameras[img.camera_id].height),
                'model': cameras[img.camera_id].model,
                'params': cameras[img.camera_id].params.tolist()
            }

    print(f"  필터링: {len(filtered_images)} train images, {len(test_cameras_info)} test cameras")

    # Points3D 필터링 (train 이미지에서 관측된 것만)
    train_image_ids = set(filtered_images.keys())
    filtered_points3D = {}

    for point_id, point in points3D.items():
        # 이 3D 포인트가 train 이미지에서 관측되는지 확인
        observed_in_train = any(img_id in train_image_ids for img_id in point.image_ids)
        if observed_in_train:
            # Train 이미지에서만 관측된 정보로 필터링
            train_mask = [img_id in train_image_ids for img_id in point.image_ids]
            filtered_image_ids = point.image_ids[train_mask]
            filtered_point2D_idxs = point.point2D_idxs[train_mask]

            if len(filtered_image_ids) >= 2:  # 최소 2개 이미지에서 관측되어야 유효
                from utils.read_write_model import Point3D
                filtered_points3D[point_id] = Point3D(
                    id=point.id,
                    xyz=point.xyz,
                    rgb=point.rgb,
                    error=point.error,
                    image_ids=filtered_image_ids,
                    point2D_idxs=filtered_point2D_idxs
                )

    print(f"  필터링된 3D points: {len(filtered_points3D)} (원본: {len(points3D)})")

    # Cameras는 그대로 유지 (intrinsics 정보)
    # 필터링된 모델 저장
    dest_sparse_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_sparse_dir / "0"
    output_path.mkdir(parents=True, exist_ok=True)

    write_model(cameras, filtered_images, filtered_points3D, str(output_path), ext=".bin")
    print(f"✓ 필터링된 COLMAP sparse 저장 완료: {dest_sparse_dir}")

    return test_cameras_info


def save_split(output_dir, train_indices, test_indices, image_names, test_cameras_info):
    """Split 정보를 JSON으로 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train indices 저장
    train_data = {
        'indices': train_indices,
        'count': len(train_indices),
        'images': [image_names[i] for i in train_indices]
    }
    with open(output_dir / 'train_indices.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    # Test indices 저장
    test_data = {
        'indices': test_indices,
        'count': len(test_indices),
        'images': [image_names[i] for i in test_indices]
    }
    with open(output_dir / 'test_indices.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    # Test 카메라 정보 저장 (Step 3에서 렌더링할 때 사용)
    with open(output_dir / 'test_cameras.json', 'w') as f:
        json.dump(test_cameras_info, f, indent=2)

    print(f"✓ Split JSON 저장 완료: {output_dir}")
    print(f"  - train_indices.json: {len(train_indices)} images")
    print(f"  - test_indices.json: {len(test_indices)} images")
    print(f"  - test_cameras.json: {len(test_cameras_info)} cameras")


def main():
    parser = argparse.ArgumentParser(description='Step 1: Train/Test Split & Data Preparation')
    parser.add_argument('--scene', required=True, help='Scene name (e.g., stump)')
    parser.add_argument('--data_root', default='./data/360_v2', help='Data root directory')
    parser.add_argument('--output_root', default='./data/processed', help='Output root directory')
    parser.add_argument('--skip_frames', type=int, default=3, help='Skip frames for test split')
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Train/Test Split & Data Preparation")
    print("=" * 60)
    print(f"Scene: {args.scene}")
    print(f"Skip frames: {args.skip_frames}")
    print("=" * 60 + "\n")

    # 경로 설정
    source_image_dir = Path(args.data_root) / args.scene / "images" #"input"
    source_sparse_dir = Path(args.data_root) / args.scene / "sparse"

    dest_root = Path(args.output_root) / args.scene
    dest_image_dir = dest_root / "images"
    dest_sparse_dir = dest_root / "sparse"
    splits_dir = dest_root / "splits"

    # 검증
    if not source_image_dir.exists():
        raise FileNotFoundError(f"이미지 디렉토리를 찾을 수 없습니다: {source_image_dir}")

    if not source_sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse 디렉토리를 찾을 수 없습니다: {source_sparse_dir}")

    # 1. Split 생성
    train_indices, test_indices, image_names = create_split(
        source_image_dir,
        skip_frames=args.skip_frames
    )

    # 2. Train 이미지만 복사
    copy_train_images(source_image_dir, train_indices, image_names, dest_image_dir)

    # 3. COLMAP sparse 필터링 (train 카메라만 남김)
    train_image_names = [image_names[i] for i in train_indices]
    test_image_names = [image_names[i] for i in test_indices]
    test_cameras_info = filter_colmap_sparse(
        source_sparse_dir,
        dest_sparse_dir,
        train_image_names,
        test_image_names
    )

    # 4. Split JSON 저장 (test 카메라 정보 포함)
    save_split(splits_dir, train_indices, test_indices, image_names, test_cameras_info)

    print("\n" + "=" * 60)
    print("Step 1 완료!")
    print("=" * 60)
    print(f"\n출력 위치: {dest_root}")
    print(f"  - images/: {len(train_indices)}장 (train only)")
    print(f"  - sparse/: COLMAP 정보 (train {len(train_indices)}장만 필터링됨)")
    print(f"  - splits/: JSON 파일 (train/test indices, test 카메라 pose)")
    print("\n✅ 엄격한 실험 격리:")
    print(f"   - sparse/에는 train {len(train_indices)}장 카메라만 포함")
    print(f"   - points3D.bin은 train에서 관측된 3D 점만 포함")
    print(f"   - test 카메라 pose는 test_cameras.json에 별도 저장")
    print(f"   - 3DGS는 train {len(train_indices)}장만 사용하여 학습")
    print("=" * 60)


if __name__ == '__main__':
    main()
