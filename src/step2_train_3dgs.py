#!/usr/bin/env python3
"""
Step 2: 3DGS Training

입력:
  - data/processed/{scene_name}/images/ (Step 1에서 복사한 train 이미지)
  - data/processed/{scene_name}/sparse/ (COLMAP 정보)

출력:
  - data/processed/{scene_name}/3dgs_output/
    ├── point_cloud/iteration_*/point_cloud.ply
    ├── cameras.json
    └── cfg_args
"""

import subprocess
from pathlib import Path
import argparse
import sys


def train_3dgs(input_dir, output_dir, iterations=30000, project_root=None):
    """
    3DGS 학습 실행.

    Args:
        input_dir: 입력 데이터 디렉토리 (images/, sparse/ 포함)
        output_dir: 출력 디렉토리
        iterations: 학습 iteration 수
        project_root: 프로젝트 루트 디렉토리 (train.py가 있는 곳)
    """
    print(f"\n3DGS 학습 시작 ({iterations} iterations)...")

    if project_root is None:
        project_root = Path(__file__).parent.parent

    train_script = Path(project_root) / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"train.py를 찾을 수 없습니다: {train_script}")

    # 3DGS train.py 호출
    # images/에 93장만 있으므로 자동으로 93장만 학습됨
    cmd = [
        sys.executable, str(train_script),
        "-s", str(input_dir),
        "-m", str(output_dir),
        "--iterations", str(iterations),
        "--test_iterations", "-1",  # Test evaluation 비활성화
        "--save_iterations", str(iterations),
        "--disable_viewer"  # Network viewer 비활성화
    ]

    print(f"실행 명령: {' '.join(cmd)}")
    print(f"작업 디렉토리: {project_root}\n")

    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(project_root)
        )
        print("\n✓ 3DGS 학습 완료")

    except subprocess.CalledProcessError:
        print(f"❌ 3DGS 학습 실패")
        raise


def main():
    parser = argparse.ArgumentParser(description='Step 2: 3DGS Training')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--output_root', default='./data/processed', help='Output root directory')
    parser.add_argument('--iterations', type=int, default=30000, help='Training iterations')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training if model already exists')
    parser.add_argument('--force', action='store_true',
                       help='Force retraining even if model exists')
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: 3DGS Training")
    print("=" * 60)
    print(f"Scene: {args.scene}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60 + "\n")

    # 경로 설정
    input_dir = Path(args.output_root) / args.scene
    output_dir = input_dir / "3dgs_output"

    # 검증
    images_dir = input_dir / "images"
    sparse_dir = input_dir / "sparse"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"이미지 디렉토리를 찾을 수 없습니다: {images_dir}\n"
            "먼저 Step 1을 실행하세요."
        )

    if not sparse_dir.exists():
        raise FileNotFoundError(
            f"COLMAP sparse 디렉토리를 찾을 수 없습니다: {sparse_dir}\n"
            "먼저 Step 1을 실행하세요."
        )

    # 이미지 개수 확인
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")) + \
                  list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG"))
    print(f"입력 이미지: {len(image_files)}장 (images/)")
    print(f"COLMAP sparse: {sparse_dir}")
    print(f"출력 디렉토리: {output_dir}\n")

    # 이미 학습된 모델이 있는지 확인
    model_exists = (output_dir / "point_cloud").exists()
    if model_exists and not args.force:
        if args.skip_training:
            print(f"✓ 학습된 모델이 이미 존재합니다: {output_dir}")
            print("  --force 옵션으로 재학습할 수 있습니다.")
            print("\n" + "=" * 60)
            print("Step 2 완료! (학습 생략)")
            print("=" * 60)
            return
        else:
            response = input(f"\n학습된 모델이 이미 존재합니다: {output_dir}\n"
                           f"재학습하시겠습니까? [y/N]: ")
            if response.lower() != 'y':
                print("학습을 건너뜁니다.")
                print("\n" + "=" * 60)
                print("Step 2 완료! (학습 생략)")
                print("=" * 60)
                return

    # 3DGS 학습
    train_3dgs(input_dir, output_dir, iterations=args.iterations)

    print(f"\n출력 저장 위치: {output_dir}")

    print("\n" + "=" * 60)
    print("Step 2 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
