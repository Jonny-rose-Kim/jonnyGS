#!/usr/bin/env python3
"""
Step 3: Novel View Rendering (Custom Version)

입력:
  - output/{scene_name}/ 또는 data/processed/{scene_name}/3dgs_output/ (학습된 모델)
  - data/processed/{scene_name}/splits/test_indices.json
  - data/raw/{scene_name}/ (COLMAP sparse data)

출력:
  - data/processed/{scene_name}/rendered_views/
    ├── render_0000.png  (test index 0의 렌더링)
    ├── gt_0000.png       (test index 0의 GT)
    ├── render_0004.png  (test index 4의 렌더링)
    ├── gt_0004.png       (test index 4의 GT)
    └── ...
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.custom_render import render_specific_views


def main():
    parser = argparse.ArgumentParser(description='Step 3: Novel View Rendering')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--model_path', default=None,
                       help='Path to trained 3DGS model (default: output/{scene})')
    parser.add_argument('--data_root', default='./data/360_v2', help='Data root directory')
    parser.add_argument('--output_root', default='./data/processed', help='Output root directory')
    parser.add_argument('--sh_degree', type=int, default=3, help='Spherical harmonics degree')
    parser.add_argument('--white_background', action='store_true', help='Use white background')
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3: Novel View Rendering (Custom)")
    print("=" * 60)
    print(f"Scene: {args.scene}")
    print("=" * 60 + "\n")

    # 경로 설정
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # 기본적으로 data/processed/{scene}/3dgs_output 사용 (93장 모델)
        model_path = Path(args.output_root) / args.scene / "3dgs_output"

    source_path = Path(args.output_root) / args.scene  # Step 1에서 준비한 데이터
    test_indices_file = Path(args.output_root) / args.scene / "splits" / "test_indices.json"
    output_dir = Path(args.output_root) / args.scene / "rendered_views"

    # 검증
    if not model_path.exists():
        raise FileNotFoundError(
            f"3DGS 모델을 찾을 수 없습니다: {model_path}\n"
            "먼저 Step 2를 실행하세요."
        )

    if not test_indices_file.exists():
        raise FileNotFoundError(
            f"Test indices 파일을 찾을 수 없습니다: {test_indices_file}\n"
            "먼저 Step 1을 실행하세요."
        )

    if not source_path.exists():
        raise FileNotFoundError(f"원본 데이터를 찾을 수 없습니다: {source_path}")

    print(f"모델 경로: {model_path}")
    print(f"원본 데이터: {source_path}")
    print(f"Test indices: {test_indices_file}")
    print(f"출력 디렉토리: {output_dir}\n")

    # 커스텀 렌더링 실행
    result_dir = render_specific_views(
        model_path=model_path,
        source_path=source_path,
        test_indices_file=test_indices_file,
        output_dir=output_dir,
        sh_degree=args.sh_degree,
        white_background=args.white_background
    )

    print("\n" + "=" * 60)
    print("Step 3 완료!")
    print(f"렌더링 결과: {result_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
