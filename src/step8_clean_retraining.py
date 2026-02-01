#!/usr/bin/env python3
"""
Step 8: Clean Initialization Retraining

Noise가 제거된 Clean PLY를 초기화로 사용하여 3DGS 재학습.

핵심 가설:
- Noise Gaussian이 제거된 상태에서 재학습하면, 빈 공간에 새로운 Gaussian이 올바른 위치에 생성됨
- Local minima에 갇혀있던 Noise Gaussian의 "예산"이 품질 개선에 재할당됨

Usage:
    python src/step8_clean_retraining.py \\
        --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \\
        --source_path data/360_v2/stump \\
        --output_path output/stump_retrained \\
        --iterations 15000

    # 테스트용 (짧은 iteration)
    python src/step8_clean_retraining.py \\
        --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \\
        --source_path data/360_v2/stump \\
        --output_path output/stump_retrained_test \\
        --iterations 100 \\
        --dry_run
"""

import argparse
import subprocess
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime


def validate_paths(clean_ply: Path, source_path: Path, source_model: Path = None):
    """입력 경로 검증"""
    errors = []

    if not clean_ply.exists():
        errors.append(f"Clean PLY not found: {clean_ply}")

    if not source_path.exists():
        errors.append(f"Source data not found: {source_path}")

    # COLMAP 데이터 확인
    sparse_dir = source_path / "sparse"
    if not sparse_dir.exists():
        errors.append(f"COLMAP sparse directory not found: {sparse_dir}")

    if source_model and not source_model.exists():
        errors.append(f"Source model not found: {source_model}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    return True


def get_ply_info(ply_path: Path) -> dict:
    """PLY 파일 정보 추출"""
    info = {"path": str(ply_path), "exists": ply_path.exists()}

    if info["exists"]:
        info["size_mb"] = ply_path.stat().st_size / (1024 * 1024)

        # vertex count 파싱
        try:
            with open(ply_path, 'rb') as f:
                header = b""
                while True:
                    line = f.readline()
                    header += line
                    if b"end_header" in line:
                        break

                header_str = header.decode('utf-8', errors='ignore')
                for line in header_str.split('\n'):
                    if line.startswith('element vertex'):
                        info["vertex_count"] = int(line.split()[-1])
                        break
        except Exception as e:
            info["error"] = str(e)

    return info


def prepare_output_directory(
    output_path: Path,
    source_model: Path = None,
    force: bool = False
) -> Path:
    """출력 디렉토리 준비"""
    if output_path.exists():
        if force:
            print(f"  Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            print(f"  WARNING: Output directory already exists: {output_path}")
            print(f"  Use --force to overwrite")

    output_path.mkdir(parents=True, exist_ok=True)

    # cfg_args 복사 (source_model에서)
    if source_model:
        cfg_args_src = source_model / "cfg_args"
        if cfg_args_src.exists():
            shutil.copy(cfg_args_src, output_path / "cfg_args")
            print(f"  Copied cfg_args from {source_model}")

        # cameras.json 복사
        cameras_src = source_model / "cameras.json"
        if cameras_src.exists():
            shutil.copy(cameras_src, output_path / "cameras.json")
            print(f"  Copied cameras.json from {source_model}")

    return output_path


def run_training(
    source_path: Path,
    output_path: Path,
    clean_ply: Path,
    iterations: int,
    eval_mode: bool = True,
    save_iterations: list = None,
    test_iterations: list = None,
    extra_args: list = None,
    dry_run: bool = False
) -> int:
    """3DGS 학습 실행"""

    # 기본 저장/테스트 iteration 설정
    if save_iterations is None:
        if iterations <= 7000:
            save_iterations = [iterations]
        else:
            save_iterations = [7000, iterations]

    if test_iterations is None:
        if iterations <= 7000:
            test_iterations = [iterations]
        else:
            test_iterations = [7000, iterations]

    # 명령어 구성
    cmd = [
        "python", "train.py",
        "-s", str(source_path),
        "-m", str(output_path),
        "--iterations", str(iterations),
        "--load_ply", str(clean_ply),
        "--save_iterations", *[str(i) for i in save_iterations],
        "--test_iterations", *[str(i) for i in test_iterations],
    ]

    if eval_mode:
        cmd.append("--eval")

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n  Command: {' '.join(cmd)}")

    if dry_run:
        print("\n  [DRY RUN] Training would be executed with the above command")
        return 0

    print(f"\n  Starting training...")
    print("-" * 60)

    try:
        # train.py는 프로젝트 루트에 있으므로 cwd 설정
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            cmd,
            check=True,
            cwd=project_root
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: Training failed with exit code {e.returncode}")
        return e.returncode


def save_run_config(output_path: Path, args: argparse.Namespace, clean_ply_info: dict):
    """실행 설정 저장"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "script": "step8_clean_retraining.py",
        "arguments": vars(args),
        "clean_ply_info": clean_ply_info,
    }

    config_path = output_path / "step8_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"  Saved run config to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Step 8: Clean Initialization Retraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard retraining (15k iterations)
  python src/step8_clean_retraining.py \\
      --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \\
      --source_path data/360_v2/stump \\
      --output_path output/stump_retrained_15k \\
      --iterations 15000

  # Quick test (100 iterations, dry run)
  python src/step8_clean_retraining.py \\
      --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \\
      --source_path data/360_v2/stump \\
      --output_path output/stump_retrained_test \\
      --iterations 100 \\
      --dry_run

  # Full retraining (30k iterations, same as baseline)
  python src/step8_clean_retraining.py \\
      --clean_ply data/processed/stump/noise_gaussians/clean_gaussians.ply \\
      --source_path data/360_v2/stump \\
      --output_path output/stump_retrained_30k \\
      --iterations 30000
        """
    )

    # Required arguments
    parser.add_argument('--clean_ply', type=str, required=True,
                       help='Path to clean_gaussians.ply (noise removed)')
    parser.add_argument('--source_path', type=str, required=True,
                       help='Path to source data (e.g., data/360_v2/stump)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for retrained model')

    # Optional arguments
    parser.add_argument('--source_model', type=str, default=None,
                       help='Original trained model path (for copying cfg_args, cameras.json)')
    parser.add_argument('--iterations', type=int, default=15000,
                       help='Number of training iterations (default: 15000)')
    parser.add_argument('--save_iterations', type=int, nargs='+', default=None,
                       help='Iterations to save model (default: auto)')
    parser.add_argument('--test_iterations', type=int, nargs='+', default=None,
                       help='Iterations to run evaluation (default: auto)')

    # Flags
    parser.add_argument('--eval', action='store_true', default=True,
                       help='Use evaluation split (default: True)')
    parser.add_argument('--no_eval', dest='eval', action='store_false',
                       help='Disable evaluation split')
    parser.add_argument('--force', action='store_true',
                       help='Force overwrite existing output directory')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print command without executing')

    # Extra training arguments
    parser.add_argument('--extra_args', type=str, nargs='+', default=None,
                       help='Additional arguments to pass to train.py')

    args = parser.parse_args()

    # 경로 변환
    clean_ply = Path(args.clean_ply).resolve()
    source_path = Path(args.source_path).resolve()
    output_path = Path(args.output_path).resolve()
    source_model = Path(args.source_model).resolve() if args.source_model else None

    # 헤더 출력
    print("=" * 70)
    print("Step 8: Clean Initialization Retraining")
    print("=" * 70)
    print(f"Clean PLY:    {clean_ply}")
    print(f"Source Data:  {source_path}")
    print(f"Output:       {output_path}")
    print(f"Iterations:   {args.iterations}")
    print(f"Eval Mode:    {args.eval}")
    if source_model:
        print(f"Source Model: {source_model}")
    if args.dry_run:
        print(f"Mode:         DRY RUN (no actual training)")
    print("=" * 70)

    # 1. 입력 검증
    print("\n[1/4] Validating inputs...")
    validate_paths(clean_ply, source_path, source_model)

    # Clean PLY 정보 출력
    clean_ply_info = get_ply_info(clean_ply)
    print(f"  Clean PLY: {clean_ply_info.get('vertex_count', 'N/A')} gaussians, "
          f"{clean_ply_info.get('size_mb', 0):.1f} MB")

    # 2. 출력 디렉토리 준비
    print("\n[2/4] Preparing output directory...")
    prepare_output_directory(output_path, source_model, args.force)

    # 3. 설정 저장
    print("\n[3/4] Saving run configuration...")
    save_run_config(output_path, args, clean_ply_info)

    # 4. 학습 실행
    print("\n[4/4] Running training...")
    return_code = run_training(
        source_path=source_path,
        output_path=output_path,
        clean_ply=clean_ply,
        iterations=args.iterations,
        eval_mode=args.eval,
        save_iterations=args.save_iterations,
        test_iterations=args.test_iterations,
        extra_args=args.extra_args,
        dry_run=args.dry_run
    )

    # 완료 메시지
    print("\n" + "=" * 70)
    if return_code == 0:
        print("Step 8: Clean Initialization Retraining - COMPLETED")
        print("=" * 70)
        print(f"Output saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Evaluate with: python src/step8_evaluate.py --baseline output/stump --retrained {output_path}")
        print(f"  2. View with: ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m {output_path}")
    else:
        print("Step 8: Clean Initialization Retraining - FAILED")
        print("=" * 70)
        print(f"Exit code: {return_code}")

    return return_code


if __name__ == '__main__':
    sys.exit(main())
