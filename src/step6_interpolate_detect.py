#!/usr/bin/env python3
"""
Step 6: Interpolated View Rendering with Noise Detection

이 스크립트는:
1. 125장으로 학습된 3DGS 모델에서 인접 카메라 쌍을 찾습니다
2. 각 쌍의 중점에서 novel view를 렌더링합니다
3. 학습된 노이즈 감지 모델을 적용합니다
4. 결과를 시각화하고 저장합니다

입력:
  - output/{scene_name}/ (125장으로 학습된 3DGS 모델)
  - experiments/{scene_name}/checkpoints/best_model.pth (학습된 detector)

출력:
  - data/processed/{scene_name}/interpolated_results/
    ├── interpolated_0000/
    │   ├── rendered.png
    │   ├── mask.png
    │   ├── overlay.png
    │   ├── visualization.png
    │   └── metadata.json
    └── ...
"""

import torch
import sys
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm
from os import makedirs
import torchvision

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gaussian_renderer import render, GaussianModel
from scene import Scene
from src.utils.camera_interpolation import create_interpolated_views
from src.utils.detector_inference import load_detector
from src.utils.visualization_utils import save_visualization, tensor_to_pil


def load_model_and_cameras(model_path, source_path, sh_degree=3):
    """
    Load 3DGS model and all cameras.

    Args:
        model_path: Path to trained 3DGS model
        source_path: Path to source data (for cameras)
        sh_degree: Spherical harmonics degree

    Returns:
        (gaussians, cameras, scene)
    """
    print("Loading 3DGS model and cameras...")

    class Args:
        def __init__(self):
            self.model_path = str(model_path)
            self.source_path = str(source_path)
            self.sh_degree = sh_degree
            self.white_background = False
            self.images = "images"
            self.resolution = -1
            self.data_device = "cpu"
            self.eval = False
            self.depths = ""
            self.train_test_exp = False

    args = Args()

    # Load Gaussian model
    gaussians = GaussianModel(sh_degree)

    # Load scene
    scene = Scene(args, gaussians, load_iteration=30000, shuffle=False)

    # Get all cameras
    cameras = scene.getTrainCameras()

    print(f"✓ Loaded model with {len(cameras)} cameras")

    return gaussians, cameras, scene


def render_interpolated_view(gaussians, interp_cam, background):
    """
    Render from interpolated camera.

    Args:
        gaussians: Gaussian model
        interp_cam: Interpolated camera
        background: Background tensor

    Returns:
        Rendered image tensor [3, H, W]
    """
    class SimplePipeline:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.antialiasing = False

    pipeline = SimplePipeline()

    # Render
    rendering = render(interp_cam, gaussians, pipeline, background)["render"]

    return rendering


def main():
    parser = argparse.ArgumentParser(description='Step 6: Interpolated View with Noise Detection')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--model_path', default=None,
                       help='Path to 125-image trained model (default: output/{scene})')
    parser.add_argument('--detector_checkpoint', default=None,
                       help='Path to detector checkpoint (default: experiments/{scene}/checkpoints/best_model.pth)')
    parser.add_argument('--data_root', default='./data/360_v2', help='Data root directory')
    parser.add_argument('--output_root', default='./data/processed', help='Output root directory')
    parser.add_argument('--num_interpolations', type=int, default=1,
                       help='Number of interpolated views per camera pair (1 = midpoint only)')
    parser.add_argument('--max_pairs', type=int, default=20,
                       help='Maximum number of camera pairs to process')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for detector model')
    parser.add_argument('--sh_degree', type=int, default=3,
                       help='Spherical harmonics degree')
    args = parser.parse_args()

    print("=" * 80)
    print("Step 6: Interpolated View Rendering + Noise Detection")
    print("=" * 80)
    print(f"Scene: {args.scene}")
    print(f"Interpolations per pair: {args.num_interpolations}")
    print(f"Max pairs: {args.max_pairs}")
    print("=" * 80 + "\n")

    # Setup paths
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path("output") / args.scene

    if args.detector_checkpoint:
        detector_path = Path(args.detector_checkpoint)
    else:
        detector_path = Path("experiments") / args.scene / "checkpoints" / "best_model.pth"

    source_path = Path(args.data_root) / args.scene
    output_dir = Path(args.output_root) / args.scene / "interpolated_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not detector_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {detector_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_path}")

    print(f"Model path: {model_path}")
    print(f"Detector path: {detector_path}")
    print(f"Source path: {source_path}")
    print(f"Output dir: {output_dir}\n")

    # Load model and cameras
    gaussians, cameras, scene = load_model_and_cameras(
        model_path, source_path, sh_degree=args.sh_degree
    )

    # Load detector
    detector = load_detector(detector_path, device='cuda', image_size=args.image_size)
    print()

    # Create interpolated views
    print(f"Creating interpolated views...")
    interpolated_views = create_interpolated_views(
        cameras,
        num_interpolations=args.num_interpolations,
        max_pairs=args.max_pairs
    )
    print(f"✓ Created {len(interpolated_views)} interpolated views\n")

    # Setup background
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Process each interpolated view
    print("Rendering and detecting noise...")
    results = []

    for idx, (interp_cam, cam1, cam2, t) in enumerate(tqdm(interpolated_views, desc="Processing")):
        # Render from interpolated camera
        with torch.no_grad():
            rendered = render_interpolated_view(gaussians, interp_cam, background)

        # Predict noise mask
        mask = detector.predict_from_tensor(rendered)

        # Compute statistics
        mask_np = mask.cpu().numpy()
        noise_ratio = mask_np.mean()
        noise_pixels = (mask_np > 0.5).sum()
        total_pixels = mask_np.size

        # Save individual results
        result_dir = output_dir / f"interpolated_{idx:04d}"
        result_dir.mkdir(exist_ok=True)

        # Save rendered image
        rendered_path = result_dir / "rendered.png"
        torchvision.utils.save_image(rendered, str(rendered_path))

        # Save mask
        mask_path = result_dir / "mask.png"
        torchvision.utils.save_image(mask, str(mask_path))

        # Save visualization
        vis_path = result_dir / "visualization.png"
        save_visualization(rendered, mask, vis_path, show_overlay=True)

        # Save metadata
        metadata = {
            'index': idx,
            'camera_1': cam1.image_name,
            'camera_2': cam2.image_name,
            'interpolation_t': float(t),
            'noise_ratio': float(noise_ratio),
            'noise_pixels': int(noise_pixels),
            'total_pixels': int(total_pixels),
            'camera_1_center': cam1.camera_center.cpu().tolist(),
            'camera_2_center': cam2.camera_center.cpu().tolist(),
        }

        metadata_path = result_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        results.append(metadata)

    # Save summary
    summary = {
        'scene': args.scene,
        'num_interpolated_views': len(interpolated_views),
        'num_interpolations_per_pair': args.num_interpolations,
        'max_pairs': args.max_pairs,
        'model_path': str(model_path),
        'detector_path': str(detector_path),
        'results': results,
        'statistics': {
            'mean_noise_ratio': float(np.mean([r['noise_ratio'] for r in results])),
            'std_noise_ratio': float(np.std([r['noise_ratio'] for r in results])),
            'min_noise_ratio': float(np.min([r['noise_ratio'] for r in results])),
            'max_noise_ratio': float(np.max([r['noise_ratio'] for r in results])),
        }
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Step 6 완료!")
    print("=" * 80)
    print(f"처리된 interpolated views: {len(interpolated_views)}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"\n노이즈 통계:")
    print(f"  평균 noise ratio: {summary['statistics']['mean_noise_ratio']:.4f}")
    print(f"  표준편차: {summary['statistics']['std_noise_ratio']:.4f}")
    print(f"  최소: {summary['statistics']['min_noise_ratio']:.4f}")
    print(f"  최대: {summary['statistics']['max_noise_ratio']:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
