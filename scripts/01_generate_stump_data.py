#!/usr/bin/env python3
"""
Script 1: Generate Stump Training Data

This script generates training data for noise detection by:
1. Splitting Stump dataset into train/test views
2. Training 3DGS on training views
3. Rendering novel views
4. Generating noise masks
5. Splitting data into train/val/test sets

Usage:
    python scripts/01_generate_stump_data.py
    python scripts/01_generate_stump_data.py --config config/stump_experiment.yaml
    python scripts/01_generate_stump_data.py --skip_training  # Skip 3DGS training for testing
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation.dataset_generator import StumpDatasetGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stump dataset for noise detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full dataset (will train 3DGS for 30k iterations)
  python scripts/01_generate_stump_data.py

  # Use custom config
  python scripts/01_generate_stump_data.py --config my_config.yaml

  # Quick test (skip 3DGS training)
  python scripts/01_generate_stump_data.py --skip_training

  # Custom parameters
  python scripts/01_generate_stump_data.py --iterations 10000 --skip_frames 5
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/stump_experiment.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--stump_path',
        type=str,
        default=None,
        help='Path to Stump dataset (overrides config)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='3DGS training iterations (overrides config)'
    )
    
    parser.add_argument(
        '--skip_frames',
        type=int,
        default=None,
        help='Skip every N frames for novel views (overrides config)'
    )
    
    parser.add_argument(
        '--noise_threshold',
        type=float,
        default=None,
        help='Noise detection threshold (overrides config)'
    )
    
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip 3DGS training (for testing pipeline only)'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Extract data generation config
    data_gen_config = config.get('data_generation', {})
    dataset_split_config = config.get('dataset_split', {})
    
    # Override with command-line arguments
    stump_path = args.stump_path or data_gen_config.get('stump_path', './data/360_v2/stump')
    output_dir = args.output_dir or data_gen_config.get('output_dir', './data/processed')
    iterations = args.iterations or data_gen_config.get('iterations', 30000)
    skip_frames = args.skip_frames or data_gen_config.get('skip_frames', 3)
    noise_threshold = args.noise_threshold or data_gen_config.get('noise_threshold', 0.15)
    min_noise_size = data_gen_config.get('min_noise_size', 50)
    blur_kernel_size = data_gen_config.get('blur_kernel_size', 5)
    device = data_gen_config.get('device', 'cuda')
    resolution = data_gen_config.get('resolution', -1)
    
    # Dataset split ratios
    train_ratio = dataset_split_config.get('train_ratio', 0.8)
    val_ratio = dataset_split_config.get('val_ratio', 0.1)
    test_ratio = dataset_split_config.get('test_ratio', 0.1)
    random_seed = dataset_split_config.get('random_seed', 42)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Stump Dataset Generation Configuration")
    print("=" * 80)
    print(f"Stump path: {stump_path}")
    print(f"Output directory: {output_dir}")
    print(f"Skip frames: {skip_frames} (every {skip_frames + 1}th frame is novel)")
    print(f"3DGS iterations: {iterations}")
    print(f"Noise threshold: {noise_threshold}")
    print(f"Min noise size: {min_noise_size} pixels")
    print(f"Dataset split: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Skip training: {args.skip_training}")
    print("=" * 80)
    
    # Check if Stump dataset exists
    stump_path_obj = Path(stump_path)
    if not stump_path_obj.exists():
        print(f"\nError: Stump dataset not found at {stump_path}")
        print("\nPlease download the Stump dataset:")
        print("  1. Download from: http://storage.googleapis.com/gresearch/refraw360/360_v2.zip")
        print("  2. Extract to: data/360_v2/stump/")
        print("  3. Verify structure:")
        print("     data/360_v2/stump/")
        print("       ├── images/")
        print("       ├── sparse/0/")
        print("       └── poses_bounds.npy")
        sys.exit(1)
    
    # Verify dataset structure
    images_dir = stump_path_obj / "images"
    sparse_dir = stump_path_obj / "sparse" / "0"
    
    if not images_dir.exists():
        print(f"\nError: Images directory not found: {images_dir}")
        sys.exit(1)
    
    if not sparse_dir.exists():
        print(f"\nError: COLMAP sparse directory not found: {sparse_dir}")
        sys.exit(1)
    
    # Count images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    num_images = len(image_files)
    print(f"\nFound {num_images} images in dataset")
    
    if num_images == 0:
        print("\nError: No images found in dataset")
        sys.exit(1)
    
    # Estimate dataset split
    num_novel = num_images // (skip_frames + 1)
    num_train_novel = int(num_novel * train_ratio)
    num_val_novel = int(num_novel * val_ratio)
    num_test_novel = num_novel - num_train_novel - num_val_novel
    
    print(f"\nEstimated data split:")
    print(f"  Total images: {num_images}")
    print(f"  Training views (for 3DGS): {num_images - num_novel}")
    print(f"  Novel views: {num_novel}")
    print(f"    → Train pairs: {num_train_novel}")
    print(f"    → Val pairs: {num_val_novel}")
    print(f"    → Test pairs: {num_test_novel}")
    
    # Confirm before starting
    if not args.skip_training:
        print(f"\nThis will train 3DGS for {iterations} iterations.")
        print("This may take several hours depending on your GPU.")
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Create generator
    print("\nInitializing dataset generator...")
    generator = StumpDatasetGenerator(
        stump_path=stump_path,
        output_dir=output_dir,
        skip_frames=skip_frames,
        iterations=iterations,
        noise_threshold=noise_threshold,
        min_noise_size=min_noise_size,
        blur_kernel_size=blur_kernel_size,
        device=device,
        resolution=resolution,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Run generation
    try:
        generator.generate(skip_3dgs_training=args.skip_training)
        
        print("\n" + "=" * 80)
        print("Dataset generation completed successfully!")
        print("=" * 80)
        print(f"\nGenerated data saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Check the generated data:")
        print(f"     ls {output_dir}/train/")
        print("  2. Train the noise detection model:")
        print("     python scripts/02_train_simple.py")
        
    except Exception as e:
        print(f"\n\nError during dataset generation:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
