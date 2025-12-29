#!/usr/bin/env python3
"""
Step 5: Pretrained Noise Detector Training (Multi-Scene)

여러 scene의 데이터를 통합하여 pretrained 모델을 학습합니다.

입력:
  - data/large_dataset_pairs/
    ├── train/
    ├── val/
    └── test/

출력:
  - experiments/pretrained/
    └── checkpoints/best_model.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.simple_unet import SimpleUNet


class NoisePairDataset(Dataset):
    """Dataset for noise detection pairs (multi-scene)"""

    def __init__(self, data_root, split='train', image_size=256):
        self.data_root = Path(data_root) / split
        self.image_size = image_size
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        if not self.data_root.exists():
            return samples

        for pair_dir in sorted(self.data_root.iterdir()):
            if not pair_dir.is_dir():
                continue

            rendered_path = pair_dir / "rendered.png"
            mask_path = pair_dir / "mask.png"

            if rendered_path.exists() and mask_path.exists():
                samples.append({
                    'rendered': rendered_path,
                    'mask': mask_path
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess
        rendered = Image.open(sample['rendered']).convert('RGB')
        rendered = rendered.resize((self.image_size, self.image_size))
        rendered = np.array(rendered).astype(np.float32) / 255.0
        rendered = torch.from_numpy(rendered).permute(2, 0, 1)  # [3, H, W]

        mask = Image.open(sample['mask']).convert('L')
        mask = mask.resize((self.image_size, self.image_size))
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        return rendered, mask


def compute_iou(pred, target, threshold=0.5):
    """Compute IoU metric"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        pred_masks = model(images)
        loss = criterion(pred_masks, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)

            pred_masks = model(images)
            loss = criterion(pred_masks, masks)

            total_loss += loss.item()
            total_iou += compute_iou(pred_masks, masks)

    return total_loss / len(loader), total_iou / len(loader)


def main():
    parser = argparse.ArgumentParser(description='Step 5: Train Pretrained Noise Detector')
    parser.add_argument('--data_root', default='./data/large_dataset_pairs',
                        help='Path to dataset with train/val/test splits')
    parser.add_argument('--exp_name', default='pretrained', help='Experiment name')
    parser.add_argument('--exp_root', default='./experiments', help='Experiment root')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Step 5: Pretrained Noise Detector Training")
    print("=" * 60)
    print(f"Data Root: {args.data_root}")
    print(f"Experiment: {args.exp_name}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Image Size: {args.image_size}")
    print("=" * 60 + "\n")

    # Paths
    data_root = Path(args.data_root)
    checkpoint_dir = Path(args.exp_root) / args.exp_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Validation
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_root}\n"
            "Ensure the data directory exists with train/val/test splits."
        )

    # Load datasets
    print("Loading datasets...")
    train_dataset = NoisePairDataset(data_root, split='train', image_size=args.image_size)
    val_dataset = NoisePairDataset(data_root, split='val', image_size=args.image_size)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples\n")

    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {data_root / 'train'}")

    if len(val_dataset) == 0:
        print("Warning: No validation samples. Using train as val.")
        val_dataset = train_dataset

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    model = SimpleUNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Training loop
    best_iou = 0.0

    print("Starting training...\n")

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_iou)

        print(f"Epoch {epoch:3d}/{args.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': best_iou
            }, checkpoint_dir / 'best_model.pth')
            print(f"  -> Best model saved (IoU: {best_iou:.4f})")

        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': val_iou
        }, checkpoint_dir / 'latest_model.pth')

    print("\n" + "=" * 60)
    print(f"Step 5 Complete! Best IoU: {best_iou:.4f}")
    print(f"Model saved: {checkpoint_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
