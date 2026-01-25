"""
Noise detector model inference utilities.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.simple_unet import SimpleUNet, DeepUNet


class NoiseDetector:
    """Wrapper for noise detection model inference"""

    def __init__(self, checkpoint_path, device='cuda', image_size=256, model_type='auto'):
        """
        Initialize detector.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
            image_size: Input image size for model
            model_type: 'simple', 'deep', or 'auto' (auto-detect from checkpoint)
        """
        self.device = device
        self.image_size = image_size

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Auto-detect model type from checkpoint
        if model_type == 'auto':
            state_dict = checkpoint['model_state_dict']
            # DeepUNet uses 'encoders' and 'decoders', SimpleUNet uses 'enc1', 'dec1' etc.
            if any('encoders' in key for key in state_dict.keys()):
                model_type = 'deep'
            else:
                model_type = 'simple'

        # Load model
        if model_type == 'deep':
            self.model = DeepUNet(in_channels=3, out_channels=1)
            print(f"✓ Using DeepUNet model")
        else:
            self.model = SimpleUNet(in_channels=3, out_channels=1)
            print(f"✓ Using SimpleUNet model")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        print(f"✓ Loaded detector model from {checkpoint_path}")
        # Try both 'best_iou' and 'iou' keys
        best_iou = checkpoint.get('best_iou', checkpoint.get('iou', None))
        if best_iou is not None:
            print(f"  - Best IoU: {best_iou:.4f}")
        else:
            print(f"  - Best IoU: N/A")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")

    @torch.no_grad()
    def predict_from_tensor(self, image_tensor):
        """
        Predict noise mask from image tensor.

        Args:
            image_tensor: Image tensor [C, H, W] or [B, C, H, W], values in [0, 1]

        Returns:
            mask_tensor: Binary mask tensor [1, H, W] or [B, 1, H, W], values in [0, 1]
        """
        # Handle single image
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Resize if needed
        B, C, H, W = image_tensor.shape
        if H != self.image_size or W != self.image_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )

        # Move to device
        image_tensor = image_tensor.to(self.device)

        # Predict
        mask_tensor = self.model(image_tensor)

        # Resize back to original size if needed
        if H != self.image_size or W != self.image_size:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        return mask_tensor

    @torch.no_grad()
    def predict_from_image(self, image, return_numpy=False):
        """
        Predict noise mask from PIL image.

        Args:
            image: PIL Image (RGB)
            return_numpy: If True, return numpy array instead of tensor

        Returns:
            mask: Binary mask (tensor or numpy array), values in [0, 1]
        """
        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [3, H, W]

        # Predict
        mask_tensor = self.predict_from_tensor(image_tensor)
        mask_tensor = mask_tensor.squeeze(0)  # Remove batch dim [1, H, W]

        if return_numpy:
            return mask_tensor.cpu().numpy()
        else:
            return mask_tensor

    @torch.no_grad()
    def predict_from_file(self, image_path, return_numpy=False):
        """
        Predict noise mask from image file.

        Args:
            image_path: Path to image file
            return_numpy: If True, return numpy array instead of tensor

        Returns:
            mask: Binary mask (tensor or numpy array), values in [0, 1]
        """
        image = Image.open(image_path).convert('RGB')
        return self.predict_from_image(image, return_numpy=return_numpy)

    def get_binary_mask(self, mask_tensor, threshold=0.5):
        """
        Convert continuous mask to binary mask.

        Args:
            mask_tensor: Mask tensor with values in [0, 1]
            threshold: Threshold for binarization

        Returns:
            Binary mask tensor with values 0 or 1
        """
        return (mask_tensor > threshold).float()


def load_detector(checkpoint_path, device='cuda', image_size=256):
    """
    Convenience function to load detector.

    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device to run inference on
        image_size: Input image size for model

    Returns:
        NoiseDetector object
    """
    return NoiseDetector(checkpoint_path, device, image_size)
