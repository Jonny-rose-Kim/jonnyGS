"""
Visualization utilities for noise detection results.
"""

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_heatmap_overlay(image, mask, alpha=0.5, colormap='hot'):
    """
    Create heatmap overlay of mask on image.

    Args:
        image: PIL Image or numpy array (H, W, 3), values in [0, 255] or [0, 1]
        mask: numpy array (H, W) or (1, H, W), values in [0, 1]
        alpha: Transparency of overlay
        colormap: Matplotlib colormap name

    Returns:
        PIL Image with overlay
    """
    # Convert image to numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()

    # Normalize image to [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np.astype(np.float32) / 255.0

    # Handle mask dimensions
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if mask.ndim == 3:
        mask = mask[0]  # Remove channel dimension

    # Ensure same size
    H, W = image_np.shape[:2]
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask).resize((W, H)))

    # Create colormap
    cmap = plt.get_cmap(colormap)
    mask_colored = cmap(mask)[:, :, :3]  # RGB only

    # Blend
    overlay = (1 - alpha) * image_np + alpha * mask_colored

    # Convert back to PIL
    overlay = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay)


def create_side_by_side(images, titles=None, gap=10):
    """
    Create side-by-side visualization of multiple images.

    Args:
        images: List of PIL Images or numpy arrays
        titles: Optional list of titles
        gap: Gap width between images in pixels

    Returns:
        PIL Image with side-by-side layout
    """
    # Convert all to PIL Images
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            if img.ndim == 2:  # Grayscale
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img, mode='L').convert('RGB')
            else:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
        pil_images.append(img)

    # Get dimensions
    widths, heights = zip(*(img.size for img in pil_images))
    max_height = max(heights)
    total_width = sum(widths) + gap * (len(pil_images) - 1)

    # Create canvas
    canvas = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    # Paste images
    x_offset = 0
    for img in pil_images:
        canvas.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    # Add titles if provided
    if titles is not None:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)

        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        x_offset = 0
        for i, (img, title) in enumerate(zip(pil_images, titles)):
            # Draw text with background
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = x_offset + (img.width - text_width) // 2
            text_y = 10

            # Background rectangle
            draw.rectangle(
                [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
                fill=(255, 255, 255, 200)
            )

            # Text
            draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)

            x_offset += img.width + gap

    return canvas


def create_mask_overlay(image, mask, mask_color=(255, 0, 0), alpha=0.5):
    """
    Create colored mask overlay on image.

    Args:
        image: PIL Image or numpy array
        mask: Binary mask (0 or 1)
        mask_color: RGB color for mask regions
        alpha: Transparency

    Returns:
        PIL Image with mask overlay
    """
    # Convert to numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()

    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)

    # Handle mask
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Remove batch dimension if present
    while mask.ndim > 2:
        mask = mask[0]

    # Ensure same size
    H, W = image_np.shape[:2]
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((W, H)))

    # Binarize mask
    mask_binary = (mask > 0.5).astype(np.float32)

    # Create colored overlay
    overlay = image_np.copy()
    for c in range(3):
        overlay[:, :, c] = (
            (1 - alpha) * image_np[:, :, c] +
            alpha * mask_binary * mask_color[c]
        )

    return Image.fromarray(overlay.astype(np.uint8))


def tensor_to_pil(tensor):
    """
    Convert torch tensor to PIL Image.

    Args:
        tensor: Torch tensor [C, H, W], [H, W], or [1, C, H, W], values in [0, 1]

    Returns:
        PIL Image
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor[0]  # [1, C, H, W] -> [C, H, W]

    # Handle dimensions
    if tensor.ndim == 3:
        if tensor.shape[0] in [1, 3]:  # [C, H, W]
            tensor = np.transpose(tensor, (1, 2, 0))
        # Now [H, W, C]

    # Squeeze single channel
    if tensor.ndim == 3 and tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)

    # Convert to [0, 255]
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).astype(np.uint8)

    # Create PIL Image
    if tensor.ndim == 2:  # Grayscale
        return Image.fromarray(tensor, mode='L')
    else:  # RGB
        return Image.fromarray(tensor, mode='RGB')


def save_visualization(rendered, mask, output_path, show_overlay=True):
    """
    Save complete visualization with rendered image, mask, and overlay.

    Args:
        rendered: Rendered image (PIL or tensor)
        mask: Predicted mask (tensor or numpy)
        output_path: Path to save visualization
        show_overlay: If True, include overlay visualization
    """
    # Convert to PIL
    if isinstance(rendered, torch.Tensor):
        rendered_pil = tensor_to_pil(rendered)
    else:
        rendered_pil = rendered

    mask_pil = tensor_to_pil(mask)

    # Create visualizations
    images = [rendered_pil, mask_pil]
    titles = ['Rendered', 'Noise Mask']

    if show_overlay:
        overlay = create_mask_overlay(rendered_pil, mask, mask_color=(255, 50, 50), alpha=0.4)
        images.append(overlay)
        titles.append('Overlay')

    # Create side-by-side
    result = create_side_by_side(images, titles=titles)

    # Save
    result.save(output_path)
    print(f"  Saved: {output_path}")
