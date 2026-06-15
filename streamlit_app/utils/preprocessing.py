import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import logging

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_classification(image: Image.Image, target_size: int = 224) -> torch.Tensor:
    """Preprocess image for classification models."""
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)


def preprocess_segmentation(image: Image.Image, target_size: int = 512) -> tuple:
    """
    Preprocess image for segmentation models.

    Returns:
        (input_tensor, resized_image): Normalized tensor and PIL resized image
    """
    image = image.convert("RGB")
    resized_image = image.resize((target_size, target_size), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    input_tensor = transform(resized_image).unsqueeze(0)

    return input_tensor, resized_image


def validate_medical_image(image: Image.Image, max_size: int = 2048) -> bool:
    """Validate medical image for reasonable dimensions."""
    if image.mode not in ["RGB", "L", "RGBA"]:
        return False
    if max(image.size) > max_size or min(image.size) < 64:
        return False
    return True


def load_image_safe(file_path_or_upload) -> Image.Image:
    """Load image from file path or uploaded file object."""
    try:
        if isinstance(file_path_or_upload, str):
            image = Image.open(file_path_or_upload)
        else:
            image = Image.open(file_path_or_upload)

        return image.convert("RGB") if image.mode != "RGB" else image
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise ValueError(f"Failed to load image: {str(e)}")


def get_image_statistics(image: Image.Image) -> dict:
    """Get basic statistics about the image."""
    img_array = np.array(image)
    return {
        "width": image.width,
        "height": image.height,
        "aspect_ratio": image.width / image.height,
        "mean_brightness": float(np.mean(img_array)),
    }


def resize_image_to_target(image: Image.Image, target_size: int = 512) -> Image.Image:
    """Resize image to target size with proper interpolation."""
    return image.resize((target_size, target_size), Image.BILINEAR)


def resize_mask_to_original(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize mask back to original image size using interpolation.

    Args:
        mask: Numpy array of shape (H, W) or (1, H, W)
        target_shape: Target (height, width)

    Returns:
        Resized mask matching target_shape
    """
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)

    if mask.shape[:2] == target_shape:
        return mask

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    resized = mask_img.resize(target_shape[::-1], Image.BILINEAR)
    return np.array(resized) / 255.0

