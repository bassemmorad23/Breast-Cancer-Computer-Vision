import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import logging

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"

logger = logging.getLogger(__name__)


class HFWrapper(nn.Module):
    """Wrapper for HuggingFace classification models."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Classic U-Net for binary segmentation."""
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool    = nn.MaxPool2d(2, 2)
        self.upconv  = nn.ModuleList()
        self.decoder = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.upconv.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i, (up, dec) in enumerate(zip(self.upconv, self.decoder)):
            x    = up(x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)


def _load_checkpoint(checkpoint_path: Path, model: nn.Module, device: torch.device):
    """
    Load checkpoint safely, handling both nested and flat state dicts.

    Supports:
    - Nested format: {"model_state_dict": {...}, "epoch": ..., "optimizer_state_dict": ...}
    - Flat format: {...} (direct state dict)
    """
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            logger.error(f"Invalid checkpoint format: {type(checkpoint)}")
            return False

        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Successfully loaded checkpoint: {checkpoint_path.name}")
        return True

    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path.name}: {str(e)}")
        return False


def load_classification_model(model_name: str):
    """
    Load pretrained classification model with proper checkpoint handling.

    Args:
        model_name: One of "ResNet50", "EfficientNet", "MobileNetV2", "ViT", "Swin", "DeiT"

    Returns:
        model: Loaded model in eval mode on appropriate device

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        if model_name == "ResNet50":
            from torchvision.models import resnet50
            model = resnet50(pretrained=False)
            model.fc = nn.Linear(2048, 2)
            checkpoint_path = CHECKPOINT_DIR / "best_model_resnet50.pth"

        elif model_name == "EfficientNet":
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(1280, 2)
            checkpoint_path = CHECKPOINT_DIR / "best_model_EfficientNet_B0.pth"

        elif model_name == "MobileNetV2":
            from torchvision.models import mobilenet_v2
            model = mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(1280, 2)
            checkpoint_path = CHECKPOINT_DIR / "best_model_mobilenet_v2.pth"

        elif model_name == "ViT":
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            model = HFWrapper(model)
            checkpoint_path = CHECKPOINT_DIR / "best_model_vit.pth"

        elif model_name == "Swin":
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            model = HFWrapper(model)
            checkpoint_path = CHECKPOINT_DIR / "best_model_swin.pth"

        elif model_name == "DeiT":
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(
                "facebook/deit-base-distilled-patch16-224",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            model = HFWrapper(model)
            checkpoint_path = CHECKPOINT_DIR / "best_model_deit.pth"

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(DEVICE)
        _load_checkpoint(checkpoint_path, model, DEVICE)
        model.eval()

        return model

    except Exception as e:
        error_msg = f"Failed to load {model_name}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def load_segmentation_model(model_name: str):
    """
    Load pretrained segmentation model with proper checkpoint handling.

    Args:
        model_name: One of "U-Net" or "SegFormer"

    Returns:
        model: Loaded model in eval mode on appropriate device

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        if model_name == "U-Net":
            model = UNet(in_channels=3, features=[64, 128, 256, 512])
            checkpoint_path = CHECKPOINT_DIR / "best_model_unet.pth"
            model = model.to(DEVICE)
            _load_checkpoint(checkpoint_path, model, DEVICE)

        elif model_name == "SegFormer":
            from transformers import SegformerForSemanticSegmentation
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b2",
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
            model = model.to(DEVICE)
            checkpoint_path = CHECKPOINT_DIR / "best_seg_segformer.pth"
            _load_checkpoint(checkpoint_path, model, DEVICE)

        else:
            raise ValueError(f"Unknown segmentation model: {model_name}")

        model.eval()
        return model

    except Exception as e:
        error_msg = f"Failed to load {model_name}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
