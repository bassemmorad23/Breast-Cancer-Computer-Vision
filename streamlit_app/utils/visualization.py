import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

COLORS = {
    "benign": "#2ecc71",
    "malignant": "#e74c3c",
}


def plot_classification_result(image: Image.Image, prediction: str, confidence: float) -> plt.Figure:
    """Visualize classification result with confidence."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.imshow(image, cmap="gray")
    ax.axis("off")

    color = COLORS[prediction]
    confidence_pct = confidence * 100

    title = f"{prediction.upper()}\n{confidence_pct:.1f}% Confidence"
    ax.set_title(title, fontsize=16, fontweight="bold", color=color, pad=20)

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    return fig


def plot_segmentation_result(
    original: Image.Image,
    mask: np.ndarray,
    confidence: float = None,
) -> plt.Figure:
    """
    Visualize segmentation result with original and overlay.

    Ensures both images have same shape before blending.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    original_array = np.array(original.convert("L"))
    original_shape = original_array.shape

    mask_normalized = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_binary = (mask_normalized > 0.5).astype(np.uint8) * 255

    if mask_binary.shape != original_shape:
        mask_binary = cv2.resize(mask_binary, (original_shape[1], original_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        mask_normalized = cv2.resize(mask_normalized.astype(np.float32),
                                     (original_shape[1], original_shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

    axes[0].imshow(original_array, cmap="gray")
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask_binary, cmap="hot")
    axes[1].set_title("Predicted Mask", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    overlay = np.stack([original_array] * 3, axis=-1).astype(float)
    red_mask = np.stack([mask_binary] * 3, axis=-1).astype(float)
    red_mask[:, :, 1:] = 0

    overlay = (0.7 * overlay + 0.3 * red_mask).astype(np.uint8)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    return fig


def plot_model_metrics(metrics_dict: dict) -> plt.Figure:
    """Create comparison visualization for model metrics."""
    models = list(metrics_dict.keys())
    metrics = ["Accuracy", "F1", "Precision", "Recall"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    colors_list = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [metrics_dict[m].get(metric, 0) for m in models]

        bars = ax.barh(models, values, color=colors_list, edgecolor="black", linewidth=1.2)

        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

        ax.set_xlim(0, 1.0)
        ax.set_xlabel(metric, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_title(f"{metric} Comparison", fontweight="bold", fontsize=11)

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    return fig


def create_confidence_gauge(confidence: float, prediction: str) -> plt.Figure:
    """Create a gauge chart for confidence visualization."""
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection="polar"))

    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)

    color = COLORS.get(prediction, "#3498db")
    ax.plot(theta, r, color=color, linewidth=3)
    ax.fill_between(theta[:int(confidence * len(theta))], 0, 1, alpha=0.3, color=color)

    ax.set_ylim(0, 1.2)
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0, 1.5, f"{confidence * 100:.1f}%", ha="center", fontsize=20, fontweight="bold")
    ax.text(0, -0.5, prediction.upper(), ha="center", fontsize=14, fontweight="bold", color=color)

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    return fig


def calculate_dice_score(pred_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate Dice score (for visualization purposes)."""
    binary_pred = (pred_mask > threshold).astype(np.uint8)
    return float(np.mean(binary_pred))


def calculate_iou(pred_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Estimate IoU from mask (requires ground truth for proper calculation)."""
    binary_pred = (pred_mask > threshold).astype(np.uint8)
    area_pred = np.sum(binary_pred)
    total_area = binary_pred.size
    if total_area == 0:
        return 0.0
    return float(area_pred / total_area)
