import streamlit as st
import torch
import numpy as np
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.inference import load_segmentation_model, DEVICE
from utils.preprocessing import preprocess_segmentation, load_image_safe, validate_medical_image
from utils.visualization import plot_segmentation_result, calculate_dice_score, calculate_iou

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("🎭 Segmentation")
st.write("Segment lesion boundaries with U-Net or SegFormer.")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader(
        "Upload ultrasound image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="seg_uploader",
        help="Supported: JPG, PNG (max 2MB)"
    )

with col2:
    st.subheader("Model Selection")
    model_options = ["U-Net", "SegFormer"]
    selected_model = st.selectbox(
        "Choose segmentation model:",
        model_options,
        help="SegFormer uses transformer architecture"
    )

st.write("---")

if uploaded_file is not None:
    try:
        image = load_image_safe(uploaded_file)

        if not validate_medical_image(image):
            st.error("❌ Invalid image dimensions. Please upload a proper medical image.")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Input Image")
                st.image(image, use_column_width=True)

            with col2:
                st.subheader("Segmentation Settings")

                threshold = st.slider(
                    "Mask threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Probability threshold for lesion detection"
                )

                st.write("---")

                if st.button("🚀 Run Segmentation", key="segment_btn", use_container_width=True):
                    try:
                        with st.spinner(f"Loading {selected_model}..."):
                            model = load_segmentation_model(selected_model)

                        with st.spinner(f"Segmenting with {selected_model}..."):
                            input_tensor, resized_image = preprocess_segmentation(image, target_size=512)
                            input_tensor = input_tensor.to(DEVICE)

                            with torch.no_grad():
                                if selected_model == "SegFormer":
                                    output = model(pixel_values=input_tensor).logits
                                    mask = output[:, :1, :, :].squeeze(0).squeeze(0).cpu().numpy()
                                else:
                                    mask = model(input_tensor).squeeze().cpu().numpy()

                            if mask.ndim == 3:
                                mask = mask.squeeze()

                            mask_probs = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

                            binary_mask = (mask_probs > threshold).astype(np.uint8)
                            lesion_area = np.sum(binary_mask)
                            total_area = binary_mask.size
                            coverage = (lesion_area / total_area) * 100

                            dice = calculate_dice_score(mask_probs, threshold)
                            iou = calculate_iou(mask_probs, threshold)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Coverage", f"{coverage:.1f}%")
                            with col2:
                                st.metric("Dice Score", f"{dice:.3f}")
                            with col3:
                                st.metric("IoU", f"{iou:.3f}")

                            st.divider()

                            fig = plot_segmentation_result(resized_image, mask_probs)
                            st.pyplot(fig, use_container_width=True)

                            with st.expander("📊 Detailed Metrics"):
                                st.write(f"**Model**: {selected_model}")
                                st.write(f"**Input Size**: 512 × 512 px")
                                st.write(f"**Lesion Pixels**: {lesion_area:,} / {total_area:,}")
                                st.write(f"**Mask Min**: {mask.min():.4f}")
                                st.write(f"**Mask Max**: {mask.max():.4f}")
                                st.write(f"**Mask Mean**: {mask.mean():.4f}")

                    except RuntimeError as e:
                        st.error(f"⚠️ Model Error: {str(e)}\n\nPlease check that model checkpoints exist in the checkpoints folder.")
                        logger.error(f"Model loading error: {str(e)}")

                    except torch.cuda.OutOfMemoryError:
                        st.error("❌ GPU out of memory. Try a smaller model or restart the app.")
                        logger.error("CUDA OOM error")

                    except ValueError as e:
                        st.error(f"❌ Shape mismatch error: {str(e)}\nThis may indicate an incompatible image size.")
                        logger.error(f"Shape error: {str(e)}")

                    except Exception as e:
                        st.error(f"❌ Segmentation failed: {str(e)}\nPlease try with another image or model.")
                        logger.error(f"Unexpected error: {str(e)}", exc_info=True)

    except ValueError as e:
        st.error(f"❌ Error loading image: {str(e)}")
        logger.error(f"Image loading error: {str(e)}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)

else:
    st.info("👆 Upload an ultrasound image to get started.")
