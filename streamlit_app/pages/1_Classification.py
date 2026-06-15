import streamlit as st
import torch
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.inference import load_classification_model, DEVICE
from utils.preprocessing import preprocess_classification, load_image_safe, validate_medical_image, get_image_statistics
from utils.visualization import plot_classification_result, create_confidence_gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("🔍 Classification")
st.write("Detect benign vs malignant lesions using advanced neural networks.")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader(
        "Upload ultrasound image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="Supported: JPG, PNG (max 2MB)"
    )

with col2:
    st.subheader("Model Selection")
    model_options = ["ResNet50 (Best)", "EfficientNet", "MobileNetV2", "ViT", "Swin", "DeiT"]
    selected_model = st.selectbox(
        "Choose classification model:",
        model_options,
        help="ResNet50 achieves 94.6% accuracy"
    )

st.write("---")

if uploaded_file is not None:
    try:
        image = load_image_safe(uploaded_file)

        if not validate_medical_image(image):
            st.error("❌ Invalid image dimensions. Please upload a proper medical image.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Input Image")
                st.image(image, use_column_width=True)

                stats = get_image_statistics(image)
                with st.expander("📊 Image Statistics"):
                    st.write(f"**Size**: {stats['width']} × {stats['height']} px")
                    st.write(f"**Aspect Ratio**: {stats['aspect_ratio']:.2f}")
                    st.write(f"**Mean Brightness**: {stats['mean_brightness']:.1f}")

            with col2:
                st.subheader("Analysis")

                model_name = selected_model.split(" ")[0]

                if st.button("🚀 Run Inference", key="classify_btn", use_container_width=True):
                    try:
                        with st.spinner(f"Loading {model_name}..."):
                            model = load_classification_model(model_name)

                        with st.spinner(f"Analyzing with {model_name}..."):
                            input_tensor = preprocess_classification(image, target_size=224)
                            input_tensor = input_tensor.to(DEVICE)

                            with torch.no_grad():
                                logits = model(input_tensor)
                                probs = torch.softmax(logits, dim=1)
                                confidence, pred_idx = torch.max(probs, dim=1)

                            classes = ["Benign", "Malignant"]
                            prediction = classes[pred_idx.item()]
                            confidence_score = confidence.item()

                            col_pred, col_conf = st.columns(2)

                            with col_pred:
                                color = "green" if prediction == "Benign" else "red"
                                st.markdown(
                                    f"<h3 style='color: {color};'>{prediction}</h3>",
                                    unsafe_allow_html=True
                                )

                            with col_conf:
                                st.metric("Confidence", f"{confidence_score * 100:.1f}%")

                            st.divider()

                            tab1, tab2, tab3 = st.tabs(["Result", "Gauge", "Details"])

                            with tab1:
                                fig = plot_classification_result(image, prediction.lower(), confidence_score)
                                st.pyplot(fig, use_container_width=True)

                            with tab2:
                                fig = create_confidence_gauge(confidence_score, prediction.lower())
                                st.pyplot(fig, use_container_width=True)

                            with tab3:
                                st.write("**Model Information**")
                                st.write(f"- Model: {model_name}")
                                st.write(f"- Input Size: 224 × 224 px")
                                st.write(f"- Architecture: {'Transformer' if model_name in ['ViT', 'Swin', 'DeiT'] else 'CNN'}")

                                st.write("\n**Class Probabilities**")
                                for idx, cls in enumerate(classes):
                                    prob = probs[0][idx].item()
                                    st.write(f"- {cls}: {prob * 100:.2f}%")

                    except RuntimeError as e:
                        st.error(f"⚠️ Model Error: {str(e)}\n\nPlease check that model checkpoints exist in the checkpoints folder.")
                        logger.error(f"Model loading error: {str(e)}")

                    except torch.cuda.OutOfMemoryError:
                        st.error("❌ GPU out of memory. Try a smaller model or restart the app.")
                        logger.error("CUDA OOM error")

                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}\nPlease try with another image or model.")
                        logger.error(f"Unexpected error: {str(e)}", exc_info=True)

    except ValueError as e:
        st.error(f"❌ Error loading image: {str(e)}")
        logger.error(f"Image loading error: {str(e)}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)

else:
    st.info("👆 Upload an ultrasound image to get started.")
