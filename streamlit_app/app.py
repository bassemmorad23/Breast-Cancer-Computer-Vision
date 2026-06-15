import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Medical AI Demo for Breast Cancer Detection v1.0"
    }
)

st.markdown("""
<style>
    .main-header {
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #34495e;
        font-size: 1.2em;
        margin-bottom: 1em;
    }
    .info-box {
        padding: 1em;
        border-radius: 0.5em;
        background-color: #ecf0f1;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏥 Cancer Detection AI")
st.sidebar.write("---")

st.markdown("<div class='main-header'>🏥 Breast Cancer Detection & Segmentation</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>AI-Powered Medical Image Analysis</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Classification Models", "7", "ResNet-50")
with col2:
    st.metric("Segmentation Models", "2", "SegFormer + U-Net")
with col3:
    st.metric("Accuracy", "94.57%", "ResNet-50")

st.write("---")

st.sidebar.info(
    "### 🔬 Medical AI Demo\n\n"
    "This application demonstrates breast cancer detection using:\n\n"
    "📊 **Classification**: ResNet50, EfficientNet, ViT, Swin, DeiT\n\n"
    "🎭 **Segmentation**: SegFormer, U-Net\n\n"
    "⚠️ **Disclaimer**: For demonstration purposes only. Not for clinical use."
)

with st.expander("📖 How to Use", expanded=False):
    st.write("""
    ### Classification
    1. Navigate to **Classification** page
    2. Upload an ultrasound image
    3. Select a model
    4. Run inference to get predictions with confidence scores

    ### Segmentation
    1. Navigate to **Segmentation** page
    2. Upload an ultrasound image
    3. Choose U-Net or SegFormer
    4. Adjust threshold and run segmentation
    5. View mask, overlay, and metrics

    ### Comparison
    1. Navigate to **Model Comparison** page
    2. Browse performance metrics
    3. View interactive visualizations
    4. Read insights and recommendations
    """)

with st.expander("⚠️ Important Disclaimer", expanded=False):
    st.warning("""
    **This application is for research and demonstration purposes only.**

    - NOT intended for clinical diagnosis
    - NOT a replacement for professional medical evaluation
    - Results should be validated by qualified healthcare professionals
    - Use at your own risk

    Always consult qualified healthcare professionals for medical decisions.
    """)

st.write("---")
st.write("### 🚀 Get Started")
st.write("Select a page from the sidebar to begin exploring the application.")
