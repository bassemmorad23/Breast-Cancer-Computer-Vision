# 🏥Breast Cancer AI Deployment System (Classification & Segmentation)

A comprehensive deep learning pipeline for breast cancer analysis combining **classification**, **segmentation**, and a **production-ready Streamlit web application**. This project benchmarks state-of-the-art neural networks on medical imaging data with both research training pipelines and an interactive inference interface.

---

## 📌 Project Overview

This repository contains:

1. **Training Pipeline** — Research-grade scripts for training 7 classification models + 2 segmentation models
2. **FastAPI Inference Server** — Real-time REST API for ResNet-50 predictions
3. **Streamlit Web App** — Interactive UI for classification, segmentation, and model comparison
4. **Medical Imaging Analysis** — Breast cancer detection from ultrasound images

| Task               | Models                                                                            |
| ------------------ | ---------------------------------------------------------------------------------- |
| **Classification** | Custom CNN, ResNet50, MobileNet-V2, EfficientNet-B0, ViT, Swin Transformer, DeiT |
| **Segmentation**   | U-Net (baseline), SegFormer (nvidia/mit-b2)                                       |

---

## 📁 Project Structure

```
Final-project/
│
├── 📂 streamlit_app/                           # Production-ready web application
│   ├── app.py                                  # Main navigation & home page
│   ├── requirements.txt                        # Streamlit app dependencies
│   ├── README.md                               # Streamlit app quick reference
│   ├── .streamlit/
│   │   └── config.toml                         # Streamlit configuration
│   ├── .gitignore                              # Git ignore rules
│   │
│   ├── 📂 pages/                               # Multi-page Streamlit app
│   │   ├── __init__.py
│   │   ├── 1_Classification.py                 # Classification inference page
│   │   ├── 2_Segmentation.py                   # Segmentation inference page
│   │   └── 3_Comparison.py                     # Model comparison dashboard
│   │
│   └── 📂 utils/                               # Utility modules
│       ├── __init__.py
│       ├── inference.py                        # Model loading & inference
│       ├── preprocessing.py                    # Image preprocessing
│       └── visualization.py                    # Plotting & visualization
│
├── 📂 checkpoints/                             # Model weights (~1.9GB total)
│   ├── best_model_resnet50.pth                 # ResNet-50 (214MB)
│   ├── best_model_EfficientNet_B0.pth          # EfficientNet-B0 (41MB)
│   ├── best_model_mobilenet_v2.pth             # MobileNetV2 (18MB)
│   ├── best_model_vit.pth                      # ViT (570MB)
│   ├── best_model_swin.pth                     # Swin Transformer (223MB)
│   ├── best_model_deit.pth                     # DeiT (570MB)
│   ├── best_model_cnn.pth                      # Custom CNN (1.5MB)
│   ├── best_seg_segformer.pth                  # SegFormer (290MB)
│   └── best_model_unet.pth                     # U-Net (optional)
│
├── 📂 breast-ultrasound-images-dataset/        # Dataset (not tracked by git)
│   └── Dataset_BUSI_with_GT/
│       ├── benign/
│       │   ├── images/
│       │   └── masks/
│       └── malignant/
│           ├── images/
│           └── masks/
│
├── 📂 Training & Model Definition Scripts
│   ├── model_CNN.py                            # Custom 3-block CNN architecture
│   ├── Transfer_Learning.py                    # ResNet50 fine-tuning
│   ├── Transfer_Learning_2.py                  # EfficientNet fine-tuning
│   ├── Transfer_Learning_3.py                  # MobileNetV2 fine-tuning
│   ├── HF_Models.py                            # HuggingFace model loaders (ViT, Swin, DeiT)
│   ├── model_seg.py                            # Segmentation architectures (U-Net, SegFormer)
│   │
│   ├── Training Scripts
│   │   ├── train.py                            # Train custom CNN
│   │   ├── train_TL.py                         # Train transfer learning models
│   │   ├── train_HF.py                         # Train HuggingFace models
│   │   └── train_seg.py                        # Train segmentation models
│   │
│   ├── Training Utilities
│   │   ├── engine.py                           # Classification train/val loop
│   │   └── engine_seg.py                       # Segmentation train/val loop
│   │
│   ├── Data & Utilities
│   │   ├── data_setup.py                       # Classification data loading
│   │   ├── data_setup_seg.py                   # Segmentation data loading
│   │   ├── utils_py.py                         # Classification utilities
│   │   └── utils_seg.py                        # Segmentation utilities
│   │
│   └── Evaluation & Comparison
│       ├── evalate.py                          # Evaluate classification models
│       ├── evalate_seg.py                      # Evaluate segmentation models
│       ├── evalute_HF.py                       # Evaluate ViT, Swin, DeiT
│       └── comparison.py                       # Compare all models
│
├── 📂 API & Inference
│   └── main.py                                 # FastAPI inference server
│       ├── GET /          — API info
│       ├── POST /classify — inference endpoint
│       └── GET /docs      — Swagger UI
│
├── 📂 Documentation Files
│   ├── README.md                               # Main project README (this file)
│   ├── BUGFIX_REPORT.md                        # Critical bug fixes documentation
│   ├── PROJECT_SUMMARY.md                      # Implementation summary
│   ├── GETTING_STARTED.md                      # Setup guide
│   ├── CONFIG.md                               # Configuration reference
│   ├── DEPLOYMENT_CHECKLIST.md                 # Pre-deployment checklist
│   └── EDA.ipynb                               # Exploratory data analysis
│
├── 📂 Configuration & Setup
│   ├── requirements.txt                        # Root project dependencies
│   ├── .env                                    # Environment variables (not tracked)
│   └── .gitignore                              # Git ignore rules
│
└── 📂 Analysis & Results
    └── results/
        └── master_comparison.png                # Generated comparison plot
```

---

## 🎯 Key Features

### Classification Module
- **7 Model Architectures** — CNN, ResNet50, EfficientNet, MobileNetV2, ViT, Swin, DeiT
- **Real-time Inference** — Fast predictions with confidence scores (50–100ms GPU)
- **Visualization** — Classification results with confidence gauges
- **Probability Breakdown** — Detailed class predictions and model metadata

### Segmentation Module
- **Lesion Boundary Detection** — Precise tumor localization
- **Dual Models** — U-Net and SegFormer with configurable thresholds
- **Metrics** — Dice score and IoU calculations
- **Multi-view Visualization** — Original image, segmentation mask, and overlay blend

### Streamlit Web Application
- **Multi-page Interface** — Classification, Segmentation, Model Comparison
- **Interactive Dashboards** — Performance metrics, visualizations, insights
- **Production Deployment** — Docker, Streamlit Cloud, or local server
- **Error Handling** — Robust checkpoint loading, shape matching, GPU OOM management

### FastAPI REST API
- **Classification Endpoint** (`/classify`) — Upload image, get prediction
- **Auto Documentation** — Swagger UI at `/docs`
- **JSON Responses** — Prediction + confidence scores

---

## 🏗️ System Architecture

```
Image Input
    ↓
[Preprocessing] — Resize (224px classification / 512px segmentation), normalize
    ↓
[Model Loading] — Load checkpoint (supports nested & flat formats)
    ↓
[Inference] — Forward pass (PyTorch, CUDA-optimized)
    ↓
[Post-processing] — Sigmoid, threshold, shape matching
    ↓
[Visualization] — Plots, overlays, metrics
    ↓
Output (Web UI / API Response)
```

---

## 📊 Model Performance

### Classification Results

| Model            | Accuracy  | F1 Score  | Precision | Recall    | Type        | Parameters |
| ---------------- | --------- | --------- | --------- | --------- | ----------- | ---------- |
| **ResNet-50**    | **94.57%**| **94.50%**| **94.62%**| **94.57%**| CNN         | 25.5M |
| Swin Transformer | 93.80%    | 93.72%    | 93.85%    | 93.80%    | Transformer | 28M |
| DeiT             | 93.02%    | 92.96%    | 93.11%    | 93.02%    | Transformer | 86M |
| ViT              | 92.25%    | 92.18%    | 92.34%    | 92.25%    | Transformer | 86M |
| EfficientNet-B0  | 89.15%    | 89.08%    | 89.26%    | 89.15%    | CNN         | 5.3M |
| MobileNet-V2     | 87.60%    | 87.54%    | 87.71%    | 87.60%    | CNN         | 3.5M |
| Custom CNN       | 72.31%    | 71.45%    | 71.18%    | 72.31%    | CNN         | 2.1M |

**Best Model**: ResNet-50 achieves 94.57% accuracy with the optimal speed/accuracy tradeoff.

### Segmentation Results

| Model         | Architecture       | Input Size | Dice Score | IoU    |
| ------------- | ------------------ | ---------- | ---------- | ------ |
| **SegFormer** | Transformer-based  | 512×512    | **76.49%** | **69.56%** |
| U-Net         | CNN-based          | Variable   | 69.83%     | 62.47% |

**Best Model**: SegFormer achieves a 76.49% Dice score.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB+ recommended)
- ~5GB disk space (including model checkpoints)

### Installation

```bash
# Clone repository
git clone https://github.com/basemmorad23/Breast-Cancer-Detection.git
cd "Final project"

# Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 How to Run

### 1. Streamlit Web Application

```bash
cd streamlit_app
streamlit run app.py
```

Opens at `http://localhost:8501`

**Features:**
- Upload ultrasound image → Classification with 7 models
- Segment lesion boundaries → Visualize mask + overlay
- Compare model performance → View metrics & insights
- Interactive threshold adjustment
- Real-time inference

**Production server:**
```bash
streamlit run app.py \
  --server.port 8080 \
  --server.address 0.0.0.0 \
  --logger.level=warning
```

### 2. FastAPI Inference Server

```bash
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`

**Endpoints:**
- `GET /` — API info
- `POST /classify` — Upload image, get prediction
- `GET /docs` — Interactive Swagger UI

### 3. Training & Evaluation

```bash
# Train custom CNN
python train.py

# Train transfer learning models
python train_TL.py

# Train Vision Transformers
python train_HF.py

# Train segmentation models
python train_seg.py

# Evaluate all models
python evalate.py
python evalate_seg.py
python evalute_HF.py

# Compare and visualize results
python comparison.py
```

---

## 🔧 Technical Details

### Classification Models

- **Custom CNN** — 3-block convolutional architecture with BatchNorm, GlobalAvgPool, Dropout
- **ResNet50** — Pretrained ImageNet; last layer (layer4) unfrozen
- **MobileNet-V2** — Lightweight; classifier head fine-tuned
- **EfficientNet-B0** — Efficient scaling; classifier head fine-tuned
- **ViT** — Last 2 transformer blocks unfrozen
- **Swin Transformer** — Last 2 blocks of final stage unfrozen
- **DeiT** — Last 2 encoder blocks + both classifiers unfrozen

### Segmentation Models

- **U-Net** — Classic encoder-decoder with skip connections
- **SegFormer** — Transformer-based with frozen backbone + unfrozen decode head

### Training Configuration

- **Optimizer** — Adam (lr=1e-5)
- **Loss (Classification)** — CrossEntropyLoss with class weights
- **Loss (Segmentation)** — Dice + BCE
- **Scheduler** — ReduceLROnPlateau (patience=3, factor=0.5)
- **Epochs** — 50
- **Checkpoint Strategy** — Save best model on lowest validation loss

---

## 📈 Performance Benchmarks

### Inference Speed (GPU: NVIDIA RTX 3090)
- Classification: 50–100ms per image
- Segmentation: 200–500ms per image

### Memory Usage
- ResNet-50: ~500MB VRAM
- SegFormer: ~800MB VRAM
- Total system: ~2GB minimum

### Optimization Tips
- Use GPU for production (≈10x faster than CPU)
- Cache models in memory for consecutive predictions
- Batch process multiple images for throughput
- Use MobileNetV2 for resource-constrained environments

---

## 🛠️ Tech Stack

**Core**
- PyTorch 2.0.1 / TorchVision 0.15.2 — Deep learning framework
- Transformers 4.34.0 — HuggingFace models
- Streamlit 1.28+ — Web application

**Image Processing**
- Pillow — Image loading/manipulation
- OpenCV — Image processing
- NumPy — Numerical computing

**Visualization**
- Matplotlib — Plotting
- Pandas — Data handling

**API & Serving**
- FastAPI — REST API framework
- Uvicorn — ASGI server

**Development**
- scikit-learn — Metrics

---

## 📦 Model Checkpoints

Checkpoints should be located in `checkpoints/`:

```
checkpoints/
├── best_model_resnet50.pth
├── best_model_EfficientNet_B0.pth
├── best_model_mobilenet_v2.pth
├── best_model_vit.pth
├── best_model_swin.pth
├── best_model_deit.pth
├── best_model_cnn.pth
├── best_seg_segformer.pth
└── best_model_unet.pth (optional)
```

---

## 🐛 Troubleshooting

### Model Loading Fails
```
Error: "Failed to load ResNet50"
→ Ensure checkpoint exists: checkpoints/best_model_resnet50.pth
→ Check PyTorch version compatibility
```

### Image Upload Fails
```
Error: "Invalid image dimensions"
→ Upload JPG or PNG only
→ Image size should be between 64px and 2048px
```

### GPU Not Detected
```
→ Install CUDA and cuDNN
→ Verify: python -c "import torch; print(torch.cuda.is_available())"
→ Falls back to CPU automatically
```

### Out of Memory
```
→ Use smaller model (MobileNetV2)
→ Reduce batch size / image input size
→ Close other applications
```

---

## 🚢 Deployment

### Docker
```bash
docker build -t breast-cancer-ai .
docker run -p 8501:8501 breast-cancer-ai
```

Example Dockerfile:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

### Streamlit Cloud
1. Push to GitHub
2. Deploy via https://share.streamlit.io
3. Connect repo and select `streamlit_app/app.py` as entry point

---

## 🔐 Security & Compliance

- ⚠️ Demonstration tool — NOT for clinical use
- No patient data is stored or transmitted
- Images are processed locally
- HIPAA compliance not guaranteed
- Use only for research and educational purposes

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Docs](https://docs.streamlit.io)
- [FastAPI](https://fastapi.tiangolo.com)
- [Project MONAI](https://github.com/Project-MONAI/MONAI)

---

## 👨‍💻 Author

**Basem Morad**
- Mechatronics Engineering Graduate
- AI/ML Engineer | Computer Vision Specialist
- [GitHub](https://github.com/basemmorad23)

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## ⚖️ Disclaimer

**This application is for research and demonstration purposes only.**

- ⚠️ NOT intended for clinical diagnosis
- ⚠️ NOT a replacement for professional medical evaluation
- Results should be validated by qualified healthcare professionals
- Use at your own risk

---

**Last Updated**: June 2026 | Version 1.0.0
