# 🚀 Getting Started with Streamlit App

This guide will help you set up and run the Breast Cancer Detection & Segmentation Streamlit application.

## Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** (Python package manager)
- **8GB RAM minimum** (16GB+ recommended)
- **CUDA 11.8+** (optional, for GPU acceleration)

## ⚡ Quick Start (Recommended)

### Windows
1. Double-click `run.bat` in the streamlit_app folder
2. The app will automatically:
   - Create a virtual environment (if needed)
   - Install dependencies
   - Launch the app in your browser

### macOS/Linux
```bash
chmod +x run.sh
./run.sh
```

The app will open at `http://localhost:8501`

---

## 📝 Manual Setup

### Step 1: Navigate to Project Directory
```bash
cd "C:\Users\basem\Desktop\Final project\streamlit_app"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

---

## 📦 What Gets Installed

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web framework |
| torch | 2.0.1 | Deep learning |
| torchvision | 0.15.2 | Vision models |
| transformers | 4.34.0 | Vision Transformers |
| Pillow | 10.0.0 | Image processing |
| numpy | 1.24.3 | Numerical computing |
| pandas | 2.0.3 | Data handling |
| matplotlib | 3.7.2 | Visualization |
| opencv-python | 4.8.0.74 | Image processing |
| scikit-learn | 1.3.0 | ML utilities |

**Total Installation Size**: ~2.5GB (with CUDA)

---

## 🎯 First Run

1. **Wait for models to load** (first run takes 30-60 seconds as models download)
2. **Check GPU status**: App will auto-detect CUDA
3. **Upload test image**: Try with a sample ultrasound image
4. **Run inference**: Click inference buttons to test

---

## 🛠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: "No module named 'torch'"
**Solution**: 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: App runs slowly / high CPU usage
**Solution**:
- Use GPU: Ensure CUDA is installed
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, install CUDA: https://developer.nvidia.com/cuda-toolkit

### Issue: "CUDA out of memory"
**Solution**:
- Use smaller model (MobileNetV2)
- Reduce image size
- Use CPU mode (will be slower)

### Issue: Port 8501 already in use
**Solution**:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Missing model checkpoints
**Solution**:
Models should be in: `../../checkpoints/`

Check that these exist:
```
best_model_resnet50.pth
best_model_EfficientNet_B0.pth
best_model_mobilenet_v2.pth
best_model_vit.pth
best_model_swin.pth
best_model_deit.pth
best_model_cnn.pth
best_seg_segformer.pth
```

---

## 🚀 Advanced Usage

### Change Port
```bash
streamlit run app.py --server.port 8080
```

### Disable Browser Auto-Open
```bash
streamlit run app.py --server.headless true
```

### Enable Debug Logging
```bash
streamlit run app.py --logger.level=debug
```

### Configure Server
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
headless = false
enableXsrfProtection = true

[theme]
primaryColor = "#2ecc71"
```

---

## 📊 GPU Acceleration Setup

### Check GPU Status
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Install CUDA (if not present)
```bash
# Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# macOS
pip install torch torchvision

# Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 Application Pages

### 1️⃣ **Home Page**
- Overview of the application
- Quick stats
- Usage instructions
- Disclaimer

### 2️⃣ **Classification** (Pages → 1_Classification.py)
- Image upload
- Model selection (7 options)
- Real-time inference
- Confidence visualization
- Class probability breakdown

### 3️⃣ **Segmentation** (Pages → 2_Segmentation.py)
- Image upload
- Model selection (U-Net, SegFormer)
- Threshold adjustment
- Dice score & IoU metrics
- Multi-view visualization

### 4️⃣ **Model Comparison** (Pages → 3_Comparison.py)
- Performance metrics table
- Comparative visualizations
- Best model highlighting
- Development phase analysis
- Insights and recommendations

---

## 🧪 Testing

### Quick Test
```bash
# Run app
streamlit run app.py

# In browser:
# 1. Go to Classification page
# 2. Upload a test image
# 3. Select ResNet50
# 4. Click "Run Inference"
```

### Full Test Suite
```bash
# Verify imports
python -c "from utils.inference import load_classification_model; print('✓ Imports OK')"

# Test model loading
python -c "from utils.inference import load_classification_model; m = load_classification_model('ResNet50'); print('✓ Model loaded')"
```

---

## 📖 Documentation

- **README.md**: Full project documentation
- **requirements.txt**: Python dependencies
- **.streamlit/config.toml**: Streamlit configuration
- **utils/inference.py**: Model loading utilities
- **utils/preprocessing.py**: Image preprocessing
- **utils/visualization.py**: Plotting utilities

---

## 🔧 Development

### Project Structure
```
streamlit_app/
├── app.py                    # Main app (home page)
├── pages/
│   ├── 1_Classification.py   # Classification page
│   ├── 2_Segmentation.py     # Segmentation page
│   └── 3_Comparison.py       # Comparison dashboard
└── utils/
    ├── inference.py          # Model loading
    ├── preprocessing.py      # Image processing
    └── visualization.py      # Plotting
```

### Adding a New Page
1. Create file in `pages/`: `4_NewPage.py`
2. Streamlit auto-detects it in sidebar
3. Prefix with number for ordering

### Customizing Styling
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#2ecc71"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

---

## 🌐 Deployment

### Local Network
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Access from other machines: `http://your-ip:8501`

### Streamlit Cloud
```bash
# Push to GitHub
git push

# Deploy via: https://share.streamlit.io
# Select repo → app.py
```

### Docker
```bash
docker build -t cancer-detection .
docker run -p 8501:8501 cancer-detection
```

---

## ⚠️ Important Notes

- First inference run downloads model (~500MB each)
- App caches models in memory for faster subsequent runs
- GPU significantly speeds up inference (10x faster)
- App is optimized for 512×512px images

---

## ✅ Verification Checklist

Before first run, verify:
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip list | grep streamlit`)
- [ ] Model checkpoints exist in `../../checkpoints/`
- [ ] Port 8501 is available
- [ ] At least 8GB RAM available

---

## 📞 Support

For issues:
1. Check Troubleshooting section above
2. Review `.streamlit/logs/` for error messages
3. Verify Python version: `python --version`
4. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

---

**Happy analyzing! 🏥**
