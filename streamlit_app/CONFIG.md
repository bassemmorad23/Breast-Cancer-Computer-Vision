# Project Configuration & Setup Guide

## Directory Structure

```
Final project/
├── streamlit_app/              ← Main Streamlit application
│   ├── app.py                  ← Home page & navigation
│   ├── requirements.txt        ← Python dependencies
│   ├── README.md               ← Full documentation
│   ├── GETTING_STARTED.md      ← Quick start guide
│   ├── run.bat                 ← Windows launcher
│   ├── run.sh                  ← Unix/Linux launcher
│   ├── .streamlit/
│   │   └── config.toml         ← Streamlit configuration
│   ├── pages/                  ← Multi-page app pages
│   │   ├── 1_Classification.py ← Classification interface
│   │   ├── 2_Segmentation.py   ← Segmentation interface
│   │   └── 3_Comparison.py     ← Model comparison dashboard
│   └── utils/                  ← Utility modules
│       ├── inference.py        ← Model loading & inference
│       ├── preprocessing.py    ← Image preprocessing
│       └── visualization.py    ← Visualization & plotting
│
├── checkpoints/                ← Model weights
│   ├── best_model_resnet50.pth
│   ├── best_model_EfficientNet_B0.pth
│   ├── best_model_mobilenet_v2.pth
│   ├── best_model_vit.pth
│   ├── best_model_swin.pth
│   ├── best_model_deit.pth
│   ├── best_model_cnn.pth
│   └── best_seg_segformer.pth
│
└── [other training files]
```

## Quick Reference

### Start Application
- **Windows**: Double-click `streamlit_app/run.bat`
- **macOS/Linux**: Run `streamlit_app/run.sh`
- **Manual**: `cd streamlit_app && streamlit run app.py`

### URLs
- **Local**: http://localhost:8501
- **Network**: http://your-ip:8501 (if using `--server.address 0.0.0.0`)

### Pages
1. **Home** - Overview & instructions
2. **Classification** - Detect benign/malignant
3. **Segmentation** - Find lesion boundaries
4. **Model Comparison** - Performance metrics

### Models Available

#### Classification (7 models)
- ResNet-50 (94.57% accuracy) ⭐ **Best**
- DeiT (93.02%)
- Swin (93.80%)
- ViT (92.25%)
- EfficientNet-B0 (89.15%)
- MobileNet-V2 (87.60%)
- Custom CNN (72.31%)

#### Segmentation (2 models)
- SegFormer (Transformer-based)
- U-Net (CNN-based)

## Configuration Files

### .streamlit/config.toml
Controls Streamlit behavior and styling:
- Theme colors
- Page layout
- Server settings
- Session state

### requirements.txt
Lists all Python dependencies with pinned versions for reproducibility.

## Performance Metrics

### Inference Speed (GPU: RTX 3090)
- Classification: ~50-100ms
- Segmentation: ~200-500ms

### Memory Usage
- ResNet-50: ~500MB VRAM
- SegFormer: ~800MB VRAM
- Total system: ~2GB minimum

### Accuracy
- Best: ResNet-50 at 94.57%
- Worst: CNN at 72.31%
- Average: ~90.7%

## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Models not found | Ensure checkpoints exist in `../checkpoints/` |
| Slow inference | Enable CUDA: `python -c "import torch; torch.cuda.is_available()"` |
| Port in use | Use alternate port: `streamlit run app.py --server.port 8502` |
| Import errors | Reinstall deps: `pip install -r requirements.txt` |
| Out of memory | Use smaller model or reduce image size |

## Advanced Configuration

### Enable GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Change Streamlit Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#2ecc71"
backgroundColor = "#ffffff"
font = "sans serif"
```

### Run on Network
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Production Deployment
```bash
streamlit run app.py \
  --server.port 8080 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --logger.level warning
```

## Development Notes

### Adding New Model
1. Add model builder function to `utils/inference.py`
2. Add checkpoint path
3. Update model selection dropdown in classification page
4. Test inference

### Adding New Page
1. Create `pages/4_PageName.py`
2. Streamlit auto-detects in sidebar
3. Number prefix controls order

### Customizing Visualizations
Edit `utils/visualization.py` to modify:
- Plot styles
- Color schemes
- Figure sizes
- Metric calculations

## File Sizes

```
Dependencies:
- PyTorch: ~750MB
- Transformers: ~500MB
- Other packages: ~250MB
Total: ~1.5GB

Models:
- ResNet-50: 214MB
- ViT: 570MB
- Swin: 223MB
- DeiT: 570MB
- EfficientNet-B0: 41MB
- MobileNet-V2: 18MB
- CNN: 1.5MB
- SegFormer: 290MB
Total: ~1.9GB

Total Installation: ~3.4GB
```

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ Full | Recommended |
| Firefox | ✅ Full | Good |
| Safari | ✅ Full | Good |
| Edge | ✅ Full | Good |
| IE 11 | ❌ Not supported | Use modern browser |

## System Requirements

| Specification | Minimum | Recommended |
|---------------|---------|-------------|
| CPU | Any modern CPU | 4+ cores |
| RAM | 8GB | 16GB+ |
| Disk | 5GB | 10GB+ |
| GPU | Optional | NVIDIA CUDA 11.8+ |

## Support & Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Docs**: https://pytorch.org/docs
- **Transformers**: https://huggingface.co/docs/transformers
- **MONAI**: https://monai.io (Medical imaging)

## Version History

### v1.0.0 (2026-06-15)
- Initial release
- 7 classification models
- 2 segmentation models
- Full comparison dashboard
- Multi-page Streamlit app
- Production-ready code

## License & Disclaimer

This project is provided for educational and research purposes only.
Not intended for clinical use. Always consult qualified healthcare professionals.
