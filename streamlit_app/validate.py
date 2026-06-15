#!/usr/bin/env python3
"""
Validation script for Streamlit app setup.
Run this before launching the app to verify all dependencies.
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print(f"\n{'='*50}")
    print(f"  {text}")
    print(f"{'='*50}\n")

def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    status = "✓" if version.major >= 3 and version.minor >= 8 else "✗"
    print(f"{status} Python Version: {version.major}.{version.minor}.{version.micro}")
    return version.major >= 3 and version.minor >= 8

def check_imports():
    """Check all required packages can be imported"""
    packages = {
        "streamlit": "Streamlit",
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "transformers": "Transformers",
        "PIL": "Pillow",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "cv2": "OpenCV",
        "sklearn": "Scikit-learn",
    }

    print("Checking package imports...")
    all_ok = True

    for import_name, display_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {display_name}")
        except ImportError as e:
            print(f"✗ {display_name}: {str(e)}")
            all_ok = False

    return all_ok

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        available = torch.cuda.is_available()
        status = "✓" if available else "⚠"
        device = torch.cuda.get_device_name(0) if available else "CPU"
        print(f"{status} CUDA/GPU: {device}")
        return True
    except Exception as e:
        print(f"✗ CUDA Check Failed: {str(e)}")
        return False

def check_checkpoints():
    """Check if model checkpoints exist"""
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"

    required_models = [
        "best_model_resnet50.pth",
        "best_model_EfficientNet_B0.pth",
        "best_model_mobilenet_v2.pth",
        "best_model_vit.pth",
        "best_model_swin.pth",
        "best_model_deit.pth",
        "best_model_cnn.pth",
        "best_seg_segformer.pth",
    ]

    optional_models = [
        "best_model_unet.pth",
    ]

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Directory exists: {checkpoint_dir.exists()}")

    if not checkpoint_dir.exists():
        print("⚠ Checkpoint directory not found!")
        return False

    all_ok = True
    print("\nRequired models:")
    for model in required_models:
        path = checkpoint_dir / model
        status = "✓" if path.exists() else "✗"
        size = f"({path.stat().st_size / 1e6:.0f}MB)" if path.exists() else ""
        print(f"{status} {model} {size}")
        if not path.exists():
            all_ok = False

    print("\nOptional models:")
    for model in optional_models:
        path = checkpoint_dir / model
        status = "✓" if path.exists() else "⚠"
        size = f"({path.stat().st_size / 1e6:.0f}MB)" if path.exists() else ""
        print(f"{status} {model} {size}")

    return all_ok

def check_file_structure():
    """Check Streamlit app file structure"""
    app_dir = Path(__file__).parent

    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "GETTING_STARTED.md",
    ]

    required_dirs = [
        "pages",
        "utils",
    ]

    print("Checking file structure...")
    all_ok = True

    for file in required_files:
        path = app_dir / file
        status = "✓" if path.exists() else "✗"
        print(f"{status} {file}")
        if not path.exists():
            all_ok = False

    print("\nChecking directories...")
    for dir_name in required_dirs:
        path = app_dir / dir_name
        status = "✓" if path.exists() else "✗"
        print(f"{status} {dir_name}/")
        if not path.exists():
            all_ok = False

    return all_ok

def check_disk_space():
    """Check available disk space"""
    import shutil
    try:
        stat = shutil.disk_usage("/")
        available_gb = stat.free / (1024**3)
        status = "✓" if available_gb > 2 else "⚠"
        print(f"{status} Available disk space: {available_gb:.1f}GB")
        return available_gb > 2
    except Exception as e:
        print(f"⚠ Could not check disk space: {str(e)}")
        return True

def check_memory():
    """Check available RAM"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        status = "✓" if available_gb > 4 else "⚠"
        print(f"{status} Available RAM: {available_gb:.1f}GB")
        return available_gb > 4
    except ImportError:
        print("⚠ psutil not installed (optional)")
        return True
    except Exception as e:
        print(f"⚠ Could not check RAM: {str(e)}")
        return True

def main():
    print_header("Streamlit App Validation")

    print("Python Configuration")
    py_ok = check_python_version()

    print_header("Dependencies")
    imports_ok = check_imports()

    print_header("Hardware")
    cuda_ok = check_cuda()
    memory_ok = check_memory()
    disk_ok = check_disk_space()

    print_header("Project Files")
    files_ok = check_file_structure()

    print_header("Model Checkpoints")
    checkpoints_ok = check_checkpoints()

    print_header("Summary")

    results = {
        "Python Version": py_ok,
        "Package Imports": imports_ok,
        "Checkpoints": checkpoints_ok,
        "File Structure": files_ok,
        "GPU Support": cuda_ok,
        "Memory": memory_ok,
        "Disk Space": disk_ok,
    }

    for check, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}")

    all_ok = all(results.values())

    if all_ok:
        print("\n✓ All checks passed! Ready to run:")
        print("  → streamlit run app.py")
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        print("  Refer to GETTING_STARTED.md for troubleshooting.")

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
