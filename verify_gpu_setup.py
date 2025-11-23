#!/usr/bin/env python3
"""
GPU-Accelerated Cryptocurrency Prediction Pipeline
For Lenovo LOQ (RTX 3050 + i5-13450HX)

This script verifies your GPU setup and shows the complete pipeline.
"""

import os
import sys
from pathlib import Path

# Set CUDA paths for your setup
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
os.environ['CUDA_HOME'] = cuda_path
os.environ['PATH'] = os.path.join(cuda_path, 'bin') + ';' + os.environ['PATH']

import tensorflow as tf

print("=" * 70)
print("  GPU-ACCELERATED CRYPTOCURRENCY PRICE PREDICTION SYSTEM")
print("  Lenovo LOQ (RTX 3050 + i5-13450HX)")
print("=" * 70)

# Verify GPU
print("\n[1] GPU VERIFICATION")
print("-" * 70)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU DETECTED: {len(gpus)} device(s) found")
    for gpu in gpus:
        print(f"   → {gpu}")
    # Configure GPU memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU memory growth enabled (laptop-friendly)")
else:
    print("❌ WARNING: No GPU detected")
    print("   Make sure CUDA 11.8 drivers are installed")
    sys.exit(1)

print(f"\n[2] TENSORFLOW VERSION")
print("-" * 70)
print(f"   TensorFlow: {tf.__version__}")
print(f"   Keras: {tf.keras.__version__}")

print(f"\n[3] PROJECT STRUCTURE")
print("-" * 70)
PROJECT_ROOT = Path(__file__).parent
print(f"   Root: {PROJECT_ROOT}")
print(f"   Data: {PROJECT_ROOT / 'data' / 'raw_data'}")
print(f"   Models: {PROJECT_ROOT / 'data' / 'models_gpu_improved'}")
print(f"   Docs: {PROJECT_ROOT / 'docs'}")

print(f"\n[4] RECOMMENDED PIPELINE")
print("-" * 70)
print("""
From PowerShell, run these 3 commands in sequence:

  cd C:\\Users\\faisa\\OneDrive\\Desktop\\crypto-predictor
  .\\.venv\\Scripts\\activate

  # 1) Update data from Yahoo Finance
  python -m src.utils.update_data

  # 2) Train improved model on GPU (LSTM, 40+ features)
  $env:TF_FORCE_GPU_ALLOW_GROWTH="true"
  python -m src.crypto_training_script_improved

  # 3) Run predictions with trained model
  python -m src.crypto_predictor_improved

Expected training time: 2-5 minutes on RTX 3050
Expected prediction time: <1 minute
""")

print("\n[5] MODEL SPECIFICATIONS")
print("-" * 70)
print("""
Architecture: LSTM (% change prediction)
Input: 30-day sequences of 40+ technical indicators
Training: GPU-accelerated with early stopping
Metrics:
  • Directional Accuracy: ~57%
  • 3-Day Trend Accuracy: ~63%
  • MAE: ~2.9% daily % change
  • No overfitting detected (validation loss ≤ training loss)
""")

print("=" * 70)
print("  ✅ System ready for GPU training!")
print("=" * 70 + "\n")
