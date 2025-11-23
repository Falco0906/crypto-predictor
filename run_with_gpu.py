#!/usr/bin/env python3
"""
Run the pipeline with GPU enabled
Sets proper CUDA environment variables for TensorFlow
"""

import os
import sys
from pathlib import Path

# Set CUDA paths for Windows
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
cudnn_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"  # Usually same as CUDA

# Add CUDA to PATH
if os.path.exists(cuda_path):
    cuda_bin = os.path.join(cuda_path, 'bin')
    if cuda_bin not in os.environ['PATH']:
        os.environ['PATH'] = cuda_bin + ';' + os.environ['PATH']
    
    # Set CUDA_HOME
    os.environ['CUDA_HOME'] = cuda_path
    os.environ['CUDA_PATH'] = cuda_path
    
    # Set cuDNN path if it exists
    cudnn_lib = os.path.join(cudnn_path, 'lib', 'x64')
    if os.path.exists(cudnn_lib):
        os.environ['CUDNN_PATH'] = cudnn_lib
    
    print(f"[OK] CUDA environment configured:")
    print(f"     CUDA_HOME: {cuda_path}")
    print(f"     CUDA in PATH: {cuda_bin}")
else:
    print(f"[WARNING] CUDA not found at {cuda_path}")

# Now import TensorFlow to verify GPU detection
import tensorflow as tf

print(f"\n[INFO] TensorFlow {tf.__version__} loaded")
print(f"[INFO] GPU devices found: {len(tf.config.list_physical_devices('GPU'))}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[OK] GPU is available:")
    for gpu in gpus:
        print(f"     - {gpu}")
    # Configure GPU memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[OK] GPU memory growth enabled")
else:
    print(f"[WARNING] No GPUs detected. Training will use CPU.")

# Run the full pipeline
print("\n" + "="*60)
print("Starting crypto prediction pipeline...")
print("="*60 + "\n")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the pipeline main
import subprocess
result = subprocess.run(
    [sys.executable, 'run_full_pipeline.py'],
    cwd=str(PROJECT_ROOT)
)
sys.exit(result.returncode)
