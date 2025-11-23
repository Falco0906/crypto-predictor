#!/usr/bin/env bash
# Quick setup helper for macOS (M1/M2) to run the pretrained model (CPU/Metal)
set -euo pipefail

echo "----- macOS CPU setup helper -----"

if command -v conda >/dev/null 2>&1; then
  echo "Conda detected. Creating conda env 'crypto' with Python 3.10..."
  conda create -n crypto python=3.10 -y
  echo "Activate it with: conda activate crypto"
  echo "After activating, run this script section manually to finish installation."
  echo "Installing TensorFlow (macOS build) and project requirements..."
  echo "Run the commands after 'conda activate crypto':"
  cat <<'CMD'
python -m pip install --upgrade pip setuptools wheel
python -m pip install tensorflow-macos
python -m pip install -r requirements.txt --no-deps
python -m pip install "numpy<2" --force-reinstall
CMD
  exit 0
fi

echo "No conda detected — falling back to a venv at .venv (system python)."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "Installing tensorflow-macos (may fail if Python version incompatible)."
if python -m pip install tensorflow-macos; then
  echo "tensorflow-macos installed"
else
  echo "tensorflow-macos failed to install. If this happens, install Miniforge/conda and re-run this script." >&2
  exit 1
fi

python -m pip install -r requirements.txt --no-deps
python -m pip install "numpy<2" --force-reinstall

echo "Setup complete. Activate your venv (.venv) and run: python run_full_pipeline.py"
