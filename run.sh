#!/bin/bash

# ============================================================
# Cryptocurrency Predictor - Linux/Mac Quick Start
# ============================================================
# This script runs the full pipeline
# Usage: ./run.sh  or  bash run.sh
# ============================================================

echo ""
echo "============================================================"
echo "  Cryptocurrency Price Predictor - Linux/Mac Launcher"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo ""
    echo "Please install Python 3 using:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "  macOS: brew install python3"
    echo ""
    exit 1
fi

echo "[OK] Python detected: $(python3 --version)"
echo ""
echo "[INFO] Starting Cryptocurrency Prediction Pipeline..."
echo ""

# Run the pipeline
python3 run_full_pipeline.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Pipeline failed. See error messages above."
    echo ""
    exit 1
fi

echo ""
echo "[OK] Pipeline completed successfully!"
