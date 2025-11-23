# 🚀 Quick Start Guide - GPU Training & Prediction

## System Requirements
- **Hardware**: RTX 3050 GPU, i5-13450HX CPU
- **Software**: CUDA 11.8, Python 3.10+, TensorFlow 2.10.0

## Quick Pipeline (3 Commands)

```powershell
# Activate environment
cd C:\Users\faisa\OneDrive\Desktop\crypto-predictor
.\venv\Scripts\activate

# 1) Update data from Yahoo Finance (2024-2025 data, 10 coins)
python -m src.utils.update_data

# 2) Train LSTM on GPU (2-5 minutes on RTX 3050)
$env:TF_FORCE_GPU_ALLOW_GROWTH="true"
python -m src.crypto_training_script_improved

# 3) Generate predictions
python -m src.crypto_predictor_improved
```

## What Each Step Does

### Step 1: Data Update
- Fetches latest prices from Yahoo Finance
- Downloads 2 years of daily OHLCV data
- Creates: `data/raw_data/coin_*.csv`

### Step 2: GPU Training
- Loads all cryptocurrency data
- Builds 40+ technical indicators
- Creates 30-day LSTM sequences  
- **Trains on GPU** (RTX 3050, ~5 min)
- Saves model to: `data/models_gpu_improved/`
- Generates visualizations (loss curves, metrics)

### Step 3: Predictions
- Loads trained model from GPU folder
- Makes 1-day percentage change predictions
- Forecasts next 3 days
- **Output**: Clean per-coin summary

## Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cryptocurrency Predictions (GPU-Trained Model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bitcoin (BTC)
   Current Price: $115,075.73
   Predicted Change: +0.10% 📈
   Predicted Price: $115,195.48
   Next 3 Days:
     Day 1: $115,195.48 (+0.10%)
     Day 2: $115,282.38 (+0.08%)
     Day 3: $115,355.01 (+0.06%)

Ethereum (ETH)
   Current Price: $4,715.91
   Predicted Change: -0.07% 📉
   Predicted Price: $4,712.37
   Next 3 Days:
     Day 1: $4,712.37 (-0.07%)
     Day 2: $4,712.21 (-0.00%)
     Day 3: $4,716.00 (+0.08%)

[... more coins ...]
```

## Model Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Directional Accuracy | **57.41%** | Better than 50% baseline |
| 3-Day Trend Accuracy | **63.24%** | Good trend detection |
| MAE | **2.90%** | Avg prediction error |
| RMSE | **4.10%** | Volatility measure |
| Training Loss | 0.3565 | ✅ Decreasing |
| Validation Loss | 0.3096 | ✅ Lower than training |
| Overfitting | **None** | ✅ Model generalizes well |

## Project Structure

```
crypto-predictor/
├── src/
│   ├── crypto_training_script_improved.py  # GPU training
│   ├── crypto_predictor_improved.py        # Predictions
│   ├── utils/
│   │   └── update_data.py                  # Data collection
│   └── legacy/                             # Old scripts (archived)
├── data/
│   ├── raw_data/                           # Yahoo Finance CSVs
│   ├── models_gpu_improved/                # Trained model + scalers
│   └── models/                             # Other model versions
├── docs/                                   # Training visualizations
├── training_visualizations/                # Loss curves, metrics
└── trained_model_improved/                 # Latest training output
```

## GPU Acceleration Details

✅ **GPU Memory Configuration**
- Enabled memory growth (prevents OOM on laptops)
- Only allocates VRAM as needed
- Supports RTX 3050 6GB efficiently

✅ **Training Speed**
- CPU-only: ~20-30 minutes
- GPU (RTX 3050): ~5 minutes
- **6x faster** on GPU!

## Troubleshooting

### GPU Not Detected
```powershell
# Check GPU status
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Set CUDA path if needed
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
```

### Out of Memory (OOM)
- Already handled by memory growth setting
- If still occurs: use smaller batch size in training script

### Prediction Shows $0.00 Prices
- Those coins have invalid data
- Already filtered out in current version
- Check `data/raw_data/` for .csv files

## Next Steps

1. Run the 3-command pipeline above
2. Check generated visualizations in `training_visualizations/`
3. Review predictions in terminal output
4. Use `data/models_gpu_improved/` model for deployment

---

**Last Updated**: Nov 23, 2025  
**GPU Status**: ✅ Working (RTX 3050, CUDA 11.8)  
**Model Status**: ✅ Trained & Validated
