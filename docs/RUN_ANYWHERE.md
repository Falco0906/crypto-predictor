# Run Anywhere - Universal Pipeline Guide

This project works on **ANY computer** without setup! Follow these simple steps.

---

## Quick Start (3 Steps)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Falco0906/crypto-predictor.git
cd crypto-predictor
```

### Step 2: Run the Pipeline
```bash
python run_full_pipeline.py
```

### Step 3: Choose Your Option
When prompted:
- **Press `n`** (default) → Uses **pre-trained model** (instant predictions) ⚡
- **Press `y`** → Trains a **new model** (requires GPU/CPU, 5-15 min) 🚀

Done! 🎉

---

## What Happens Automatically

### On First Run:
1. ✅ Checks Python packages
2. ✅ Installs missing packages (pandas, numpy, tensorflow, etc.)
3. ✅ Downloads latest crypto data from Yahoo Finance
4. ✅ Asks if you want to train or use pre-trained model
5. ✅ Generates predictions

### On Later Runs:
1. ✅ Updates crypto data
2. ✅ Asks about training
3. ✅ Generates fresh predictions

---

## Pre-trained Model Details

**What's Included:**
- Trained on Bitcoin, Ethereum, Solana, Litecoin, Cardano, Polkadot, Chainlink, Polygon, Avalanche
- 2 years of historical data
- 40 technical indicators
- Accuracy: 57% directional, 64% trend
- GPU-trained: RTX 3050 (works on any GPU/CPU)

**Location:** `data/models_gpu_improved/best_improved_model.h5`

---

## System Requirements

### Minimum (Pre-trained Model Only)
- **Python:** 3.8 or higher
- **RAM:** 2 GB
- **Storage:** 500 MB
- **Time:** 5 minutes

### For Training (Optional)
- **GPU:** Recommended (NVIDIA, any CUDA-capable GPU)
- **RAM:** 4 GB+
- **Time:** 5-15 minutes (GPU), 30+ minutes (CPU)

---

## Installation Details

### Automatic Installation
The script auto-installs:
```
pandas              - Data handling
numpy<2             - Numerical computing
tensorflow==2.10.0  - Deep learning
scikit-learn        - ML utilities
yfinance            - Crypto data
matplotlib          - Visualization
seaborn             - Enhanced plots
joblib              - Model serialization
```

### Manual Installation (if needed)
```bash
pip install -r requirements.txt
```

---

## File Structure

```
crypto-predictor/
├── run_full_pipeline.py           ← RUN THIS FILE
├── requirements.txt
├── RUN_ANYWHERE.md               ← YOU ARE HERE
│
├── src/
│   ├── crypto_training_script_improved.py    (Training code)
│   ├── crypto_predictor_improved.py           (Prediction code)
│   └── utils/update_data.py                   (Yahoo Finance integration)
│
├── data/
│   ├── models_gpu_improved/                   (Pre-trained model)
│   │   ├── best_improved_model.h5
│   │   ├── price_scaler.pkl
│   │   └── feature_scaler.pkl
│   │
│   └── raw_data/                              (Auto-downloaded data)
│       ├── coin_BTC.csv
│       ├── coin_ETH.csv
│       └── ... (more coins)
│
└── docs/
    └── (Training visualizations after training)
```

---

## Output Explained

### Predictions Format
```
BTC (Bitcoin):
  Current Price: $47,382.50
  1-Day Forecast: $47,889.23 (+1.07%)
  3-Day Trend: UPTREND (63.24% accuracy)

ETH (Ethereum):
  Current Price: $3,124.75
  1-Day Forecast: $3,158.42 (+1.07%)
  3-Day Trend: UPTREND
```

### Metrics
- **Directional Accuracy:** Did it predict the direction correctly? (57%)
- **Trend Accuracy:** 3-day trend prediction (64%)
- **MAE:** Average error in percentage (2.97%)

---

## Troubleshooting

### Issue: "tensorflow not found"
```bash
pip install tensorflow==2.10.0
```

### Issue: "numpy version error"
```bash
pip install numpy<2
```

### Issue: "GPU not detected" (if you have GPU)
Make sure:
- NVIDIA drivers installed
- CUDA toolkit 11.8 installed (for RTX 3050)
- TensorFlow can see GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Issue: Slow prediction
This is normal for first run. Subsequent runs are cached.

---

## Advanced: Training Your Own Model

### Step 1: Choose Training Option
```
[?] Do you want to train a NEW model? (y=train new, n=use pre-trained) [n]: y
```

### Step 2: What Happens
1. Downloads 2 years of data
2. Trains LSTM neural network
3. Saves model to `data/models_gpu_improved/`
4. Generates visualizations

### Step 3: Use Your Model
Next time you run the script, your trained model will be used automatically.

---

## Tips & Tricks

### Faster Predictions (Skip Data Update)
Edit `run_full_pipeline.py`, comment out Step 1:
```python
# updater.update_all_crypto_data(period='2y', interval='1d')
```

### Train on Different Data
Modify the coins in `src/utils/update_data.py`:
```python
COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']  # Change this
```

### Custom Prediction Horizon
Edit `src/crypto_predictor_improved.py`:
```python
prediction_days = 5  # Instead of 3
```

---

## FAQ

**Q: Does it require GPU?**  
A: No. Predictions use pre-trained model (works on CPU). Training is optional and faster on GPU.

**Q: Can I run this on Mac/Linux?**  
A: Yes! The script is cross-platform. Just run `python run_full_pipeline.py`

**Q: How accurate are predictions?**  
A: 57% directional accuracy (better than 50% random). For actual trading, use multiple models.

**Q: Can I modify the model?**  
A: Yes! Edit training parameters in `src/crypto_training_script_improved.py`

**Q: How often should I retrain?**  
A: Every 1-3 months with fresh data for best results.

---

## Support

### Check Logs
```bash
# See full training output
python run_full_pipeline.py  # Run and watch terminal

# Or save to file
python run_full_pipeline.py > output.log 2>&1
```

### Manual Steps
```bash
# Just update data
python -m src.utils.update_data

# Just predict
python -m src.crypto_predictor_improved

# Just train
python -m src.crypto_training_script_improved
```

---

## Summary

✅ **Works on any PC with Python**  
✅ **Auto-installs all dependencies**  
✅ **Auto-downloads latest data**  
✅ **Includes pre-trained model**  
✅ **Optional: Train your own model**  

**Ready to use!** Just run:
```bash
python run_full_pipeline.py
```

Enjoy predicting! 🚀
