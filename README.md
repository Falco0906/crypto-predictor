# 🚀 Cryptocurrency Price Prediction System

A comprehensive machine learning system for predicting cryptocurrency prices using advanced deep learning techniques. **GPU-accelerated training with improved accuracy and realistic predictions.**

## 📊 Project Status

✅ **GPU-Accelerated Training** - Optimized for RTX 3050 and similar GPUs  
✅ **Improved Model Architecture** - LSTM-based with percentage change prediction  
✅ **Enhanced Feature Engineering** - 40+ percentage-based technical indicators  
✅ **Comprehensive Training Visualizations** - Auto-generated charts and analysis  
✅ **Multi-day Predictions** - Sequential forecasting with dynamic feature updates  
✅ **Clean Repository Structure** - Organized for production and presentation  

## 🏗️ Project Structure

```
crypto-predictor/
├── src/                          # Source code
│   ├── crypto_training_script_improved.py    # Main training script (GPU-optimized)
│   ├── crypto_predictor_improved.py         # Main prediction script
│   ├── utils/                    # Utility scripts
│   │   ├── update_data.py       # Data update script
│   │   ├── auto_update.py       # Automated daily updates
│   │   └── yahoo_finance_updater.py  # Yahoo Finance integration
│   └── legacy/                   # Legacy scripts (for reference)
│       ├── crypto_training_script.py
│       └── crypto_predictor.py
├── data/                         # Data storage
│   ├── models_gpu_improved/      # GPU-trained models (active)
│   │   ├── crypto_improved_model.h5
│   │   ├── price_scaler.pkl
│   │   ├── feature_scaler.pkl
│   │   └── model_metadata.json
│   ├── models/                   # Other model versions
│   └── raw_data/                 # Raw cryptocurrency CSV files
├── docs/                         # Documentation and visualizations
│   ├── training_results.png
│   ├── loss_convergence_analysis.png
│   ├── metrics_correlation.png
│   ├── training_progress_summary.png
│   └── training_history.csv
├── misc/                         # Non-essential files
│   ├── crypto_web_predictor.tsx
│   └── best_*.h5
├── README.md
├── USAGE_GUIDE.md
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (3.9+ recommended)
- CUDA-enabled GPU (optional but recommended for faster training)
- 8GB+ RAM (16GB+ recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Falco0906/crypto-predictor.git
   cd crypto-predictor
   ```

2. **Set up virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # Ensure NumPy < 2.0 for TensorFlow 2.10 compatibility
   pip install "numpy<2" --force-reinstall
   ```

4. **Verify GPU setup (if using GPU)**
   ```bash
   python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
   ```

## 📋 Usage Pipeline

### Complete Workflow (3 Simple Steps)

```bash
# 1) Update data from Yahoo Finance
python -m src.utils.update_data

# 2) Train improved model on GPU (saves directly to data/models_gpu_improved/)
$env:TF_FORCE_GPU_ALLOW_GROWTH="true"  # Windows PowerShell
# export TF_FORCE_GPU_ALLOW_GROWTH="true"  # macOS/Linux
python -m src.crypto_training_script_improved

# 3) Run predictions with updated data + GPU-trained model
python -m src.crypto_predictor_improved
```

### Step-by-Step Details

#### Step 1: Update Data
Fetches 2 years of historical data for 9 cryptocurrencies (BTC, ETH, SOL, LTC, ADA, DOT, LINK, MATIC, AVAX) from Yahoo Finance and saves to `data/raw_data/`.

#### Step 2: Train Model
- Loads data from `data/raw_data/`
- Creates 40+ technical indicators
- Trains LSTM model on GPU (if available)
- Saves model directly to `data/models_gpu_improved/`
- Generates training visualizations in `docs/`

#### Step 3: Make Predictions
- Loads model from `data/models_gpu_improved/`
- Processes all valid cryptocurrency files
- Outputs clean predictions with current price, predicted change, and 3-day forecast

## 📈 Model Performance

### Latest Training Results

- **Directional Accuracy**: ~57.4% (predicts up/down direction correctly)
- **3-Day Trend Accuracy**: ~63.2% (predicts multi-day trends)
- **Mean Absolute Error (MAE)**: ~2.9% (average error in daily % change)
- **Root Mean Squared Error (RMSE)**: ~4.1%
- **Overfitting Status**: No overfitting detected (validation loss < training loss)

### Model Architecture

- **Type**: LSTM (Long Short-Term Memory) neural network
- **Input**: 30-day sequences of 40 technical indicators
- **Output**: Percentage change prediction (not absolute price)
- **Features**: RSI, MACD, Bollinger Bands, Moving Averages, Volatility, Momentum, Support/Resistance, Volume indicators
- **Regularization**: BatchNormalization + Dropout to prevent overfitting

### Training Characteristics

- **Loss Convergence**: Training and validation loss decrease steadily
- **No Overfitting**: Validation loss remains below training loss
- **Stable Predictions**: Realistic ±1-2% daily changes (no extreme -99% outputs)
- **GPU Acceleration**: Automatic GPU detection and memory growth configuration

## 🔧 Key Features

### 1. **GPU-Optimized Training**
- Automatic GPU detection (RTX 3050, RTX 3060, etc.)
- Memory growth configuration for laptop GPUs
- Direct model saving to `data/models_gpu_improved/`

### 2. **Realistic Predictions**
- Predicts percentage changes instead of absolute prices
- Clips predictions to ±20% for stability
- Multi-day sequential forecasting with feature updates

### 3. **Comprehensive Visualizations**
- Training vs validation loss curves
- MAE/MSE convergence analysis
- Metrics correlation matrix
- Training progress summary
- Export to high-resolution PNG + CSV

### 4. **Clean Data Pipeline**
- Automatic data fetching from Yahoo Finance
- Robust outlier removal and data cleaning
- Skips invalid/zero-price coins automatically

### 5. **Production-Ready Structure**
- Organized directory structure
- Legacy scripts separated
- Clear separation of concerns

## 📊 Supported Cryptocurrencies

| Coin | Symbol | Status |
|------|--------|---------|
| Bitcoin | BTC-USD | ✅ Active |
| Ethereum | ETH-USD | ✅ Active |
| Solana | SOL-USD | ✅ Active |
| Litecoin | LTC-USD | ✅ Active |
| Cardano | ADA-USD | ✅ Active |
| Polkadot | DOT-USD | ✅ Active |
| Chainlink | LINK-USD | ✅ Active |
| Polygon | MATIC-USD | ✅ Active |
| Avalanche | AVAX-USD | ✅ Active |

## 🛠️ Technical Requirements

### Python Packages
```
tensorflow==2.10.0
pandas>=1.3.0
numpy<2.0.0  # Required for TF 2.10 compatibility
scikit-learn>=1.0.0
joblib>=1.1.0
yfinance>=0.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### System Requirements
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 2GB+ free space
- **GPU**: Optional but recommended (RTX 3050+ for faster training)
- **Python**: 3.8+ (3.9+ recommended)

## 📝 File Locations

### Models
- **Active Model**: `data/models_gpu_improved/` (used by predictor)
- **Training Output**: Automatically saved to `data/models_gpu_improved/` during training

### Data
- **Raw Data**: `data/raw_data/coin_*.csv`
- **Combined Data**: `data/raw_data/combined_crypto_data.csv`

### Visualizations
- **Training Charts**: `docs/training_*.png`
- **Training Data**: `docs/training_history.csv`

## 🚨 Important Notes

### Model Limitations
- **Not Financial Advice**: This is for educational purposes only
- **Market Conditions**: Models trained on historical data may not work in changing markets
- **Volatility**: Crypto markets are extremely volatile and unpredictable
- **Accuracy**: ~57% directional accuracy means predictions are better than random but not perfect

### Best Practices
- **Use Improved Scripts**: Always use `crypto_training_script_improved.py` and `crypto_predictor_improved.py`
- **Monitor Training**: Check `docs/` folder after each training run
- **Regular Updates**: Retrain with fresh data weekly/monthly
- **Risk Management**: Don't invest more than you can afford to lose

## 🔍 Troubleshooting

### Common Issues

1. **NumPy/TensorFlow Compatibility**
   ```bash
   pip install "numpy<2" --force-reinstall
   pip install --force-reinstall "tensorflow==2.10.0"
   ```

2. **GPU Not Detected**
   - Verify CUDA/cuDNN installation
   - Check GPU drivers are up to date
   - Model will fall back to CPU if GPU unavailable

3. **Import Errors**
   ```bash
   # Ensure you're in the project root
   python -m src.crypto_training_script_improved
   ```

4. **Data Not Found**
   - Run `python -m src.utils.update_data` first
   - Check `data/raw_data/` contains CSV files

## 📞 Support

### Getting Help
1. Check error messages carefully
2. Verify all dependencies are installed
3. Ensure data files are in `data/raw_data/`
4. Check file paths and permissions

## 🎯 Next Steps

### Future Enhancements
- [ ] Sentiment analysis integration
- [ ] Cross-asset correlation features
- [ ] Regime detection for different market conditions
- [ ] Uncertainty quantification (prediction confidence intervals)
- [ ] Ensemble methods (combine multiple models)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This system is for educational and research purposes only.** Cryptocurrency trading involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Always do your own research and consult with financial advisors before making investment decisions.

---

**Built with ❤️ using TensorFlow, Pandas, and Yahoo Finance**
