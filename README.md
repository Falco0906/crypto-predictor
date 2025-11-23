# Cryptocurrency Price Prediction System

A compact, consolidated guide for running and understanding the crypto-predictor project. Full, detailed documentation has been moved to the `docs/` folder — this README contains the essential quick start, architecture summary, and how to use the multi-model system.

## Quick Summary

- Purpose: GPU-accelerated LSTM-based cryptocurrency percentage-change predictor.
- Entrypoint: `run_full_pipeline.py` — updates data, lets you choose a model (pre-trained or user-trained), trains if requested, and produces predictions for supported coins.
- Pre-trained model: Included in `data/models_gpu_improved/` for instant use.
- Models: User-trained models are saved with timestamps and tracked in `data/models_gpu_improved/model_registry.json`.

## Quick Start (3 steps)

1) Create and activate a virtual environment

   # Windows PowerShell
   python -m venv .venv; .\.venv\Scripts\Activate

2) Install dependencies

   pip install -r requirements.txt
   pip install "numpy<2" --force-reinstall  # required for TensorFlow 2.10

3) Run the pipeline (interactive)

   # This script will update data, show the model selection menu, and run/train as requested
   python run_full_pipeline.py

Tip: If you prefer non-interactive runs, inspect `run_full_pipeline.py` to see command-line or environment options.

## Supported Coins

BTC, ETH, SOL, LTC, ADA, DOT, LINK, MATIC, AVAX (data files live in `data/raw_data/`).

## Architecture & Multi-Model System (summary)

This project uses an LSTM-based model trained on 30-day sequences of ~40 engineered features (RSI, MACD, moving averages, volatility, momentum, volume indicators, etc.). The model predicts percentage changes (not absolute prices), which are clipped for stability and used to produce multi-day sequential forecasts.

Key pipeline steps:

- Data collection: `src/utils/update_data.py` collects 2 years of historical data from Yahoo Finance and saves CSVs into `data/raw_data/`.
- Model manager: `src/utils/model_manager.py` scans `data/models_gpu_improved/`, registers a pre-trained model, and lists any user-trained models recorded in `model_registry.json`.
- Interactive selection: `run_full_pipeline.py` shows an interactive menu with all models and a "Train NEW" option.
- Training flow: `src/crypto_training_script_improved.py` saves the trained model to `best_improved_model.h5`, then the ModelManager renames and registers it as `user_model_<timestamp>.h5` along with metrics (directional accuracy, trend accuracy, MAE, Sharpe ratio).
- Prediction flow: `src/crypto_predictor_improved.py` accepts model selection info and runs predictions across the supported coins.

For a full diagram and detailed internals see `docs/ARCHITECTURE.md` and `docs/MULTI_MODEL_SYSTEM.md`.

### Pipeline Visual

Below is a compact visual of the pipeline (click to open full-size):

![Pipeline Diagram](docs/images/pipeline_diagram.svg)

Figure: Data Update → Model Manager → Selection Menu → Train / Use → Predict & Output.

### Architecture Diagram (inline)

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER RUNS: run_full_pipeline.py              │
└────────────────────────────┬────────────────────────────────────┘
                   │
      ┌────────────────────▼─────────────────────┐
      │  STEP 1: Data Collection                 │
      │  ├─ Check dependencies                   │
      │  ├─ Download crypto data (Yahoo Finance) │
      │  ├─ 9 cryptocurrencies                   │
      │  ├─ 6K+ records, 2 years                 │
      │  └─ Save to data/raw_data/               │
      └────────────────────┬─────────────────────┘
                   │
      ┌────────────────────▼──────────────────────────┐
      │  STEP 2: Initialize Model Manager             │
      │  ├─ Create ModelManager instance              │
      │  ├─ Auto-register pre-trained model           │
      │  ├─ Load model_registry.json                  │
      │  └─ Scan for user-trained models              │
      └────────────────────┬──────────────────────────┘
                   │
      ┌────────────────────▼──────────────────────────┐
      │  STEP 3: Display Model Selection Menu          │
      │  ├─ List all available models                 │
      │  ├─ Show accuracy metrics                     │
      │  ├─ Show creation dates                       │
      │  └─ Wait for user input (1, 2, 3, etc.)       │
      └────────────────────┬──────────────────────────┘
                   │
          ┌──────────────┼──────────────┐
          │              │              │
       ┌────▼─────┐   ┌────▼────┐   ┌────▼──────┐
       │ User: 1  │   │ User: 2 │   │ User: 3   │
       │(Pre-tr)  │   │(User MD)│   │(Train NEW)│
       └────┬─────┘   └────┬────┘   └────┬──────┘
          │              │              │
       ┌────▼─────────┐    │    ┌─────────▼──────┐
       │ Selection: 1 │    │    │ Selection: 3   │
       └────┬─────────┘    │    └─────────┬──────┘
          │              │              │
          │    ┌─────────▼────────┐     │
          │    │ Selection: 2     │     │
          │    └────────┬─────────┘     │
          │             │               │
          │   ┌─────────┴────────┐      │
          │   │  get_model_files │      │
          │   │  Return:         │      │
          │   │  - model path    │      │
          │   │  - scaler paths  │      │
          │   └────────┬────────┘      │
          │            │               │
       ┌────▼────────────▼───┐      ┌────▼────────────┐
       │  Load Selected Model │      │  Train New Model│
       │  (Instant - 2 sec)   │      │  (40-50 sec GPU)│
       └────┬────────────────┘      └────┬──────┬─────┘
          │                            │      │
          │                       ┌────▼─┐  ┌▼────────┐
          │                       │Train │  │Register │
          │                       │Model │  │+ Version│
          │                       └────┬─┘  └┬────────┘
          │                            │     │
          │                       ┌────▼────▼────┐
          │                       │Save with:    │
          │                       │-Timestamp    │
          │                       │-Metrics      │
          │                       │-Registry     │
          │                       └────┬─────────┘
          │                            │
          ┌────────────────────────────┴──────┐
          │                                   │
      ┌─────▼────────────┐             ┌───────▼──────────┐
      │ STEP 4: Predict  │             │ STEP 4: Predict  │
      │ Using Model 1/2  │             │ Using New Model  │
      │ (Instant)        │             │ (Immediate)      │
      └─────┬────────────┘             └───────┬──────────┘
          │                                   │
          └────────────────┬──────────────────┘
                   │
      ┌──────────────────────▼─────────────────┐
      │  Generate Predictions                  │
      │  ├─ Process all 9 cryptos              │
      │  ├─ Calculate features                 │
      │  ├─ Run LSTM predictions               │
      │  ├─ Show current price                 │
      │  ├─ Show predicted change              │
      │  ├─ Show 3-day forecast                │
      │  └─ Display results                    │
      └──────────────────────┬──────────────────┘
                   │
      ┌──────────────────────▼──────────────────┐
      │  Pipeline Complete ✅                   │
      │  - Data: Updated                        │
      │  - Model: Selected/Trained              │
      │  - Predictions: Generated               │
      └───────────────────────────────────────┘
```

## Where to find the detailed docs

All other markdown files have been moved to `docs/`. Open the folder for detailed guides, examples, and full architectural diagrams. Notable files:

- `docs/USAGE_GUIDE.md` - Yahoo Finance updater and automated update instructions
- `docs/ARCHITECTURE.md` - Full system architecture diagram and model manager flow
- `docs/MULTI_MODEL_SYSTEM.md` - Complete multi-model UX and registry details
- `docs/QUICK_START.md` - Short quick-start snippets and environment notes

## Model performance (summary)

- Directional accuracy: ~57% (better than random but not perfect)
- 3-day trend accuracy: ~63%
- MAE (mean absolute error): ~2.9%
- RMSE: ~4.1%

These values are approximate. See detailed `docs/` files for full training logs and metrics.

## Troubleshooting (common)

- TensorFlow/Numpy compatibility: Use `tensorflow==2.10.0` with `numpy<2.0.0` for best GPU compatibility. Example:

  pip install "numpy<2" --force-reinstall
  pip install --force-reinstall "tensorflow==2.10.0"

- GPU not detected: verify CUDA and cuDNN installation and that drivers are up-to-date. The model will fall back to CPU if no GPU is available.

- Data not found: run the updater first:

  python -m src.utils.update_data

## How to add / train models

1. Run `python run_full_pipeline.py` and choose "Train NEW" from the menu.
2. Training saves `best_improved_model.h5` then the ModelManager renames and registers it as `user_model_<timestamp>.h5` in `data/models_gpu_improved/`.
3. Models are tracked in `data/models_gpu_improved/model_registry.json` and will appear in the menu on subsequent runs.

## Next steps and contribution notes

- The codebase is ready to share. If you want me to commit these changes and push them to your remote repository, tell me and confirm that I should run the git commands here (I will need push access / configured remote). If you prefer to push yourself, run:

  git add docs README.md
  git commit -m "Move docs into docs/ and add consolidated root README"
  git push

## License & Disclaimer

This project is provided under the MIT License (see LICENSE). It is for educational/research purposes only and not financial advice.

---

For full documentation, diagrams, training results and developer notes: open the `docs/` folder.

Built with TensorFlow, Pandas, and Yahoo Finance
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
