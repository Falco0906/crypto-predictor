# 🚀 Cryptocurrency Price Prediction System

A comprehensive machine learning system for predicting cryptocurrency prices using advanced deep learning techniques. **Now with improved accuracy, better predictions, and comprehensive training visualizations!**

## 📊 Current Status

✅ **FIXED Poor Accuracy Issues** - Resolved -99% extreme predictions  
✅ **Improved Model Architecture** - LSTM-based with percentage change prediction  
✅ **Enhanced Feature Engineering** - 40+ percentage-based technical indicators  
✅ **Comprehensive Training Visualizations** - Auto-generated charts and analysis  
✅ **Multi-day Predictions** - Sequential forecasting with dynamic feature updates  
✅ **Better Data Handling** - Robust outlier removal and data cleaning  

## 🏗️ System Architecture

### 1. **Data Management**
- `yahoo_finance_updater.py` - Fetches live data from Yahoo Finance
- `update_data.py` - Simple script to update all cryptocurrency data

### 2. **Training Scripts**
- `crypto_training_script.py` - Original training script (for reference)
- `crypto_training_script_improved.py` - **IMPROVED training with better accuracy**

### 3. **Prediction Scripts**
- `crypto_predictor.py` - Original prediction script (for reference)
- `crypto_predictor_improved.py` - **IMPROVED predictions with multi-day forecasting**

### 4. **Web Interface**
- `crypto_web_predictor.tsx` - React-based web application

## 🚀 Quick Start

### **🖥️ First Time Setup (New Machine)**

#### **Step 1: Clone the Repository**
```bash
# Clone the repository to your local machine
git clone https://github.com/Falco0906/crypto-predictor.git

# Navigate to the project directory
cd crypto_predictor

# Verify the files are present
ls -la
```

#### **Step 2: Set Up Python Environment**
```bash
# Create a virtual environment (recommended)
python3 -m venv crypto_env

# Activate the virtual environment
# On macOS/Linux:
source crypto_env/bin/activate

# On Windows:
# crypto_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

#### **Step 3: Verify Installation**
```bash
# Check if TensorFlow is working
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Check other dependencies
python -c "import pandas, numpy, sklearn, matplotlib; print('All packages imported successfully!')"
```

### **📊 Regular Usage (After Setup)**

#### **Step 1: Update Data (Live from Yahoo Finance)**
```bash
# Get latest cryptocurrency data
python update_data.py
```

#### **Step 2: Train the IMPROVED Model**
```bash
# Train with improved architecture and features
python crypto_training_script_improved.py
```

#### **Step 3: Make IMPROVED Predictions**
```bash
# Load improved model and make multi-day predictions
python crypto_predictor_improved.py
```

#### **Step 4: View Training Visualizations**
```bash
# Check the 'training_visualizations/' folder for auto-generated charts
open training_visualizations/  # On macOS
# explorer training_visualizations\  # On Windows
```

## 🖥️ **Complete Setup Guide for New Machines**

### **📋 Prerequisites**

#### **System Requirements**
- **Operating System**: macOS 10.14+, Ubuntu 18.04+, Windows 10+
- **Python**: Python 3.8+ (3.9+ recommended)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 2GB+ free space
- **Git**: Latest version installed

#### **Install Prerequisites on macOS**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+
brew install python@3.9

# Install Git (if not already installed)
brew install git

# Verify installations
python3 --version
git --version
```

#### **Install Prerequisites on Ubuntu/Debian**
```bash
# Update package list
sudo apt update

# Install Python 3.9+
sudo apt install python3.9 python3.9-venv python3.9-pip

# Install Git
sudo apt install git

# Verify installations
python3.9 --version
git --version
```

#### **Install Prerequisites on Windows**
```bash
# Download and install Python from: https://www.python.org/downloads/
# Make sure to check "Add Python to PATH" during installation

# Download and install Git from: https://git-scm.com/download/win

# Verify installations (in Command Prompt or PowerShell)
python --version
git --version
```

### **🚀 Complete Setup Process**

#### **Step 1: Clone the Repository**
```bash
# Navigate to your desired directory
cd ~/Documents/Projects  # or wherever you want the project

# Clone the repository
git clone https://github.com/YOUR_USERNAME/crypto_predictor.git

# Navigate into the project directory
cd crypto_predictor

# Verify all files are present
ls -la
```

#### **Step 2: Set Up Python Virtual Environment**
```bash
# Create a virtual environment
python3 -m venv crypto_env

# Activate the virtual environment
# On macOS/Linux:
source crypto_env/bin/activate

# On Windows:
# crypto_env\Scripts\activate

# Verify activation (you should see (crypto_env) in your prompt)
which python
```

#### **Step 3: Install Dependencies**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install tensorflow==2.10.0
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install joblib
pip install yfinance
```

#### **Step 4: Verify Installation**
```bash
# Test TensorFlow
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__} installed successfully!')"

# Test other packages
python -c "import pandas as pd; print('✅ Pandas installed successfully!')"
python -c "import numpy as np; print('✅ NumPy installed successfully!')"
python -c "import sklearn; print('✅ Scikit-learn installed successfully!')"
python -c "import matplotlib.pyplot as plt; print('✅ Matplotlib installed successfully!')"
```

#### **Step 5: Test the System**
```bash
# Test data loading
python -c "
import pandas as pd
import os
csv_files = [f for f in os.listdir('.') if f.startswith('coin_') and f.endswith('.csv')]
print(f'✅ Found {len(csv_files)} cryptocurrency data files')
"

# Test basic functionality
python -c "
print('🚀 Testing basic imports and functionality...')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
print('✅ All basic functionality working!')
"
```

### **🔧 Troubleshooting Common Issues**

#### **Python Version Issues**
```bash
# If you get "python3: command not found"
# On macOS, try:
brew install python@3.9
brew link python@3.9

# On Ubuntu, try:
sudo apt install python3.9 python3.9-venv
```

#### **TensorFlow Installation Issues**
```bash
# If TensorFlow fails to install, try:
pip install tensorflow-cpu==2.10.0  # CPU-only version

# Or for Apple Silicon Macs:
pip install tensorflow-macos==2.10.0
```

#### **Permission Issues**
```bash
# If you get permission errors on macOS/Linux:
sudo chown -R $(whoami) crypto_predictor/
chmod +x *.py
```

#### **Virtual Environment Issues**
```bash
# If virtual environment doesn't activate:
# On macOS/Linux:
source crypto_env/bin/activate

# If that doesn't work, recreate it:
rm -rf crypto_env
python3 -m venv crypto_env
source crypto_env/bin/activate
```

### **📱 Platform-Specific Notes**

#### **macOS (Apple Silicon M1/M2)**
```bash
# Install Rosetta 2 if needed
softwareupdate --install-rosetta

# Use specific TensorFlow version for Apple Silicon
pip install tensorflow-macos==2.10.0
pip install tensorflow-metal  # For GPU acceleration
```

#### **Windows**
```bash
# Use Command Prompt or PowerShell (not Git Bash for some operations)
# If you get path issues, try:
python -m pip install --user tensorflow==2.10.0
```

#### **Linux (Ubuntu/Debian)**
```bash
# Install system dependencies
sudo apt install python3-dev build-essential

# If you get SSL errors:
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org tensorflow==2.10.0
```

## 📈 Model Performance

### 🆕 IMPROVED Model Results
- **Better Accuracy**: Resolved extreme -99% predictions
- **Realistic Predictions**: Percentage changes instead of absolute prices
- **Multi-day Forecasting**: Sequential predictions with dynamic feature updates
- **Training Visualizations**: Auto-generated charts for every training run

### 📊 Training Visualization Features
- **Loss Curves**: Training vs validation loss over time
- **Metrics Analysis**: MAE, MSE, and convergence analysis
- **Overfitting Detection**: Loss ratio analysis and thresholds
- **Correlation Matrix**: Training metrics relationships
- **Progress Summary**: Start vs end metrics with improvement percentages
- **Export Options**: High-resolution PNG + CSV data for further analysis

### 🔧 Key Technical Improvements
- **Prediction Target**: Percentage changes instead of absolute prices
- **Scaling**: StandardScaler for better numerical stability
- **Feature Engineering**: 40+ percentage-based technical indicators
- **Model Architecture**: Simplified LSTM for stability
- **Data Quality**: Aggressive outlier removal and filtering

## 🔧 Key Improvements Made

### 1. **🆕 FIXED Poor Accuracy Issues**
- **Root Cause Analysis**: Identified scaling, prediction target, and feature engineering problems
- **Percentage Change Prediction**: Now predicts realistic percentage changes instead of extreme absolute prices
- **Better Scaling**: StandardScaler instead of RobustScaler for numerical stability
- **Improved Data Quality**: Aggressive outlier removal and filtering

### 2. **🆕 Enhanced Feature Engineering**
- **Percentage-Based Features**: All technical indicators normalized by price for stability
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Momentum, Support/Resistance
- **Volatility Features**: Rolling standard deviations and regime detection
- **Time Features**: Day of week, month, weekend indicators
- **Lagged Features**: Previous price changes for sequence learning

### 3. **🆕 Improved Model Architecture**
- **Simplified LSTM**: More stable than complex GRU+Attention
- **Better Regularization**: BatchNormalization + Dropout for generalization
- **Training Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **No Data Leakage**: Removed overlapping sequences that caused overfitting

## 📁 File Structure

```
crypto_predictor/
├── yahoo_finance_updater.py           # Yahoo Finance data fetcher
├── update_data.py                     # Simple data update script
├── crypto_training_script.py          # Original training script (reference)
├── crypto_training_script_improved.py # 🆕 IMPROVED training script
├── crypto_predictor.py                # Original prediction script (reference)
├── crypto_predictor_improved.py       # 🆕 IMPROVED prediction script
├── crypto_web_predictor.tsx           # Web interface
├── trained_model_advanced/            # Original model directory
│   ├── crypto_advanced_model.h5       # Original trained model
│   ├── price_scaler.pkl               # Original price scaler
│   ├── feature_scaler.pkl             # Original feature scaler
│   └── model_metadata.json            # Original model metadata
├── trained_model_improved/            # 🆕 IMPROVED model directory
│   ├── crypto_improved_model.h5       # 🆕 Improved trained model
│   ├── price_scaler.pkl               # 🆕 Improved price scaler
│   ├── feature_scaler.pkl             # 🆕 Improved feature scaler
│   └── model_metadata.json            # 🆕 Improved model metadata
├── training_visualizations/            # 🆕 Auto-generated training charts
│   ├── training_results.png            # Main training overview
│   ├── loss_convergence_analysis.png  # Loss analysis
│   ├── metrics_correlation.png        # Metrics correlation
│   ├── training_progress_summary.png  # Progress summary
│   └── training_history.csv           # Raw training data
├── yahoo_finance_data/                # Live data directory
│   ├── combined_crypto_data.csv       # Combined dataset
│   └── coin_*.csv                     # Individual coin data
├── coin_BTC.csv                       # Bitcoin (live data)
├── coin_ETH.csv                       # Ethereum (live data)
├── coin_SOL.csv                       # Solana (live data)
├── coin_LTC.csv                       # Litecoin (live data)
├── coin_ADA.csv                       # Cardano (live data)
├── coin_DOT.csv                       # Polkadot (live data)
├── coin_LINK.csv                       # Chainlink (live data)
├── coin_UNI.csv                       # Uniswap (live data)
├── coin_MATIC.csv                     # Polygon (live data)
├── coin_AVAX.csv                      # Avalanche (live data)
├── .gitignore                         # Git ignore rules
├── WHY_POOR_ACCURACY.md               # 🆕 Root cause analysis
└── README.md                          # This file
```

## 🔄 **Working with the Repository**

### **📥 Keeping Your Local Copy Updated**
```bash
# Pull the latest changes from GitHub
git pull origin main

# If you have local changes, you might need to stash them first:
git stash
git pull origin main
git stash pop
```

### **📤 Contributing Changes**
```bash
# Check what files have changed
git status

# Add your changes
git add .

# Commit your changes
git commit -m "Description of your changes"

# Push to GitHub
git push origin main
```

### **🔍 Checking Repository Status**
```bash
# See current branch and status
git branch
git status

# See commit history
git log --oneline -10

# See remote repository info
git remote -v
```

## 🎯 Next Steps to Improve Accuracy

### 1. **🆕 Use the IMPROVED System**
```bash
# Train with improved architecture
python crypto_training_script_improved.py

# Make improved predictions
python crypto_predictor_improved.py
```

### 2. **🆕 Monitor Training with Visualizations**
- Check `training_visualizations/` folder after each training run
- Analyze loss convergence and overfitting detection
- Use correlation matrices to understand metric relationships
- Export training data for further analysis

### 3. **🆕 Multi-day Prediction Analysis**
- Use the improved sequential prediction system
- Monitor prediction progression across days
- Analyze feature updates and their impact

### 4. **Future Enhancements**
- **Sentiment Analysis**: Add news and social media sentiment
- **Cross-Asset Correlations**: Include traditional market indicators
- **Regime Detection**: Identify different market conditions
- **Uncertainty Quantification**: Add prediction confidence intervals
- **Ensemble Methods**: Combine multiple model predictions

## 🔍 Understanding the Results

### **🆕 IMPROVED Model Performance**
- **Realistic Predictions**: No more extreme -99% predictions
- **Percentage Changes**: Predicts realistic daily price movements
- **Multi-day Forecasting**: Sequential predictions with feature updates
- **Better Training**: Comprehensive visualizations and monitoring

### **📊 Training Visualization Benefits**
- **Loss Monitoring**: Track training vs validation loss in real-time
- **Overfitting Detection**: Identify when model starts overfitting
- **Metric Relationships**: Understand correlations between different metrics
- **Progress Tracking**: See improvement from start to finish
- **Export Options**: High-resolution charts and CSV data for analysis

### **🎯 Key Improvements Achieved**
- **Fixed Scaling Issues**: StandardScaler for better numerical stability
- **Better Prediction Target**: Percentage changes instead of absolute prices
- **Improved Features**: 40+ percentage-based technical indicators
- **Stable Architecture**: Simplified LSTM without complex attention
- **No Data Leakage**: Removed overlapping sequences that caused overfitting

## 🚨 Important Notes

### **🆕 Model Improvements**
- **Fixed Extreme Predictions**: No more -99% unrealistic forecasts
- **Better Data Quality**: Aggressive outlier removal and filtering
- **Stable Training**: Early stopping and learning rate scheduling
- **Comprehensive Monitoring**: Auto-generated training visualizations

### **Model Limitations**
- **Not Financial Advice**: This is for educational purposes only
- **Market Conditions**: Models trained on historical data may not work in changing markets
- **Volatility**: Crypto markets are extremely volatile and unpredictable
- **Overfitting Risk**: Monitor training visualizations for signs of overfitting

### **Best Practices**
- **Use Improved Scripts**: `crypto_training_script_improved.py` and `crypto_predictor_improved.py`
- **Monitor Training**: Check `training_visualizations/` folder after each run
- **Validate Performance**: Use out-of-sample data for testing
- **Risk Management**: Don't invest more than you can afford to lose
- **Regular Updates**: Retrain with fresh data periodically

## 🛠️ Technical Requirements

### **Python Packages**
```bash
pip install tensorflow==2.10.0
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install joblib
```

### **System Requirements**
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 2GB+ free space
- **GPU**: Optional but recommended for faster training
- **Python**: 3.8+ recommended

## 📞 Support & Troubleshooting

### **Common Issues**
1. **CUDA Errors**: Ignore GPU warnings - model will run on CPU
2. **Memory Issues**: Reduce batch size or sequence length
3. **Import Errors**: Check TensorFlow version compatibility
4. **Data Issues**: Ensure CSV files have proper price columns

### **Getting Help**
- Check error messages carefully
- Verify all dependencies are installed
- Ensure data files are in correct format
- Check file paths and permissions

## 🎉 Success Metrics

### **🆕 What We've ACHIEVED**
✅ **FIXED Poor Accuracy Issues** - Resolved extreme -99% predictions  
✅ **Improved Model Architecture** - LSTM-based with percentage change prediction  
✅ **Enhanced Feature Engineering** - 40+ percentage-based technical indicators  
✅ **Comprehensive Training Visualizations** - Auto-generated charts and analysis  
✅ **Multi-day Predictions** - Sequential forecasting with dynamic feature updates  
✅ **Better Data Handling** - Robust outlier removal and data cleaning  

### **🎯 Next Milestones**
🎯 **Stable Realistic Predictions** - No more extreme values  
🎯 **60%+ Directional Accuracy** - Improve prediction reliability  
🎯 **65%+ Trend Accuracy** - Better multi-day forecasting  
🎯 **Professional Training Monitoring** - Use visualizations for model improvement  
🎯 **Real-time Prediction System** - Live cryptocurrency forecasting  

---

**Disclaimer**: This system is for educational and research purposes only. Cryptocurrency trading involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results.
