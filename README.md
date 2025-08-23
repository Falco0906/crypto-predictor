# 🚀 Cryptocurrency Price Prediction System

A comprehensive machine learning system for predicting cryptocurrency prices using advanced deep learning techniques.

## 📊 Current Status

✅ **Fixed Compatibility Issues** - Works with TensorFlow 2.10.0  
✅ **Successfully Trained Model** - 51.13% directional accuracy  
✅ **Advanced Feature Engineering** - 95+ technical indicators  
✅ **Hybrid Architecture** - GRU + Attention mechanism  

## 🏗️ System Architecture

### 1. **Data Management**
- `yahoo_finance_updater.py` - Fetches live data from Yahoo Finance
- `update_data.py` - Simple script to update all cryptocurrency data

### 2. **Training Scripts**
- `crypto_training_script.py` - Working model with 51.13% directional accuracy

### 3. **Prediction Script**
- `crypto_predictor.py` - Loads trained model and makes predictions

### 4. **Web Interface**
- `crypto_web_predictor.tsx` - React-based web application

## 🚀 Quick Start

### Step 1: Update Data (Live from Yahoo Finance)
```bash
# Get latest cryptocurrency data
python update_data.py
```

### Step 2: Train the Model (Optional)
```bash
# Train with fresh data
python crypto_training_script.py
```

### Step 3: Make Predictions
```bash
# Load trained model and make predictions
python crypto_predictor.py
```

## 📈 Model Performance

### Current Results (Fixed Model)
- **Directional Accuracy**: 51.13%
- **Trend Accuracy**: 51.69%
- **Trading Return**: 1732.34%
- **Sharpe Ratio**: 1.099

### Current Results (Working Model)
- **Directional Accuracy**: 51.13%
- **Feature Engineering**: 95+ technical indicators
- **Architecture**: GRU + Attention mechanism
- **Training**: 264 epochs with early stopping

## 🔧 Key Improvements Made

### 1. **Fixed TensorFlow Compatibility**
- Replaced `AdamW` with `Adam` optimizer
- Added proper error handling and validation
- Fixed import issues for TF 2.10.0

### 2. **Enhanced Feature Engineering**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Williams %R
- **Volatility Features**: Multiple timeframes, regime detection
- **Pattern Recognition**: Support/resistance, trend consistency
- **Time Features**: Cyclical encoding, day/month patterns
- **Statistical Features**: Skewness, kurtosis, rolling statistics

### 3. **Improved Model Architecture**
- **Hybrid Model**: CNN for pattern recognition + GRU for sequences
- **Attention Mechanism**: Focus on important time steps
- **Regularization**: L1/L2 regularization, dropout, batch normalization
- **Better Training**: Learning rate scheduling, early stopping

## 📁 File Structure

```
crypto_predictor/
├── yahoo_finance_updater.py           # Yahoo Finance data fetcher
├── update_data.py                     # Simple data update script
├── crypto_training_script.py          # Working training script
├── crypto_predictor.py                # Prediction script
├── crypto_web_predictor.tsx           # Web interface
├── trained_model_advanced/            # Working model directory
│   ├── crypto_advanced_model.h5       # Trained model
│   ├── price_scaler.pkl               # Price scaler
│   ├── feature_scaler.pkl             # Feature scaler
│   └── model_metadata.json            # Model metadata
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
└── README.md                          # This file
```

## 🎯 Next Steps to Improve Accuracy

### 1. **Use Live Data**
```bash
# Update data daily/weekly
python update_data.py
```

### 2. **Retrain with Fresh Data**
```bash
# Train model with latest data
python crypto_training_script.py
```

### 3. **Make Real-time Predictions**
```bash
# Predict with latest data
python crypto_predictor.py
```

### 4. **Feature Engineering Improvements**
- Add more market microstructure indicators
- Include sentiment analysis data
- Add cross-asset correlations
- Implement regime-switching models

### 3. **Model Architecture Improvements**
- Try different attention mechanisms
- Experiment with transformer architectures
- Implement ensemble methods
- Add uncertainty quantification

### 4. **Data Improvements**
- Get more recent data
- Add more cryptocurrencies
- Include on-chain metrics
- Add news/sentiment data

## 🔍 Understanding the Results

### **Directional Accuracy (51.13%)**
- This means the model correctly predicts price direction 51.13% of the time
- Random guessing would be 50%, so we're slightly above random
- **Target**: Get this above 60% for profitable trading

### **Trend Accuracy (51.69%)**
- Model predicts 3-day trends correctly 51.69% of the time
- **Target**: Get this above 65% for swing trading

### **Trading Return (1732.34%)**
- This is the simulated return from following model predictions
- **Caution**: This is likely overfitted - real trading would be much lower

## 🚨 Important Notes

### **Model Limitations**
- **Not Financial Advice**: This is for educational purposes only
- **Overfitting Risk**: High returns suggest potential overfitting
- **Market Conditions**: Models trained on historical data may not work in changing markets
- **Volatility**: Crypto markets are extremely volatile and unpredictable

### **Best Practices**
- Always validate on out-of-sample data
- Use proper risk management
- Don't invest more than you can afford to lose
- Consider transaction costs and slippage
- Monitor model performance regularly

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

### **What We've Achieved**
✅ Fixed all compatibility issues  
✅ Successfully trained a working model  
✅ Created comprehensive feature engineering  
✅ Built prediction pipeline  
✅ Achieved baseline performance  

### **Next Milestones**
🎯 **60%+ Directional Accuracy**  
🎯 **65%+ Trend Accuracy**  
🎯 **Stable Trading Returns**  
🎯 **Real-time Predictions**  

---

**Disclaimer**: This system is for educational and research purposes only. Cryptocurrency trading involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results.
