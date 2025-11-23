#!/usr/bin/env python3
"""
Cryptocurrency Price Predictor
Loads trained model and makes predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class CryptoPredictor:
    def __init__(self, model_dir='trained_model_advanced'):
        """
        Initialize predictor with trained model
        """
        self.model = None
        self.price_scaler = None
        self.feature_scaler = None
        self.feature_columns = []
        self.sequence_length = 45
        
        # Try to load the model
        self.load_model(model_dir)
    
    def load_model(self, model_dir):
        """Load the trained model and scalers"""
        try:
            # Try to load from advanced model directory
            model_path = os.path.join(model_dir, 'crypto_advanced_model.h5')
            if os.path.exists(model_path):
                print(f"📁 Loading model from {model_dir}...")
                self.model = load_model(model_path)
                
                # Load scalers
                self.price_scaler = joblib.load(os.path.join(model_dir, 'price_scaler.pkl'))
                self.feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
                
                # Load metadata
                metadata_path = os.path.join(model_dir, 'model_metadata.json')
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.sequence_length = metadata.get('sequence_length', 45)
                
                print("✅ Model loaded successfully!")
                return True
                
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def create_features(self, df):
        """Create features for prediction (complete version matching training)"""
        print("🔧 Creating features for prediction...")
        
        # === BASIC PRICE FEATURES ===
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_acceleration'] = df['returns'].diff()
        df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # === MULTI-TIMEFRAME ANALYSIS ===
        for window in [3, 7, 14, 21, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_position_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
            df[f'sma_slope_{window}'] = df[f'sma_{window}'].diff(5) / df[f'sma_{window}'].shift(5)
        
        # === VOLATILITY REGIME DETECTION ===
        for window in [5, 14, 30]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'volatility_regime_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(window=50).mean()
        
        # === ADVANCED MOMENTUM INDICATORS ===
        # Williams %R
        high_14 = df['close'].rolling(window=14).max()
        low_14 = df['close'].rolling(window=14).min()
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        # Commodity Channel Index (CCI)
        typical_price = df['close']
        df['cci'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * typical_price.rolling(window=20).std())
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # === RSI WITH MULTIPLE TIMEFRAMES ===
        for rsi_period in [9, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
            df[f'rsi_{rsi_period}_norm'] = (df[f'rsi_{rsi_period}'] - 50) / 50
        
        # === MACD FAMILY ===
        # Standard MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_histogram_slope'] = df['macd_histogram'].diff()
        
        # Fast MACD
        ema_5 = df['close'].ewm(span=5).mean()
        ema_13 = df['close'].ewm(span=13).mean()
        df['macd_fast'] = ema_5 - ema_13
        df['macd_fast_signal'] = df['macd_fast'].ewm(span=5).mean()
        
        # === BOLLINGER BANDS ADVANCED ===
        for bb_window, bb_std in [(20, 2), (10, 1.5), (50, 2.5)]:
            sma = df['close'].rolling(window=bb_window).mean()
            std = df['close'].rolling(window=bb_window).std()
            df[f'bb_upper_{bb_window}'] = sma + (std * bb_std)
            df[f'bb_lower_{bb_window}'] = sma - (std * bb_std)
            df[f'bb_position_{bb_window}'] = (df['close'] - df[f'bb_lower_{bb_window}']) / (df[f'bb_upper_{bb_window}'] - df[f'bb_lower_{bb_window}'])
            df[f'bb_squeeze_{bb_window}'] = (df[f'bb_upper_{bb_window}'] - df[f'bb_lower_{bb_window}']) / sma
        
        # === SUPPORT/RESISTANCE LEVELS ===
        for sr_window in [10, 20, 50]:
            df[f'support_{sr_window}'] = df['close'].rolling(window=sr_window).min()
            df[f'resistance_{sr_window}'] = df['close'].rolling(window=sr_window).max()
            df[f'support_strength_{sr_window}'] = (df['close'] - df[f'support_{sr_window}']) / df[f'support_{sr_window}']
            df[f'resistance_strength_{sr_window}'] = (df[f'resistance_{sr_window}'] - df['close']) / df['close']
        
        # === VOLUME ANALYSIS (if available) ===
        if 'volume' in df.columns:
            # Volume moving averages
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # On Balance Volume (OBV)
            df['obv'] = (df['volume'] * df['returns'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['obv_divergence'] = df['obv'] - df['obv_ema']
            
            # Volume Price Trend (VPT)
            df['vpt'] = (df['volume'] * df['returns']).cumsum()
            
            # Money Flow Index components
            df['volume_weighted_price'] = df['close'] * df['volume']
        
        # === PATTERN RECOGNITION FEATURES ===
        # Higher highs, lower lows detection
        df['local_maxima'] = df['close'].rolling(window=5, center=True).max() == df['close']
        df['local_minima'] = df['close'].rolling(window=5, center=True).min() == df['close']
        
        # Trend consistency (how many periods in same direction)
        returns_sign = df['returns'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        df['trend_consistency'] = returns_sign.rolling(window=10).sum() / 10
        
        # === TIME-BASED FEATURES ===
        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # === CROSS-ASSET FEATURES ===
        df['coin_rank'] = df['close'].rank(pct=True)
        
        return df
    
    def prepare_prediction_data(self, df):
        """Prepare data for prediction"""
        if self.feature_scaler is None or len(self.feature_columns) == 0:
            print("❌ Model not loaded properly!")
            return None
        
        # Select only the features used during training
        available_features = [col for col in self.feature_columns if col in df.columns]
        if len(available_features) == 0:
            print("❌ No matching features found!")
            return None
        
        # Fill missing values
        df_features = df[available_features].fillna(method='ffill').fillna(0)
        
        # Scale features
        features_scaled = self.feature_scaler.transform(df_features)
        
        # Create sequence
        if len(features_scaled) < self.sequence_length:
            print(f"❌ Need at least {self.sequence_length} data points!")
            return None
        
        # Take the last sequence_length points
        sequence = features_scaled[-self.sequence_length:]
        return sequence.reshape(1, self.sequence_length, len(available_features))
    
    def predict(self, df):
        """Make price prediction"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        # Create features
        df_with_features = self.create_features(df.copy())
        
        # Prepare prediction data
        X = self.prepare_prediction_data(df_with_features)
        if X is None:
            return None
        
        # Make prediction
        prediction_scaled = self.model.predict(X, verbose=0)
        prediction = self.price_scaler.inverse_transform(prediction_scaled)[0][0]
        
        return prediction
    
    def predict_next_days(self, df, days=5):
        """Predict prices for the next few days"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        predictions = []
        current_df = df.copy()
        
        for day in range(days):
            # Make prediction
            pred_price = self.predict(current_df)
            if pred_price is None:
                break
            
            predictions.append(pred_price)
            
            # Add prediction to dataframe for next iteration
            new_row = current_df.iloc[-1:].copy()
            new_row['close'] = pred_price
            new_row.index = new_row.index + pd.Timedelta(days=1)
            current_df = pd.concat([current_df, new_row])
        
        return predictions

def main():
    """Main function for making predictions"""
    print("🔮 Cryptocurrency Price Predictor")
    print("=" * 40)
    
    # Initialize predictor
    predictor = CryptoPredictor()
    
    if predictor.model is None:
        print("❌ Failed to load model. Please train the model first.")
        return
    
    # Example: Load some data and make predictions
    csv_files = ['coin_BTC.csv', 'coin_ETH.csv', 'coin_SOL.csv', 'coin_LTC.csv', 'coin_ADA.csv', 'coin_DOT.csv', 'coin_LINK.csv', 'coin_UNI.csv', 'coin_MATIC.csv', 'coin_AVAX.csv']
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\n📊 Making predictions for {csv_file}...")
            
            try:
                # Load data
                df = pd.read_csv(csv_file)
                df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
                
                # Handle dates
                date_cols = ['date', 'timestamp', 'time', 'datetime']
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.sort_values(col).set_index(col)
                        break
                
                # Standardize price columns
                if 'close' not in df.columns:
                    price_cols = ['price', 'closing_price', 'close_price', 'last']
                    for col in price_cols:
                        if col in df.columns:
                            df['close'] = df[col]
                            break
                
                if 'close' not in df.columns:
                    continue
                
                # Make single prediction
                current_price = df['close'].iloc[-1]
                predicted_price = predictor.predict(df)
                
                if predicted_price is not None:
                    change = ((predicted_price - current_price) / current_price) * 100
                    direction = "📈" if change > 0 else "📉"
                    
                    print(f"   Current Price: ${current_price:.2f}")
                    print(f"   Predicted Price: ${predicted_price:.2f}")
                    print(f"   Expected Change: {change:+.2f}% {direction}")
                    
                    # Predict next 3 days
                    next_predictions = predictor.predict_next_days(df, days=3)
                    if next_predictions:
                        print(f"   Next 3 days: {[f'${p:.2f}' for p in next_predictions]}")
                
            except Exception as e:
                print(f"   ❌ Error processing {csv_file}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
