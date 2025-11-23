#!/usr/bin/env python3
"""
IMPROVED Cryptocurrency Price Predictor
Loads trained model and makes percentage change predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class ImprovedCryptoPredictor:
    def __init__(self, model_dir=None, verbose=False):
        """
        Initialize improved predictor with trained model
        """
        self.model = None
        self.price_scaler = None
        self.feature_scaler = None
        self.feature_columns = []
        self.sequence_length = 30
        self.verbose = verbose
        
        # Default to data/models_gpu_improved
        if model_dir is None:
            model_dir = PROJECT_ROOT / 'data' / 'models_gpu_improved'
        
        # Try to load the model
        self.load_model(model_dir)
    
    def load_model(self, model_dir):
        """Load the trained model and scalers"""
        try:
            # Convert to Path if string
            if isinstance(model_dir, str):
                model_dir = Path(model_dir)
            
            # Try to load from improved model directory
            model_path = model_dir / 'crypto_improved_model.h5'
            if model_path.exists():
                print(f"📁 Loading improved model from {model_dir}...")
                self.model = load_model(str(model_path))
                
                # Load scalers
                self.price_scaler = joblib.load(str(model_dir / 'price_scaler.pkl'))
                self.feature_scaler = joblib.load(str(model_dir / 'feature_scaler.pkl'))
                
                # Load metadata
                metadata_path = model_dir / 'model_metadata.json'
                if metadata_path.exists():
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.sequence_length = metadata.get('sequence_length', 30)
                
                print("✅ Improved model loaded successfully!")
                
                # Debug info
                print(f"   📊 Model expects {len(self.feature_columns)} features")
                print(f"   📏 Sequence length: {self.sequence_length}")
                print(f"   🔧 First 5 features: {self.feature_columns[:5]}")
                
                return True
                
        except Exception as e:
            print(f"❌ Error loading improved model: {str(e)}")
            return False
    
    def create_improved_features(self, df, verbose=None):
        """Create features matching the training process"""
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print("🔧 Preparing technical features...")
        
        # === PRICE-BASED FEATURES (normalized by price) ===
        # Use percentage changes instead of absolute differences
        df['returns_1d'] = df['close'].pct_change()
        df['returns_3d'] = df['close'].pct_change(3)
        df['returns_7d'] = df['close'].pct_change(7)
        df['returns_14d'] = df['close'].pct_change(14)
        
        # Volatility (rolling standard deviation of returns)
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['returns_1d'].rolling(window=window).std()
        
        # Moving averages (as percentage from current price)
        for window in [5, 10, 20, 50]:
            sma = df['close'].rolling(window=window).mean()
            df[f'sma_{window}d_pct'] = (df['close'] - sma) / sma * 100
            
            ema = df['close'].ewm(span=window).mean()
            df[f'ema_{window}d_pct'] = (df['close'] - ema) / ema * 100
        
        # RSI (more stable calculation)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD components
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / df['close'] * 100  # Normalized
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for window in [20]:
            sma = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = (sma + 2*std - df['close']) / df['close'] * 100
            df[f'bb_lower_{window}'] = (df['close'] - (sma - 2*std)) / df['close'] * 100
            df[f'bb_width_{window}'] = (2*std) / sma * 100
        
        # Momentum indicators (normalized)
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # Support and resistance levels (simplified)
        for window in [20]:
            high = df['close'].rolling(window=window).max()
            low = df['close'].rolling(window=window).min()
            df[f'resistance_{window}d_pct'] = (high - df['close']) / df['close'] * 100
            df[f'support_{window}d_pct'] = (df['close'] - low) / df['close'] * 100
        
        # Volume features (if available)
        if 'volume' in df.columns:
            # Volume moving average
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # On Balance Volume (simplified)
            df['obv'] = (df['volume'] * df['returns_1d'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['obv_divergence'] = (df['obv'] - df['obv_ema']) / df['obv_ema'] * 100
        
        # Time-based features
        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Lagged features (previous price changes)
        df['price_change_pct'] = df['close'].pct_change() * 100
        for lag in [1, 2, 3, 5]:
            df[f'lag_{lag}d_change'] = df['price_change_pct'].shift(lag)
        
        return df
    
    def prepare_prediction_data(self, df):
        """Prepare data for prediction"""
        if len(df) < self.sequence_length:
            print(f"   ❌ Insufficient data: need {self.sequence_length}, have {len(df)}")
            return None
        
        # Get the last sequence_length rows
        recent_data = df.tail(self.sequence_length)
        
        # Ensure all required features exist
        missing_features = []
        for col in self.feature_columns:
            if col not in recent_data.columns:
                missing_features.append(col)
                recent_data[col] = 0  # Fill with 0
        
        if missing_features:
            print(f"   ⚠️  Filled {len(missing_features)} missing features with 0")
        
        # Select features in the exact order used during training
        feature_data = recent_data[self.feature_columns].values
        
        # Check for NaN values and replace them
        if np.isnan(feature_data).any():
            feature_data = np.nan_to_num(feature_data, nan=0.0)
            print("   ⚠️  Replaced NaN values with 0")
        
        # Scale features using the same scaler from training
        try:
            features_scaled = self.feature_scaler.transform(feature_data)
        except Exception as e:
            print(f"   ❌ Scaling error: {str(e)}")
            print(f"   Feature data shape: {feature_data.shape}")
            print(f"   Expected features: {len(self.feature_columns)}")
            return None
        
        # Reshape for LSTM: (1, sequence_length, num_features)
        X = features_scaled.reshape(1, self.sequence_length, len(self.feature_columns))
        
        return X
    
    def predict(self, df, return_confidence=False):
        """Make percentage change prediction with optional confidence estimate"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        try:
            # Create features
            df_with_features = self.create_improved_features(df.copy(), verbose=False)
            
            # Prepare prediction data
            X = self.prepare_prediction_data(df_with_features)
            if X is None:
                return None
            
            # Make prediction
            prediction_scaled = self.model.predict(X, verbose=0)
            prediction_pct = self.price_scaler.inverse_transform(prediction_scaled)[0][0]
            
            # Calculate basic confidence estimate (based on prediction magnitude)
            # Smaller predictions are generally more reliable
            if return_confidence:
                confidence = max(0, min(100, 100 - abs(prediction_pct) * 2))
                return prediction_pct, confidence
            
            return prediction_pct
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Prediction error: {str(e)}")
            return None
    
    def predict_price_from_change(self, current_price, predicted_change_pct):
        """Convert percentage change to actual price"""
        # Ensure prediction is within reasonable bounds
        predicted_change_pct = np.clip(predicted_change_pct, -20, 20)  # Max ±20% change
        
        # Calculate new price
        new_price = current_price * (1 + predicted_change_pct / 100)
        
        return new_price, predicted_change_pct
    
    def predict_next_days(self, df, days=3):
        """Predict prices for the next few days with fresh predictions each day"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        predictions = []
        current_df = df.copy()
        
        for day in range(days):
            # Make percentage change prediction based on current data
            pred_change_pct = self.predict(current_df)
            if pred_change_pct is None:
                break
            
            # Get current price
            current_price = current_df['close'].iloc[-1]
            
            # Convert to actual price
            predicted_price, clipped_change = self.predict_price_from_change(current_price, pred_change_pct)
            
            predictions.append({
                'day': day + 1,
                'predicted_change_pct': clipped_change,
                'predicted_price': predicted_price,
                'current_price': current_price
            })
            
            # Create a new row with the predicted price and update all features
            new_row = current_df.iloc[-1:].copy()
            new_row['close'] = predicted_price
            
            # Update the index to the next day
            if hasattr(new_row.index, 'dayofweek'):
                new_row.index = new_row.index + pd.Timedelta(days=1)
            else:
                # If no datetime index, create a simple increment
                new_row.index = [new_row.index[-1] + 1]
            
            # Add the new row to the dataframe
            current_df = pd.concat([current_df, new_row])
            
            # IMPORTANT: Recalculate all features for the new data
            # This ensures the next prediction uses updated technical indicators
            current_df = self.create_improved_features(current_df, verbose=False)
            
            # Remove any NaN rows that might have been created
            current_df = current_df.dropna()
            
            # Ensure we have enough data for the next prediction
            if len(current_df) < self.sequence_length:
                print(f"   ⚠️  Insufficient data for Day {day + 2}, stopping predictions")
                break
        
        return predictions

def main():
    """Main function for making predictions"""
    print("🔮 IMPROVED Cryptocurrency Price Predictor")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ImprovedCryptoPredictor(verbose=False)
    
    if predictor.model is None:
        print("❌ Failed to load improved model. Please train the model first.")
        print("💡 Run: python -m src.crypto_training_script_improved")
        return
    
    # Look for CSV files in data/raw_data first, then project root
    raw_data_dir = PROJECT_ROOT / 'data' / 'raw_data'
    csv_files = [
        'coin_BTC.csv', 'coin_ETH.csv', 'coin_SOL.csv', 'coin_LTC.csv', 
        'coin_ADA.csv', 'coin_DOT.csv', 'coin_LINK.csv', 
        'coin_MATIC.csv', 'coin_AVAX.csv'
    ]
    
    # Try data/raw_data first, then project root
    found_files = []
    for csv_file in csv_files:
        # Try raw_data directory first
        file_path = raw_data_dir / csv_file
        if file_path.exists():
            found_files.append((str(file_path), csv_file))
        # Fallback to project root
        elif (PROJECT_ROOT / csv_file).exists():
            found_files.append((str(PROJECT_ROOT / csv_file), csv_file))
    
    if not found_files:
        print("❌ No CSV files found!")
        print(f"   Looking in: {raw_data_dir} and {PROJECT_ROOT}")
        return
    
    print(f"📊 Processing {len(found_files)} cryptocurrency files...\n")
    
    for file_path, csv_file in found_files:
        try:
            # Load data
            df = pd.read_csv(file_path)
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
                print(f"⚠️  Skipping {csv_file}: No price data found")
                continue
            
            # Check for valid current price (skip zero or near-zero prices)
            current_price = df['close'].iloc[-1]
            if current_price is None or pd.isna(current_price) or current_price <= 1e-6:
                print(f"⚠️  Skipping {csv_file}: Invalid current price ({current_price})")
                continue
            
            # Make single prediction
            predicted_change_pct = predictor.predict(df)
            
            if predicted_change_pct is None:
                print(f"⚠️  Skipping {csv_file}: Prediction failed")
                continue
            
            # Convert to actual price with bounds checking
            predicted_price, clipped_change = predictor.predict_price_from_change(
                current_price, predicted_change_pct
            )
            
            change = clipped_change
            direction = "📈" if change > 0 else "📉"
            
            print("─" * 60)
            print(f"📊 Coin: {csv_file.replace('coin_', '').replace('.csv', '').upper()}")
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Predicted Change: {clipped_change:+.2f}% {direction}")
            print(f"   Predicted Price: ${predicted_price:,.2f}")
            
            # Predict next 3 days
            next_predictions = predictor.predict_next_days(df, days=3)
            if next_predictions:
                print(f"   Next 3 days:")
                for pred in next_predictions:
                    print(f"     Day {pred['day']}: ${pred['predicted_price']:,.2f} ({pred['predicted_change_pct']:+.2f}%)")
                
                # Show the progression
                if len(next_predictions) > 1:
                    print(f"   📈 Prediction Progression:")
                    for i in range(len(next_predictions) - 1):
                        day1_price = next_predictions[i]['predicted_price']
                        day2_price = next_predictions[i + 1]['predicted_price']
                        day_change = ((day2_price - day1_price) / day1_price) * 100
                        print(f"     Day {i+1} → Day {i+2}: {day_change:+.2f}%")
            print()
                
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {str(e)}\n")
            continue
    
    print("=" * 60)
    print("✅ Prediction complete!")

if __name__ == "__main__":
    main()
