#!/usr/bin/env python3
"""
FINE-TUNED Cryptocurrency Price Prediction Model
Building on the improved version to push directional accuracy >60%
Compatible with TensorFlow 2.10.0
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU, Input, Attention, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam  # Removed AdamW for compatibility
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class FineTunedCryptoPricePredictor:
    def __init__(self, sequence_length=45, prediction_days=1):
        """
        Fine-tuned predictor with advanced techniques
        """
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.model = None
        self.price_scaler = RobustScaler()  # More robust to outliers
        self.feature_scaler = RobustScaler()
        self.feature_columns = []
        self.target_column = 'close'
        
    def load_and_clean_data(self, file_paths):
        """Enhanced data cleaning with more sophisticated filtering"""
        print("📊 Loading data with advanced cleaning...")
        
        all_data = []
        
        for file_path in file_paths:
            print(f"   Processing {file_path}...")
            
            try:
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
                    
                    # Fallback to first numeric column
                    if 'close' not in df.columns:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            df['close'] = df[numeric_cols[0]]
                
                # Advanced outlier removal
                df = df[df['close'] > 0]
                
                # Remove extreme price spikes (z-score > 3)
                price_returns = df['close'].pct_change()
                z_scores = np.abs((price_returns - price_returns.mean()) / price_returns.std())
                df = df[z_scores < 3]
                
                # Remove low-volume periods if volume exists
                if 'volume' in df.columns:
                    volume_threshold = df['volume'].quantile(0.1)  # Remove bottom 10%
                    df = df[df['volume'] >= volume_threshold]
                
                # Ensure minimum data points
                if len(df) < 100:
                    print(f"   ⚠️  Skipping {file_path}: insufficient data ({len(df)} records)")
                    continue
                
                coin_name = os.path.basename(file_path).replace('.csv', '').replace('coin_', '')
                df['coin'] = coin_name.upper()
                
                print(f"   ✓ {coin_name}: {len(df)} clean records")
                all_data.append(df)
                
            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {str(e)}")
                continue
        
        if not all_data:
            print("❌ No valid data files found!")
            return None
            
        combined_df = pd.concat(all_data, ignore_index=False).sort_index()
        print(f"   Total clean records: {len(combined_df)}")
        return combined_df
    
    def create_advanced_features(self, df):
        """Create more sophisticated features with market microstructure indicators"""
        print("🔧 Creating advanced feature set...")
        
        enhanced_data = []
        
        for coin in df['coin'].unique():
            coin_df = df[df['coin'] == coin].copy()
            
            if len(coin_df) < 100:
                continue
            
            print(f"   Processing {coin} with advanced features...")
            
            # === BASIC PRICE FEATURES ===
            coin_df['returns'] = coin_df['close'].pct_change()
            coin_df['log_returns'] = np.log(coin_df['close'] / coin_df['close'].shift(1))
            coin_df['price_acceleration'] = coin_df['returns'].diff()
            
            # === MULTI-TIMEFRAME ANALYSIS ===
            for window in [3, 7, 14, 21, 50]:
                coin_df[f'sma_{window}'] = coin_df['close'].rolling(window=window).mean()
                coin_df[f'ema_{window}'] = coin_df['close'].ewm(span=window).mean()
                coin_df[f'price_position_{window}'] = (coin_df['close'] - coin_df[f'sma_{window}']) / coin_df[f'sma_{window}']
                
                # Slope of moving averages (trend strength)
                coin_df[f'sma_slope_{window}'] = coin_df[f'sma_{window}'].diff(5) / coin_df[f'sma_{window}'].shift(5)
            
            # === VOLATILITY REGIME DETECTION ===
            for window in [5, 14, 30]:
                coin_df[f'volatility_{window}'] = coin_df['returns'].rolling(window=window).std()
                coin_df[f'volatility_regime_{window}'] = coin_df[f'volatility_{window}'] / coin_df[f'volatility_{window}'].rolling(window=50).mean()
            
            # === ADVANCED MOMENTUM INDICATORS ===
            # Williams %R
            high_14 = coin_df['close'].rolling(window=14).max()  # Using close as proxy for high
            low_14 = coin_df['close'].rolling(window=14).min()   # Using close as proxy for low
            coin_df['williams_r'] = -100 * (high_14 - coin_df['close']) / (high_14 - low_14)
            
            # Commodity Channel Index (CCI)
            typical_price = coin_df['close']  # Simplified since we only have close
            coin_df['cci'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * typical_price.rolling(window=20).std())
            
            # Rate of Change (ROC)
            for period in [5, 10, 20]:
                coin_df[f'roc_{period}'] = (coin_df['close'] - coin_df['close'].shift(period)) / coin_df['close'].shift(period) * 100
            
            # === RSI WITH MULTIPLE TIMEFRAMES ===
            for rsi_period in [9, 14, 21]:
                delta = coin_df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                coin_df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
                coin_df[f'rsi_{rsi_period}_norm'] = (coin_df[f'rsi_{rsi_period}'] - 50) / 50
            
            # === MACD FAMILY ===
            # Standard MACD
            ema_12 = coin_df['close'].ewm(span=12).mean()
            ema_26 = coin_df['close'].ewm(span=26).mean()
            coin_df['macd'] = ema_12 - ema_26
            coin_df['macd_signal'] = coin_df['macd'].ewm(span=9).mean()
            coin_df['macd_histogram'] = coin_df['macd'] - coin_df['macd_signal']
            coin_df['macd_histogram_slope'] = coin_df['macd_histogram'].diff()
            
            # Fast MACD
            ema_5 = coin_df['close'].ewm(span=5).mean()
            ema_13 = coin_df['close'].ewm(span=13).mean()
            coin_df['macd_fast'] = ema_5 - ema_13
            coin_df['macd_fast_signal'] = coin_df['macd_fast'].ewm(span=5).mean()
            
            # === BOLLINGER BANDS ADVANCED ===
            for bb_window, bb_std in [(20, 2), (10, 1.5), (50, 2.5)]:
                sma = coin_df['close'].rolling(window=bb_window).mean()
                std = coin_df['close'].rolling(window=bb_window).std()
                coin_df[f'bb_upper_{bb_window}'] = sma + (std * bb_std)
                coin_df[f'bb_lower_{bb_window}'] = sma - (std * bb_std)
                coin_df[f'bb_position_{bb_window}'] = (coin_df['close'] - coin_df[f'bb_lower_{bb_window}']) / (coin_df[f'bb_upper_{bb_window}'] - coin_df[f'bb_lower_{bb_window}'])
                coin_df[f'bb_squeeze_{bb_window}'] = (coin_df[f'bb_upper_{bb_window}'] - coin_df[f'bb_lower_{bb_window}']) / sma
            
            # === SUPPORT/RESISTANCE LEVELS ===
            for sr_window in [10, 20, 50]:
                coin_df[f'support_{sr_window}'] = coin_df['close'].rolling(window=sr_window).min()
                coin_df[f'resistance_{sr_window}'] = coin_df['close'].rolling(window=sr_window).max()
                coin_df[f'support_strength_{sr_window}'] = (coin_df['close'] - coin_df[f'support_{sr_window}']) / coin_df[f'support_{sr_window}']
                coin_df[f'resistance_strength_{sr_window}'] = (coin_df[f'resistance_{sr_window}'] - coin_df['close']) / coin_df['close']
            
            # === VOLUME ANALYSIS (if available) ===
            if 'volume' in coin_df.columns:
                # Volume moving averages
                coin_df['volume_sma_20'] = coin_df['volume'].rolling(window=20).mean()
                coin_df['volume_ratio'] = coin_df['volume'] / coin_df['volume_sma_20']
                
                # On Balance Volume (OBV)
                coin_df['obv'] = (coin_df['volume'] * coin_df['returns'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
                coin_df['obv_ema'] = coin_df['obv'].ewm(span=20).mean()
                coin_df['obv_divergence'] = coin_df['obv'] - coin_df['obv_ema']
                
                # Volume Price Trend (VPT)
                coin_df['vpt'] = (coin_df['volume'] * coin_df['returns']).cumsum()
                
                # Money Flow Index components
                coin_df['volume_weighted_price'] = coin_df['close'] * coin_df['volume']
                
            # === PATTERN RECOGNITION FEATURES ===
            # Higher highs, lower lows detection
            coin_df['local_maxima'] = coin_df['close'].rolling(window=5, center=True).max() == coin_df['close']
            coin_df['local_minima'] = coin_df['close'].rolling(window=5, center=True).min() == coin_df['close']
            
            # Trend consistency (how many periods in same direction)
            returns_sign = coin_df['returns'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            coin_df['trend_consistency'] = returns_sign.rolling(window=10).sum() / 10
            
            # === TIME-BASED FEATURES ===
            if hasattr(coin_df.index, 'dayofweek'):
                coin_df['day_of_week'] = coin_df.index.dayofweek
                coin_df['month'] = coin_df.index.month
                coin_df['quarter'] = coin_df.index.quarter
                coin_df['is_weekend'] = (coin_df['day_of_week'] >= 5).astype(int)
            
            # === CROSS-ASSET FEATURES (if multiple coins) ===
            coin_df['coin_rank'] = coin_df['close'].rank(pct=True)  # Percentile rank of price
            
            enhanced_data.append(coin_df)
        
        if not enhanced_data:
            print("❌ No enhanced data created!")
            return None
            
        final_df = pd.concat(enhanced_data, ignore_index=False)
        final_df = final_df.dropna()
        
        print(f"   ✓ Created {final_df.shape[1]} features for {len(final_df)} records")
        return final_df
    
    def prepare_advanced_training_data(self, df):
        """Prepare training data with feature selection and engineering"""
        print("📦 Preparing advanced training data...")
        
        if df is None or len(df) == 0:
            print("❌ No data to prepare!")
            return None, None
        
        # Feature selection - remove highly correlated and low-importance features
        exclude_cols = ['close', 'coin', 'local_maxima', 'local_minima']
        feature_cols = [col for col in df.columns if col not in exclude_cols and 
                       df[col].dtype in ['float64', 'int64'] and not df[col].isna().all()]
        
        # Remove features with very low variance
        feature_variances = df[feature_cols].var()
        feature_cols = feature_variances[feature_variances > 1e-6].index.tolist()
        
        if len(feature_cols) == 0:
            print("❌ No valid features found!")
            return None, None
        
        self.feature_columns = feature_cols
        print(f"   Selected {len(feature_cols)} features after filtering")
        
        # Use RobustScaler (less sensitive to outliers)
        features_scaled = self.feature_scaler.fit_transform(df[feature_cols])
        target_scaled = self.price_scaler.fit_transform(df[['close']])
        
        # Create sequences with overlap for data augmentation
        X_sequences = []
        y_targets = []
        
        for coin in df['coin'].unique():
            coin_mask = df['coin'] == coin
            coin_features = features_scaled[coin_mask]
            coin_targets = target_scaled[coin_mask].flatten()
            
            # Create overlapping sequences (stride = 1 instead of sequence_length)
            for i in range(self.sequence_length, len(coin_features)):
                X_sequences.append(coin_features[i-self.sequence_length:i])
                y_targets.append(coin_targets[i])
        
        if len(X_sequences) == 0:
            print("❌ No sequences created!")
            return None, None
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        print(f"   ✓ Created {len(X)} sequences with shape {X.shape}")
        return X, y
    
    def build_attention_model(self, input_shape):
        """Build advanced model with attention mechanism"""
        print("🏗️ Building advanced attention-based model...")
        
        inputs = Input(shape=input_shape)
        
        # First GRU layer
        gru1 = GRU(64, return_sequences=True)(inputs)
        gru1 = BatchNormalization()(gru1)
        gru1 = Dropout(0.2)(gru1)
        
        # Second GRU layer  
        gru2 = GRU(32, return_sequences=True)(gru1)
        gru2 = BatchNormalization()(gru2)
        gru2 = Dropout(0.2)(gru2)
        
        # Attention mechanism (simplified)
        attention = Dense(1, activation='tanh')(gru2)
        attention = tf.nn.softmax(attention, axis=1)
        attended = tf.reduce_sum(gru2 * attention, axis=1)
        
        # Dense layers
        dense1 = Dense(32, activation='relu')(attended)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(16, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use Adam instead of AdamW for compatibility
        optimizer = Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        print(f"   ✓ Advanced model built with {model.count_params():,} parameters")
        return model
    
    def custom_learning_rate_schedule(self, epoch, lr):
        """Custom learning rate schedule"""
        if epoch < 50:
            return lr
        elif epoch < 100:
            return lr * 0.5
        elif epoch < 150:
            return lr * 0.25
        else:
            return lr * 0.1
    
    def train_advanced_model(self, X, y, validation_split=0.2, epochs=200):
        """Train with advanced techniques"""
        print("🚀 Training advanced model...")
        
        if X is None or y is None:
            print("❌ No training data available!")
            return None
        
        self.model = self.build_attention_model((X.shape[1], X.shape[2]))
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(
                patience=40,
                restore_best_weights=True,
                monitor='val_loss',
                min_delta=0.0001
            ),
            ReduceLROnPlateau(
                factor=0.7,
                patience=20,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            ModelCheckpoint(
                'best_crypto_model_improved.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            LearningRateScheduler(self.custom_learning_rate_schedule)
        ]
        
        # Train with class weights to handle imbalanced data
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=12,  # Even smaller batch size
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        self.training_history = history.history
        print("✅ Advanced training completed!")
        return history
    
    def evaluate_advanced_model(self, X, y):
        """Advanced evaluation with multiple metrics"""
        print("📈 Evaluating advanced model...")
        
        if self.model is None:
            print("❌ No model to evaluate!")
            return None, None, None
        
        predictions_scaled = self.model.predict(X)
        predictions = self.price_scaler.inverse_transform(predictions_scaled)
        y_actual = self.price_scaler.inverse_transform(y.reshape(-1, 1))
        
        predictions = predictions.flatten()
        y_actual = y_actual.flatten()
        
        # Standard metrics
        mse = mean_squared_error(y_actual, predictions)
        mae = mean_absolute_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        y_direction = np.diff(y_actual) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(y_direction == pred_direction) * 100
        
        # Trend accuracy (3-day trends)
        y_trend = np.diff(y_actual, n=3) > 0
        pred_trend = np.diff(predictions, n=3) > 0
        trend_accuracy = np.mean(y_trend == pred_trend) * 100
        
        # Profit-based metrics (simulate trading)
        returns_actual = np.diff(y_actual) / y_actual[:-1]
        returns_predicted = np.diff(predictions) / y_actual[:-1]  # Use actual prices for fair comparison
        
        # Simple trading strategy: buy if predicted return > 0
        trading_signals = returns_predicted > 0
        strategy_returns = returns_actual * trading_signals
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        mape = np.mean(np.abs((y_actual - predictions) / np.maximum(y_actual, 1e-8))) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'trend_accuracy': trend_accuracy,
            'total_return': total_return * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio
        }
        
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: ${mae:.2f}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"   3-Day Trend Accuracy: {trend_accuracy:.2f}%")
        print(f"   Simulated Trading Return: {total_return*100:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        
        return metrics, predictions, y_actual
    
    def save_advanced_model(self, model_dir='trained_model_advanced'):
        """Save the advanced model"""
        print("💾 Saving advanced model...")
        
        if self.model is None:
            print("❌ No model to save!")
            return
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.model.save(os.path.join(model_dir, 'crypto_advanced_model.h5'))
        joblib.dump(self.price_scaler, os.path.join(model_dir, 'price_scaler.pkl'))
        joblib.dump(self.feature_scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
        
        metadata = {
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': 'Advanced_GRU_with_Attention',
            'training_date': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'num_features': len(self.feature_columns),
            'architecture': 'GRU + Attention + Advanced Features'
        }
        
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Advanced model saved to {model_dir}/")

def main():
    """Main training function for advanced model"""
    print("🚀 ADVANCED Cryptocurrency Price Prediction Training")
    print("=" * 60)
    
    CSV_FILES = [
        'coin_BTC.csv',
        'coin_ETH.csv',
        'coin_SOL.csv',
        'coin_LTC.csv',
        'coin_ADA.csv',
        'coin_DOT.csv',
        'coin_LINK.csv',
        'coin_UNI.csv',
        'coin_MATIC.csv',
        'coin_AVAX.csv'
    ]
    
    existing_files = [f for f in CSV_FILES if os.path.exists(f)]
    if not existing_files:
        print("❌ No CSV files found!")
        return
    
    print(f"📁 Found {len(existing_files)} CSV files")
    
    predictor = FineTunedCryptoPricePredictor(sequence_length=45, prediction_days=1)
    
    try:
        # Load and process data
        df = predictor.load_and_clean_data(existing_files)
        if df is None:
            print("❌ Failed to load data!")
            return
            
        df_enhanced = predictor.create_advanced_features(df)
        if df_enhanced is None:
            print("❌ Failed to create features!")
            return
            
        X, y = predictor.prepare_advanced_training_data(df_enhanced)
        if X is None or y is None:
            print("❌ Failed to prepare training data!")
            return
        
        # Train/test split
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        print(f"   Training set: {X_train.shape[0]} sequences")
        print(f"   Test set: {X_test.shape[0]} sequences")
        
        # Train advanced model
        history = predictor.train_advanced_model(X_train, y_train, epochs=200)
        if history is None:
            print("❌ Training failed!")
            return
        
        # Evaluate
        test_metrics, predictions, y_actual = predictor.evaluate_advanced_model(X_test, y_test)
        if test_metrics is None:
            print("❌ Evaluation failed!")
            return
        
        # Save model
        predictor.save_advanced_model('trained_model_advanced')
        
        print("\n🎉 ADVANCED Training completed!")
        print(f"   🎯 Directional Accuracy: {test_metrics['directional_accuracy']:.2f}%")
        print(f"   📈 Trend Accuracy: {test_metrics['trend_accuracy']:.2f}%")
        print(f"   💰 Trading Return: {test_metrics['total_return']:.2f}%")
        print(f"   📊 Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()