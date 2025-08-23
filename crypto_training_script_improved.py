#!/usr/bin/env python3
"""
IMPROVED Cryptocurrency Price Prediction Model
Fixes scaling issues and predicts percentage changes instead of absolute prices
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class ImprovedCryptoPricePredictor:
    def __init__(self, sequence_length=30, prediction_days=1):
        """
        Improved predictor that predicts percentage changes
        """
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.model = None
        self.price_scaler = StandardScaler()  # Better for percentage changes
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'price_change_pct'  # Predict percentage change, not absolute price
        
    def load_and_clean_data(self, file_paths):
        """Enhanced data cleaning with coin-specific handling"""
        print("📊 Loading data with improved cleaning...")
        
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
                
                # Basic data quality checks
                df = df[df['close'] > 0]
                df = df.dropna(subset=['close'])
                
                # Remove extreme outliers (keep 99% of data)
                price_q99 = df['close'].quantile(0.99)
                price_q01 = df['close'].quantile(0.01)
                df = df[(df['close'] >= price_q01) & (df['close'] <= price_q99)]
                
                # Ensure minimum data points
                if len(df) < 200:  # Increased minimum
                    print(f"   ⚠️  Skipping {file_path}: insufficient data ({len(df)} records)")
                    continue
                
                coin_name = os.path.basename(file_path).replace('.csv', '').replace('coin_', '')
                df['coin'] = coin_name.upper()
                
                # Calculate percentage change as target (more stable than absolute prices)
                df['price_change_pct'] = df['close'].pct_change() * 100
                
                # Remove first row (NaN from pct_change)
                df = df.iloc[1:]
                
                # Remove extreme percentage changes (>50% in one day is usually an error)
                df = df[abs(df['price_change_pct']) <= 50]
                
                all_data.append(df)
                print(f"   ✓ Loaded {len(df)} records for {coin_name}")
                
            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {str(e)}")
                continue
        
        if not all_data:
            print("❌ No data loaded!")
            return None
            
        final_df = pd.concat(all_data, ignore_index=False)
        final_df = final_df.sort_index()
        
        print(f"   ✓ Total data loaded: {len(final_df)} records across {len(all_data)} coins")
        return final_df
    
    def create_improved_features(self, df):
        """Create features optimized for percentage change prediction"""
        print("🔧 Creating improved features...")
        
        enhanced_data = []
        
        for coin in df['coin'].unique():
            coin_df = df[df['coin'] == coin].copy()
            
            # === PRICE-BASED FEATURES (normalized by price) ===
            # Use percentage changes instead of absolute differences
            coin_df['returns_1d'] = coin_df['close'].pct_change()
            coin_df['returns_3d'] = coin_df['close'].pct_change(3)
            coin_df['returns_7d'] = coin_df['close'].pct_change(7)
            coin_df['returns_14d'] = coin_df['close'].pct_change(14)
            
            # Volatility (rolling standard deviation of returns)
            for window in [5, 10, 20]:
                coin_df[f'volatility_{window}d'] = coin_df['returns_1d'].rolling(window=window).std()
            
            # Moving averages (as percentage from current price)
            for window in [5, 10, 20, 50]:
                sma = coin_df['close'].rolling(window=window).mean()
                coin_df[f'sma_{window}d_pct'] = (coin_df['close'] - sma) / sma * 100
                
                ema = coin_df['close'].ewm(span=window).mean()
                coin_df[f'ema_{window}d_pct'] = (coin_df['close'] - ema) / ema * 100
            
            # RSI (more stable calculation)
            delta = coin_df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            coin_df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD components
            ema12 = coin_df['close'].ewm(span=12).mean()
            ema26 = coin_df['close'].ewm(span=26).mean()
            coin_df['macd'] = (ema12 - ema26) / coin_df['close'] * 100  # Normalized
            coin_df['macd_signal'] = coin_df['macd'].ewm(span=9).mean()
            coin_df['macd_histogram'] = coin_df['macd'] - coin_df['macd_signal']
            
            # Bollinger Bands
            for window in [20]:
                sma = coin_df['close'].rolling(window=window).mean()
                std = coin_df['close'].rolling(window=window).std()
                coin_df[f'bb_upper_{window}'] = (sma + 2*std - coin_df['close']) / coin_df['close'] * 100
                coin_df[f'bb_lower_{window}'] = (coin_df['close'] - (sma - 2*std)) / coin_df['close'] * 100
                coin_df[f'bb_width_{window}'] = (2*std) / sma * 100
            
            # Momentum indicators (normalized)
            for period in [5, 10, 20]:
                coin_df[f'momentum_{period}d'] = (coin_df['close'] - coin_df['close'].shift(period)) / coin_df['close'].shift(period) * 100
            
            # Support and resistance levels (simplified)
            for window in [20]:
                high = coin_df['close'].rolling(window=window).max()
                low = coin_df['close'].rolling(window=window).min()
                coin_df[f'resistance_{window}d_pct'] = (high - coin_df['close']) / coin_df['close'] * 100
                coin_df[f'support_{window}d_pct'] = (coin_df['close'] - low) / coin_df['close'] * 100
            
            # Volume features (if available)
            if 'volume' in coin_df.columns:
                # Volume moving average
                coin_df['volume_sma_20'] = coin_df['volume'].rolling(window=20).mean()
                coin_df['volume_ratio'] = coin_df['volume'] / coin_df['volume_sma_20']
                
                # On Balance Volume (simplified)
                coin_df['obv'] = (coin_df['volume'] * coin_df['returns_1d'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
                coin_df['obv_ema'] = coin_df['obv'].ewm(span=20).mean()
                coin_df['obv_divergence'] = (coin_df['obv'] - coin_df['obv_ema']) / coin_df['obv_ema'] * 100
            
            # Time-based features
            if hasattr(coin_df.index, 'dayofweek'):
                coin_df['day_of_week'] = coin_df.index.dayofweek
                coin_df['month'] = coin_df.index.month
                coin_df['is_weekend'] = (coin_df['day_of_week'] >= 5).astype(int)
            
            # Lagged features (previous price changes)
            for lag in [1, 2, 3, 5]:
                coin_df[f'lag_{lag}d_change'] = coin_df['price_change_pct'].shift(lag)
            
            enhanced_data.append(coin_df)
        
        if not enhanced_data:
            print("❌ No enhanced data created!")
            return None
            
        final_df = pd.concat(enhanced_data, ignore_index=False)
        final_df = final_df.dropna()
        
        print(f"   ✓ Created {final_df.shape[1]} features for {len(final_df)} records")
        return final_df
    
    def prepare_training_data(self, df):
        """Prepare training data with proper scaling"""
        print("📦 Preparing training data...")
        
        if df is None or len(df) == 0:
            print("❌ No data to prepare!")
            return None, None
        
        # Feature selection - exclude target and metadata
        exclude_cols = ['price_change_pct', 'coin', 'close']
        feature_cols = [col for col in df.columns if col not in exclude_cols and 
                       df[col].dtype in ['float64', 'int64'] and not df[col].isna().all()]
        
        # Remove features with very low variance
        feature_variances = df[feature_cols].var()
        feature_cols = feature_variances[feature_variances > 1e-8].index.tolist()
        
        if len(feature_cols) == 0:
            print("❌ No valid features found!")
            return None, None
        
        self.feature_columns = feature_cols
        print(f"   Selected {len(feature_cols)} features after filtering")
        
        # Scale features and target
        features_scaled = self.feature_scaler.fit_transform(df[feature_cols])
        target_scaled = self.price_scaler.fit_transform(df[['price_change_pct']])
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        for coin in df['coin'].unique():
            coin_mask = df['coin'] == coin
            coin_features = features_scaled[coin_mask]
            coin_targets = target_scaled[coin_mask].flatten()
            
            # Create sequences without overlap to prevent overfitting
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
    
    def build_improved_model(self, input_shape):
        """Build a simpler, more stable model"""
        print("🏗️ Building improved model...")
        
        model = Sequential([
            # First LSTM layer
            LSTM(64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer - predict percentage change
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers
            metrics=['mae', 'mse']
        )
        
        print(f"   ✓ Model built with {model.count_params():,} parameters")
        return model
    
    def train_improved_model(self, X, y, validation_split=0.2):
        """Train the improved model"""
        print("🚀 Training improved model...")
        
        if self.model is None:
            self.model = self.build_improved_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint('best_improved_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        self.training_history = history.history
        print("✅ Improved training completed!")
        return history
    
    def create_training_visualizations(self, save_dir='training_visualizations'):
        """Create comprehensive training visualizations"""
        print("📊 Creating training visualizations...")
        
        if not self.training_history:
            print("❌ No training history available!")
            return
        
        # Create directory for visualizations
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cryptocurrency Price Prediction Model Training Results', fontsize=16, fontweight='bold')
        
        # 1. Loss curves
        axes[0, 0].plot(self.training_history['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')
        axes[0, 0].set_title('Model Loss Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MAE curves
        axes[0, 1].plot(self.training_history['mae'], label='Training MAE', linewidth=2, color='#F18F01')
        axes[0, 1].plot(self.training_history['val_mae'], label='Validation MAE', linewidth=2, color='#C73E1D')
        axes[0, 1].set_title('Mean Absolute Error Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MSE curves
        axes[0, 2].plot(self.training_history['mse'], label='Training MSE', linewidth=2, color='#6B5B95')
        axes[0, 2].plot(self.training_history['val_mse'], label='Validation MSE', linewidth=2, color='#F7CAC9')
        axes[0, 2].set_title('Mean Squared Error Over Time', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MSE')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Learning rate over time
        if 'lr' in self.training_history:
            axes[1, 0].plot(self.training_history['lr'], linewidth=2, color='#88D8C0')
            axes[1, 0].set_title('Learning Rate Over Time', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learning Rate Over Time', fontweight='bold')
        
        # 5. Loss comparison (final epoch)
        final_train_loss = self.training_history['loss'][-1]
        final_val_loss = self.training_history['val_loss'][-1]
        
        bars = axes[1, 1].bar(['Training', 'Validation'], [final_train_loss, final_val_loss], 
                              color=['#2E86AB', '#A23B72'], alpha=0.8)
        axes[1, 1].set_title('Final Loss Comparison', fontweight='bold')
        axes[1, 1].set_ylabel('Loss')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Training summary metrics
        summary_text = f"""
Training Summary:
• Total Epochs: {len(self.training_history['loss'])}
• Final Training Loss: {final_train_loss:.4f}
• Final Validation Loss: {final_val_loss:.4f}
• Final Training MAE: {self.training_history['mae'][-1]:.2f}%
• Final Validation MAE: {self.training_history['val_mae'][-1]:.2f}%
• Best Validation Loss: {min(self.training_history['val_loss']):.4f}
• Overfitting: {'Yes' if final_val_loss > final_train_loss * 1.1 else 'No'}
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1, 2].set_title('Training Summary', fontweight='bold')
        axes[1, 2].axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save high-resolution plot
        plot_path = os.path.join(save_dir, 'training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📈 Training visualization saved to: {plot_path}")
        
        # Create additional detailed plots
        self._create_detailed_plots(save_dir)
        
        plt.show()
    
    def _create_detailed_plots(self, save_dir):
        """Create additional detailed training plots"""
        
        # 1. Loss convergence analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss convergence
        epochs = range(1, len(self.training_history['loss']) + 1)
        ax1.plot(epochs, self.training_history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Convergence Analysis', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss ratio (validation/training)
        loss_ratio = [v/t for v, t in zip(self.training_history['val_loss'], self.training_history['loss'])]
        ax2.plot(epochs, loss_ratio, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Equal Loss')
        ax2.axhline(y=1.1, color='orange', linestyle='--', alpha=0.7, label='10% Overfitting Threshold')
        ax2.set_title('Loss Ratio (Validation/Training)', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = os.path.join(save_dir, 'loss_convergence_analysis.png')
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📊 Loss convergence analysis saved to: {convergence_path}")
        
        # 2. Metrics correlation plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create correlation matrix
        metrics_df = pd.DataFrame({
            'Training_Loss': self.training_history['loss'],
            'Validation_Loss': self.training_history['val_loss'],
            'Training_MAE': self.training_history['mae'],
            'Validation_MAE': self.training_history['val_mae'],
            'Training_MSE': self.training_history['mse'],
            'Validation_MSE': self.training_history['val_mse']
        })
        
        correlation_matrix = metrics_df.corr()
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.columns)
        ax.set_title('Training Metrics Correlation Matrix', fontweight='bold', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
        
        plt.tight_layout()
        correlation_path = os.path.join(save_dir, 'metrics_correlation.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   🔗 Metrics correlation matrix saved to: {correlation_path}")
        
        # 3. Training progress summary
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a comprehensive summary
        summary_data = {
            'Metric': ['Training Loss', 'Validation Loss', 'Training MAE', 'Validation MAE'],
            'Start': [
                self.training_history['loss'][0],
                self.training_history['val_loss'][0],
                self.training_history['mae'][0],
                self.training_history['val_mae'][0]
            ],
            'End': [
                self.training_history['loss'][-1],
                self.training_history['val_loss'][-1],
                self.training_history['mae'][-1],
                self.training_history['val_mae'][-1]
            ],
            'Improvement': [
                ((self.training_history['loss'][0] - self.training_history['loss'][-1]) / self.training_history['loss'][0]) * 100,
                ((self.training_history['val_loss'][0] - self.training_history['val_loss'][-1]) / self.training_history['val_loss'][0]) * 100,
                ((self.training_history['mae'][0] - self.training_history['mae'][-1]) / self.training_history['mae'][0]) * 100,
                ((self.training_history['val_mae'][0] - self.training_history['val_mae'][-1]) / self.training_history['val_mae'][0]) * 100
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create bar plot
        x = np.arange(len(summary_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, summary_df['Start'], width, label='Start', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, summary_df['End'], width, label='End', color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Training Progress Summary', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['Metric'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add improvement percentages
        for i, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, summary_df['Improvement'])):
            ax.text(i, max(bar1.get_height(), bar2.get_height()) + max(bar1.get_height(), bar2.get_height()) * 0.05,
                   f'{improvement:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        summary_path = os.path.join(save_dir, 'training_progress_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📋 Training progress summary saved to: {summary_path}")
        
        # Save training data as CSV for further analysis
        training_data_path = os.path.join(save_dir, 'training_history.csv')
        training_df = pd.DataFrame(self.training_history)
        training_df.to_csv(training_data_path, index=False)
        print(f"   💾 Training history data saved to: {training_data_path}")
        
        plt.show()
    
    def evaluate_improved_model(self, X, y):
        """Evaluate with focus on percentage change accuracy"""
        print("📈 Evaluating improved model...")
        
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
        
        # Directional accuracy (more important for trading)
        y_direction = y_actual > 0
        pred_direction = predictions > 0
        directional_accuracy = np.mean(y_direction == pred_direction) * 100
        
        # Trend accuracy (3-day trends)
        y_trend = np.diff(y_actual, n=3) > 0
        pred_trend = np.diff(predictions, n=3) > 0
        trend_accuracy = np.mean(y_trend == pred_trend) * 100
        
        # Percentage accuracy (how close predictions are to actual changes)
        pct_accuracy = np.mean(np.abs(predictions - y_actual) <= 1.0) * 100  # Within 1%
        
        # Trading simulation
        # Buy if predicted change > 0.5%, sell if < -0.5%
        trading_signals = np.where(predictions > 0.5, 1, np.where(predictions < -0.5, -1, 0))
        strategy_returns = y_actual * trading_signals / 100  # Convert to decimal
        total_return = np.prod(1 + strategy_returns) - 1
        
        # Sharpe ratio
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'trend_accuracy': trend_accuracy,
            'pct_accuracy': pct_accuracy,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio
        }
        
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.2f}%")
        print(f"   RMSE: {rmse:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"   Trend Accuracy: {trend_accuracy:.2f}%")
        print(f"   Within 1% Accuracy: {pct_accuracy:.2f}%")
        print(f"   Trading Return: {total_return*100:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        
        return metrics, predictions, y_actual
    
    def save_improved_model(self, model_dir='trained_model_improved'):
        """Save the improved model"""
        print("💾 Saving improved model...")
        
        if self.model is None:
            print("❌ No model to save!")
            return
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.model.save(os.path.join(model_dir, 'crypto_improved_model.h5'))
        joblib.dump(self.price_scaler, os.path.join(model_dir, 'price_scaler.pkl'))
        joblib.dump(self.feature_scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
        
        metadata = {
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': 'Improved_LSTM_Percentage_Change',
            'training_date': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'num_features': len(self.feature_columns),
            'architecture': 'LSTM + Percentage Change Prediction'
        }
        
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Improved model saved to {model_dir}/")

def main():
    """Main training function"""
    print("🚀 IMPROVED Cryptocurrency Price Predictor Training")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ImprovedCryptoPricePredictor(sequence_length=30)
    
    # Load and clean data
    csv_files = [
        'coin_BTC.csv', 'coin_ETH.csv', 'coin_SOL.csv', 'coin_LTC.csv', 
        'coin_ADA.csv', 'coin_DOT.csv', 'coin_LINK.csv', 'coin_UNI.csv', 
        'coin_MATIC.csv', 'coin_AVAX.csv'
    ]
    
    # Filter to existing files
    existing_files = [f for f in csv_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No CSV files found!")
        return
    
    print(f"📁 Found {len(existing_files)} data files")
    
    # Load and clean data
    data = predictor.load_and_clean_data(existing_files)
    if data is None:
        return
    
    # Create features
    enhanced_data = predictor.create_improved_features(data)
    if enhanced_data is None:
        return
    
    # Prepare training data
    X, y = predictor.prepare_training_data(enhanced_data)
    if X is None or y is None:
        return
    
    # Train model
    history = predictor.train_improved_model(X, y)
    
    # Create comprehensive training visualizations
    predictor.create_training_visualizations()
    
    # Evaluate model
    metrics, predictions, actuals = predictor.evaluate_improved_model(X, y)
    
    # Save model
    predictor.save_improved_model()
    
    print("\n🎉 IMPROVED Training completed!")
    print(f"   🎯 Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    print(f"   📈 Trend Accuracy: {metrics['trend_accuracy']:.2f}%")
    print(f"   💰 Trading Return: {metrics['total_return']:.2f}%")
    print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    
    print("\n📊 Training visualizations have been saved to 'training_visualizations/' folder!")
    print("   📈 training_results.png - Main training overview")
    print("   📊 loss_convergence_analysis.png - Loss analysis")
    print("   🔗 metrics_correlation.png - Metrics correlation")
    print("   📋 training_progress_summary.png - Progress summary")
    print("   💾 training_history.csv - Raw training data")

if __name__ == "__main__":
    main()
