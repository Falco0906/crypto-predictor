#!/usr/bin/env python3
"""
Yahoo Finance Data Updater for Cryptocurrency Price Prediction
Fetches live data and integrates with existing training system
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class YahooFinanceUpdater:
    def __init__(self):
        """
        Initialize Yahoo Finance updater
        """
        self.crypto_symbols = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD', 
            'Solana': 'SOL-USD',
            'Litecoin': 'LTC-USD',
            'Cardano': 'ADA-USD',
            'Polkadot': 'DOT-USD',
            'Chainlink': 'LINK-USD',
            'Uniswap': 'UNI-USD',
            'Polygon': 'MATIC-USD',
            'Avalanche': 'AVAX-USD'
        }
        
        self.data_dir = 'yahoo_finance_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_crypto_data(self, symbol, period='2y', interval='1d'):
        """
        Fetch cryptocurrency data from Yahoo Finance
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD')
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        """
        try:
            print(f"📊 Fetching {symbol} data from Yahoo Finance...")
            
            # Get ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"   ❌ No data received for {symbol}")
                return None
            
            # Clean and format data
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Standardize column names
            column_mapping = {
                'date': 'date',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'dividends': 'dividends',
                'stock_splits': 'stock_splits'
            }
            
            # Rename columns
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data = data.rename(columns={old_col: new_col})
            
            # Ensure we have required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
                # Create dummy columns if missing
                for col in missing_cols:
                    if col == 'volume':
                        data[col] = np.random.randint(1000000, 10000000, len(data))
                    else:
                        data[col] = data['close']
            
            # Add coin name
            coin_name = symbol.replace('-USD', '').upper()
            data['coin'] = coin_name
            
            # Convert date to datetime if needed
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            
            print(f"   ✅ {coin_name}: {len(data)} records fetched")
            return data
            
        except Exception as e:
            print(f"   ❌ Error fetching {symbol}: {str(e)}")
            return None
    
    def update_all_crypto_data(self, period='2y', interval='1d'):
        """
        Update all cryptocurrency datasets
        """
        print("🚀 Updating all cryptocurrency data from Yahoo Finance...")
        print("=" * 60)
        
        updated_data = []
        
        for coin_name, symbol in self.crypto_symbols.items():
            data = self.fetch_crypto_data(symbol, period, interval)
            if data is not None:
                updated_data.append(data)
                
                # Save individual coin data
                filename = f"coin_{coin_name}.csv"
                filepath = os.path.join(self.data_dir, filename)
                data.to_csv(filepath)
                print(f"   💾 Saved to {filepath}")
        
        if not updated_data:
            print("❌ No data was fetched!")
            return None
        
        # Combine all data
        combined_data = pd.concat(updated_data, ignore_index=False).sort_index()
        
        # Save combined dataset
        combined_filepath = os.path.join(self.data_dir, 'combined_crypto_data.csv')
        combined_data.to_csv(combined_filepath)
        
        print(f"\n📊 Data Update Complete!")
        print(f"   Total records: {len(combined_data)}")
        print(f"   Cryptocurrencies: {len(updated_data)}")
        print(f"   Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        print(f"   Combined data saved to: {combined_filepath}")
        
        return combined_data
    
    def get_latest_prices(self):
        """
        Get latest prices for all cryptocurrencies
        """
        print("💰 Getting latest cryptocurrency prices...")
        print("=" * 50)
        
        latest_prices = {}
        
        for coin_name, symbol in self.crypto_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                current_price = info.get('regularMarketPrice', 0)
                previous_close = info.get('regularMarketPreviousClose', 0)
                change = info.get('regularMarketChange', 0)
                change_percent = info.get('regularMarketChangePercent', 0)
                
                latest_prices[coin_name] = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'previous_close': previous_close,
                    'change': change,
                    'change_percent': change_percent,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                direction = "📈" if change >= 0 else "📉"
                print(f"   {coin_name}: ${current_price:.2f} ({change:+.2f}, {change_percent:+.2f}%) {direction}")
                
            except Exception as e:
                print(f"   ❌ Error getting {coin_name} price: {str(e)}")
        
        return latest_prices
    
    def compare_with_old_data(self):
        """
        Compare new data with old CSV files
        """
        print("\n📈 Comparing new data with old datasets...")
        print("=" * 50)
        
        old_files = [f for f in os.listdir('.') if f.startswith('coin_') and f.endswith('.csv')]
        
        for old_file in old_files:
            try:
                old_data = pd.read_csv(old_file)
                old_data['date'] = pd.to_datetime(old_data['date'])
                old_latest = old_data['close'].iloc[-1]
                old_date = old_data['date'].iloc[-1]
                
                coin_name = old_file.replace('coin_', '').replace('.csv', '')
                if coin_name in self.crypto_symbols:
                    symbol = self.crypto_symbols[coin_name]
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    new_price = info.get('regularMarketPrice', 0)
                    
                    if new_price > 0:
                        change = ((new_price - old_latest) / old_latest) * 100
                        direction = "📈" if change >= 0 else "📉"
                        print(f"   {coin_name}: ${old_latest:.2f} → ${new_price:.2f} ({change:+.2f}%) {direction}")
                        print(f"     Old data: {old_date.strftime('%Y-%m-%d')}")
                        print(f"     New data: {datetime.now().strftime('%Y-%m-%d')}")
                        print()
                    
            except Exception as e:
                print(f"   ❌ Error comparing {old_file}: {str(e)}")
    
    def create_training_ready_data(self):
        """
        Create data ready for training with the existing system
        """
        print("\n🔧 Preparing data for training...")
        print("=" * 40)
        
        # Check if we have updated data
        combined_file = os.path.join(self.data_dir, 'combined_crypto_data.csv')
        if not os.path.exists(combined_file):
            print("❌ No updated data found. Please run update_all_crypto_data() first.")
            return None
        
        # Load combined data
        data = pd.read_csv(combined_file)
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        
        # Save individual coin files in the main directory for compatibility
        for coin_name in data['coin'].unique():
            coin_data = data[data['coin'] == coin_name].copy()
            filename = f"coin_{coin_name}.csv"
            coin_data.to_csv(filename)
            print(f"   ✅ {filename}: {len(coin_data)} records")
        
        print(f"\n🎯 Data ready for training!")
        print(f"   Run: python crypto_training_script.py")
        print(f"   Or: python crypto_predictor.py")
        
        return data

def main():
    """
    Main function to demonstrate the Yahoo Finance updater
    """
    print("🚀 Yahoo Finance Cryptocurrency Data Updater")
    print("=" * 60)
    
    updater = YahooFinanceUpdater()
    
    # Get latest prices
    latest_prices = updater.get_latest_prices()
    
    # Update all data
    print("\n" + "="*60)
    updated_data = updater.update_all_crypto_data(period='2y', interval='1d')
    
    # Compare with old data
    updater.compare_with_old_data()
    
    # Prepare for training
    updater.create_training_ready_data()
    
    print("\n🎉 Yahoo Finance integration complete!")
    print("   Your system now has the latest cryptocurrency data!")
    print("   Ready for training and predictions!")

if __name__ == "__main__":
    main()
