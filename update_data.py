#!/usr/bin/env python3
"""
Simple script to update cryptocurrency data from Yahoo Finance
"""

from yahoo_finance_updater import YahooFinanceUpdater

if __name__ == "__main__":
    print("🚀 Starting Yahoo Finance Data Update...")
    
    # Create updater instance
    updater = YahooFinanceUpdater()
    
    # Get latest prices
    print("\n1️⃣ Getting latest prices...")
    latest_prices = updater.get_latest_prices()
    
    # Update all data
    print("\n2️⃣ Updating historical data...")
    updated_data = updater.update_all_crypto_data(period='2y', interval='1d')
    
    # Compare with old data
    print("\n3️⃣ Comparing with old data...")
    updater.compare_with_old_data()
    
    # Prepare for training
    print("\n4️⃣ Preparing data for training...")
    updater.create_training_ready_data()
    
    print("\n🎉 Data update complete!")
    print("   You can now:")
    print("   - Train with new data: python crypto_training_script.py")
    print("   - Make predictions: python crypto_predictor.py")
