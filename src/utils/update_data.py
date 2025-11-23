#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to update cryptocurrency data from Yahoo Finance
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul 2>&1')  # Set console to UTF-8
    except:
        pass

from src.utils.yahoo_finance_updater import YahooFinanceUpdater

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
    print("   - Train with new data: python -m src.crypto_training_script_improved")
    print("   - Make predictions: python -m src.crypto_predictor_improved")
