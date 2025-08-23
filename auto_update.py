#!/usr/bin/env python3
"""
Automated Daily Data Update Script
Run this script daily to keep your cryptocurrency data fresh
"""

import schedule
import time
from datetime import datetime
from yahoo_finance_updater import YahooFinanceUpdater

def daily_update():
    """
    Daily update function
    """
    print(f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Starting daily cryptocurrency data update...")
    
    try:
        updater = YahooFinanceUpdater()
        
        # Get latest prices
        latest_prices = updater.get_latest_prices()
        
        # Update historical data (last 2 years)
        updated_data = updater.update_all_crypto_data(period='2y', interval='1d')
        
        # Prepare for training
        updater.create_training_ready_data()
        
        print("✅ Daily update completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during daily update: {str(e)}")

def main():
    """
    Main function to set up automated updates
    """
    print("🤖 Cryptocurrency Data Auto-Updater")
    print("=" * 50)
    print("This script will update your data daily at 9:00 AM")
    print("Press Ctrl+C to stop the automation")
    print("=" * 50)
    
    # Schedule daily update at 9:00 AM
    schedule.every().day.at("09:00").do(daily_update)
    
    # Run initial update
    print("🔄 Running initial update...")
    daily_update()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Auto-updater stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
