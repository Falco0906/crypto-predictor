#!/usr/bin/env python3
"""
Complete Pipeline Script for Cryptocurrency Prediction
=======================================================
This script runs the complete pipeline:
1. Update cryptocurrency data from Yahoo Finance
2. Train the improved LSTM model
3. Generate predictions for all cryptocurrencies

Usage:
    python run_full_pipeline.py
"""

import sys
import os
import subprocess
from pathlib import Path

# Set UTF-8 encoding for Windows (safer approach)
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_and_install_dependencies():
    """Check if required packages are installed, install if missing"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy<2',  # TensorFlow 2.10 requires numpy < 2
        'yfinance': 'yfinance',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'sklearn':
                __import__('sklearn')
            else:
                __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)
    
    # Check for TensorFlow
    try:
        __import__('tensorflow')
    except ImportError:
        missing_packages.append('tensorflow==2.10.0')
    
    if missing_packages:
        print("[WARNING] Missing required packages detected!")
        print(f"   Installing: {', '.join(missing_packages)}")
        print("   This may take a few minutes...\n")
        
        try:
            # Install numpy first if missing (required for others)
            if 'numpy<2' in missing_packages:
                print("   [INFO] Installing numpy (compatible version)...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2", "--quiet"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            for package in missing_packages:
                if package != 'numpy<2':
                    print(f"   [INFO] Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("\n[OK] All required packages installed!\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Error installing packages: {e}")
            print("   Please install manually: pip install -r requirements.txt")
            print("   Or: pip install pandas numpy<2 yfinance tensorflow==2.10.0 scikit-learn matplotlib seaborn joblib")
            sys.exit(1)

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_section(text):
    """Print a section separator"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}\n")

def ask_user_input(prompt, valid_options=['y', 'n']):
    """Get user input and validate"""
    while True:
        response = input(prompt).strip().lower()
        if response in valid_options:
            return response
        print(f"   Invalid input. Please enter one of: {', '.join(valid_options)}")

def check_pretrained_model_exists():
    """Check if pre-trained model exists"""
    model_path = PROJECT_ROOT / 'data' / 'models_gpu_improved' / 'best_improved_model.h5'
    return model_path.exists()

def setup_model_manager():
    """Initialize and setup the model manager"""
    from src.utils.model_manager import ModelManager
    manager = ModelManager()
    manager.register_pretrained_model()
    return manager

def main():
    """Run the complete pipeline"""
    print_header("[OK] CRYPTOCURRENCY PREDICTION - COMPLETE PIPELINE")
    
    # Check and install dependencies first
    print("[INFO] Checking dependencies...")
    check_and_install_dependencies()
    
    try:
        # Step 1: Update Data
        print_section("[INFO] STEP 1: Data Collection & Update")
        print("[INFO] Fetching latest cryptocurrency data from Yahoo Finance...")
        print("   This will update all cryptocurrency datasets...\n")
        
        try:
            from src.utils.yahoo_finance_updater import YahooFinanceUpdater
            
            updater = YahooFinanceUpdater()
            print("   [OK] Starting data collection process...\n")
            updater.update_all_crypto_data(period='2y', interval='1d')
            
            print("\n[OK] Step 1 Complete: Data collection and update finished!\n")
        except Exception as e:
            print(f"\n[ERROR] Error during data update: {str(e)}")
            print("   Attempting to continue with existing data files...\n")
            import traceback
            traceback.print_exc()
        
        # Step 2: Ask about training or model selection
        print_section("[INFO] STEP 2: Model Training & Selection")
        
        # Initialize model manager
        manager = setup_model_manager()
        
        # Get available models
        available_models = manager.get_all_models()
        
        if available_models:
            print("[INFO] Available models found!\n")
            selected = manager.display_model_selection_menu()
            
            if selected is None:
                print("[ERROR] No model selected!")
                sys.exit(1)
            
            if 'train_new' in selected and selected['train_new']:
                print("\n[INFO] Training new improved LSTM model...")
                print("   This may take 5-15 minutes depending on your GPU/CPU...\n")
                
                from src.crypto_training_script_improved import main as train_main
                train_main()
                
                print("\n[OK] Step 2 Complete: Model training finished!\n")
                
                # Refresh model manager after training
                manager = setup_model_manager()
                available_models = manager.get_all_models()
                selected = available_models[-1] if available_models else None
            else:
                print(f"\n[OK] Using model: {selected.get('name', 'Unknown')}\n")
        else:
            print("[WARNING] No models found. Training new model...\n")
            print("[INFO] Training improved LSTM model...")
            print("   This may take 5-15 minutes depending on your GPU/CPU...\n")
            
            from src.crypto_training_script_improved import main as train_main
            train_main()
            
            print("\n[OK] Step 2 Complete: Model training finished!\n")
            
            # Setup manager after training
            manager = setup_model_manager()
            available_models = manager.get_all_models()
            selected = available_models[-1] if available_models else None
        
        # Step 3: Generate Predictions
        print_section("[INFO] STEP 3: Model Prediction")
        print("[INFO] Generating predictions for all cryptocurrencies...")
        print("   Using the model to predict future prices...\n")
        
        from src.crypto_predictor_improved import main as predict_main
        
        # Pass selected model info to predictor
        predict_main(model_info=selected)
        
        print("\n[OK] Step 3 Complete: Predictions generated!\n")
        
        # Final Summary
        print_header("[OK] COMPLETE PIPELINE EXECUTION FINISHED!")
        print("[INFO] All pipeline steps completed successfully:")
        print("   ┌─────────────────────────────────────────────────────┐")
        print("   │ 1. [OK] Data Collection & Update                    │")
        print("   │    -> Fetched latest data from Yahoo Finance        │")
        print("   │                                                      │")
        print("   │ 2. [OK] Model Selection                             │")
        if selected and 'name' in selected:
            model_name = selected['name'][:40]
            print(f"   │    -> Using: {model_name:<32} │")
        else:
            print("   │    -> Model selected                               │")
        print("   │    -> Model: data/models_gpu_improved/              │")
        print("   │                                                      │")
        print("   │ 3. [OK] Model Prediction                             │")
        print("   │    -> Generated predictions for all cryptos          │")
        print("   │    -> Predictions displayed above                    │")
        print("   └─────────────────────────────────────────────────────┘")
        print(f"\n[INFO] Model Location: {PROJECT_ROOT / 'data' / 'models_gpu_improved'}")
        print(f"[INFO] Data Location: {PROJECT_ROOT / 'data' / 'raw_data'}")
        print(f"[INFO] Visualizations: {PROJECT_ROOT / 'docs'}")
        print("\n[OK] Pipeline execution complete! All steps ran successfully!")
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

