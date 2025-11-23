# Architecture Diagram - Multi-Model System

## System Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER RUNS: run_full_pipeline.py              │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  STEP 1: Data Collection                 │
        │  ├─ Check dependencies                   │
        │  ├─ Download crypto data (Yahoo Finance) │
        │  ├─ 9 cryptocurrencies                   │
        │  ├─ 6K+ records, 2 years                 │
        │  └─ Save to data/raw_data/               │
        └────────────────────┬─────────────────────┘
                             │
        ┌────────────────────▼──────────────────────────┐
        │  STEP 2: Initialize Model Manager             │
        │  ├─ Create ModelManager instance              │
        │  ├─ Auto-register pre-trained model           │
        │  ├─ Load model_registry.json                  │
        │  └─ Scan for user-trained models              │
        └────────────────────┬──────────────────────────┘
                             │
        ┌────────────────────▼──────────────────────────┐
        │  STEP 3: Display Model Selection Menu          │
        │  ├─ List all available models                 │
        │  ├─ Show accuracy metrics                     │
        │  ├─ Show creation dates                       │
        │  └─ Wait for user input (1, 2, 3, etc.)       │
        └────────────────────┬──────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼─────┐   ┌────▼────┐   ┌────▼──────┐
         │ User: 1  │   │ User: 2 │   │ User: 3   │
         │(Pre-tr)  │   │(User MD)│   │(Train NEW)│
         └────┬─────┘   └────┬────┘   └────┬──────┘
              │              │              │
         ┌────▼─────────┐    │    ┌─────────▼──────┐
         │ Selection: 1 │    │    │ Selection: 3   │
         └────┬─────────┘    │    └─────────┬──────┘
              │              │              │
              │    ┌─────────▼────────┐     │
              │    │ Selection: 2     │     │
              │    └────────┬─────────┘     │
              │             │               │
              │   ┌─────────┴────────┐      │
              │   │  get_model_files │      │
              │   │  Return:         │      │
              │   │  - model path    │      │
              │   │  - scaler paths  │      │
              │   └────────┬────────┘      │
              │            │               │
         ┌────▼────────────▼───┐      ┌────▼────────────┐
         │  Load Selected Model │      │  Train New Model│
         │  (Instant - 2 sec)   │      │  (40-50 sec GPU)│
         └────┬────────────────┘      └────┬──────┬─────┘
              │                            │      │
              │                       ┌────▼─┐  ┌▼────────┐
              │                       │Train │  │Register │
              │                       │Model │  │+ Version│
              │                       └────┬─┘  └┬────────┘
              │                            │     │
              │                       ┌────▼────▼────┐
              │                       │Save with:    │
              │                       │-Timestamp    │
              │                       │-Metrics      │
              │                       │-Registry     │
              │                       └────┬─────────┘
              │                            │
              ┌────────────────────────────┴──────┐
              │                                   │
        ┌─────▼────────────┐             ┌───────▼──────────┐
        │ STEP 4: Predict  │             │ STEP 4: Predict  │
        │ Using Model 1/2  │             │ Using New Model  │
        │ (Instant)        │             │ (Immediate)      │
        └─────┬────────────┘             └───────┬──────────┘
              │                                   │
              └────────────────┬──────────────────┘
                               │
        ┌──────────────────────▼─────────────────┐
        │  Generate Predictions                  │
        │  ├─ Process all 9 cryptos              │
        │  ├─ Calculate features                 │
        │  ├─ Run LSTM predictions               │
        │  ├─ Show current price                 │
        │  ├─ Show predicted change              │
        │  ├─ Show 3-day forecast                │
        │  └─ Display results                    │
        └──────────────────────┬──────────────────┘
                               │
        ┌──────────────────────▼──────────────────┐
        │  Pipeline Complete ✅                   │
        │  - Data: Updated                        │
        │  - Model: Selected/Trained              │
        │  - Predictions: Generated               │
        └───────────────────────────────────────┘
```

## Model Manager Internal Flow

```
┌─────────────────────────────────────────┐
│        ModelManager.__init__()            │
│  ├─ Set models_dir                       │
│  ├─ Create directory if needed           │
│  ├─ Load model_registry.json             │
│  │  └─ If not exists: create empty       │
│  └─ Ready to manage models               │
└──────────────┬──────────────────────────┘
               │
       ┌───────▼────────────────────┐
       │ register_pretrained_model() │
       │ ├─ Check for:              │
       │ │  - best_improved_model.h5│
       │ │  - crypto_improved_model │
       │ ├─ If found:               │
       │ │  └─ Add to registry      │
       │ └─ Save registry.json      │
       └───────┬────────────────────┘
               │
       ┌───────▼──────────────────────┐
       │ register_user_model(metrics)  │
       │ ├─ Get timestamp             │
       │ ├─ Find best_improved_model  │
       │ ├─ Rename to:                │
       │ │  user_model_TIMESTAMP.h5   │
       │ ├─ Record metrics:           │
       │ │  - accuracy                │
       │ │  - training_date           │
       │ │  - model_type              │
       │ ├─ Add to registry           │
       │ └─ Save registry.json        │
       └───────┬──────────────────────┘
               │
       ┌───────▼──────────────────────┐
       │ get_all_models()              │
       │ ├─ Read registry.json         │
       │ ├─ Check each model file      │
       │ ├─ Verify files exist         │
       │ ├─ Get file sizes             │
       │ └─ Return models list         │
       └───────┬──────────────────────┘
               │
       ┌───────▼──────────────────────────┐
       │ display_model_selection_menu()    │
       │ ├─ For each model:               │
       │ │  ├─ Print name                 │
       │ │  ├─ Print type (pre/user)      │
       │ │  ├─ Print date                 │
       │ │  ├─ Print accuracy metrics     │
       │ │  └─ Add option number          │
       │ ├─ Add: "Train NEW"              │
       │ ├─ Get user input               │
       │ ├─ Validate selection           │
       │ └─ Return selected model info   │
       └───────┬──────────────────────────┘
               │
       ┌───────▼──────────────────────┐
       │ get_model_files(model_id)     │
       │ ├─ Look up model in registry  │
       │ ├─ Find .h5 or .keras file   │
       │ ├─ Locate scalers (.pkl)     │
       │ └─ Return paths              │
       └──────────────────────────────┘
```

## Data Flow: Training Path

```
┌─────────────────────────────────────────┐
│  Training Script: main()                 │
└────────────┬────────────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Load & Clean Data           │
    │  ├─ 9 CSV files              │
    │  ├─ 6,188 records            │
    │  └─ Remove nulls/duplicates  │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Create Features             │
    │  ├─ 48 initial indicators    │
    │  ├─ Select top 40            │
    │  └─ Normalize (scalers)      │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Create Sequences            │
    │  ├─ Window size: 30 days     │
    │  ├─ Total sequences: 5,477   │
    │  └─ Shape: (5477, 30, 40)    │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Train LSTM Model            │
    │  ├─ GPU: RTX 3050 (3.6GB)    │
    │  ├─ Epochs: up to 100        │
    │  ├─ Early stopping: 15 epoch │
    │  └─ Time: 40-50 seconds      │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Save Model                  │
    │  └─ Path: best_improved...h5 │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  REGISTER WITH MODEL MANAGER │
    │  ├─ Create timestamp         │
    │  ├─ Rename file:             │
    │  │  user_model_TIMESTAMP.h5  │
    │  ├─ Capture metrics:         │
    │  │  - directional_acc        │
    │  │  - trend_acc              │
    │  │  - mae                    │
    │  │  - sharpe_ratio           │
    │  ├─ Update registry.json     │
    │  └─ Mark as last_used        │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Evaluate Model              │
    │  └─ Generate metrics         │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Create Visualizations       │
    │  ├─ Training plots           │
    │  ├─ Loss curves              │
    │  └─ Save to docs/            │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Generate Predictions        │
    │  ├─ Load new trained model   │
    │  ├─ Process 9 cryptos        │
    │  └─ Show results             │
    └──────────────────────────────┘
```

## Model Registry Structure

```
model_registry.json
│
├── "models": {
│   │
│   ├── "pretrained": {
│   │   ├── "name": "Pre-trained Model"
│   │   ├── "filename": "crypto_improved_model.h5"
│   │   ├── "type": "pretrained"
│   │   ├── "created_date": "Included with repo"
│   │   ├── "accuracy": {
│   │   │   ├── "directional_accuracy": 56.97
│   │   │   ├── "trend_accuracy": 64.30
│   │   │   └── "mae": 2.97
│   │   └── "description": "Official pre-trained..."
│   │
│   ├── "user_20251123_130505": {
│   │   ├── "name": "User Model - 20251123_130505"
│   │   ├── "filename": "user_model_20251123_130505.h5"
│   │   ├── "type": "user_trained"
│   │   ├── "created_date": "2025-11-23T13:05:05"
│   │   ├── "accuracy": {
│   │   │   ├── "directional_accuracy": 56.98
│   │   │   ├── "trend_accuracy": 60.30
│   │   │   ├── "mae": 2.97
│   │   │   └── "sharpe_ratio": 2.47
│   │   └── "description": "Model trained on..."
│   │
│   └── "user_20251123_140000": { ... }  # More models
│
└── "last_used": "pretrained"
```

## Directory Tree with Multi-Model System

```
crypto-predictor/
│
├── data/
│   ├── models_gpu_improved/
│   │   ├── crypto_improved_model.h5           [INCLUDED: Pre-trained]
│   │   ├── user_model_20251123_130505.h5      [USER TRAINED v1]
│   │   ├── user_model_20251123_130505.keras   [v1 faster format]
│   │   ├── user_model_20251123_140000.h5      [USER TRAINED v2]
│   │   ├── price_scaler.pkl                   [Shared scaler]
│   │   ├── feature_scaler.pkl                 [Shared scaler]
│   │   ├── model_metadata.json                [Pre-trained info]
│   │   └── model_registry.json                [ALL models registry]
│   │
│   └── raw_data/
│       ├── coin_BTC.csv
│       ├── coin_ETH.csv
│       ├── ... (9 total)
│       └── combined_crypto_data.csv
│
├── src/
│   ├── utils/
│   │   ├── model_manager.py        [NEW: Core system]
│   │   └── ... (other utils)
│   │
│   ├── crypto_training_script_improved.py      [MODIFIED]
│   ├── crypto_predictor_improved.py            [MODIFIED]
│   └── ... (other modules)
│
├── run_full_pipeline.py                        [MODIFIED]
├── SYSTEM_COMPLETE.md                          [NEW]
├── MULTI_MODEL_SYSTEM.md                       [NEW]
├── IMPLEMENTATION_COMPLETE.md                  [NEW]
└── ... (other docs)
```

## State Diagram: Model Lifecycle

```
                    ┌──────────────────┐
                    │ Pre-trained Model │
                    │ (Included)        │
                    │ Status: Ready     │
                    └─────────┬─────────┘
                              │
                    [User selects option 1]
                              │
                    ┌─────────▼──────────┐
                    │ Load Pre-trained   │
                    │ Generate Predictions
                    └────────────────────┘
                              │
                    [User selects option 3]
                              │
                    ┌─────────▼──────────┐
                    │ Train New Model    │
                    │ Status: Training   │
                    └─────────┬──────────┘
                              │
                    [Training Complete]
                              │
                    ┌─────────▼────────────────┐
                    │ Rename: best_improved... │
                    │ TO: user_model_TIME.h5   │
                    │ Status: Renaming         │
                    └─────────┬────────────────┘
                              │
                    [Register in system]
                              │
                    ┌─────────▼────────────────┐
                    │ User Model v1            │
                    │ Status: Registered       │
                    │ In Registry: Yes         │
                    │ Available: In Menu       │
                    └─────────┬────────────────┘
                              │
        [Next run: User selects option 2]
                              │
                    ┌─────────▼──────────┐
                    │ Load User Model    │
                    │ Generate Predictions
                    └────────────────────┘
                              │
        [User selects option 3 again]
                              │
                    ┌─────────▼──────────┐
                    │ Train New Model    │
                    │ (Old one not lost!) │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼────────────────┐
                    │ User Model v2            │
                    │ Status: Registered       │
                    │ In Registry: Yes         │
                    │ Available: In Menu       │
                    └─────────┬────────────────┘
                              │
        [Next run: Menu shows both v1 & v2]
                              │
                    [User can select either!]
```

This architecture ensures:
- ✅ Pre-trained always available
- ✅ User models never deleted
- ✅ Multiple versions supported
- ✅ Easy comparison
- ✅ Full history preserved
- ✅ Automatic versioning
- ✅ Zero friction for users
