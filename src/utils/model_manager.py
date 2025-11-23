"""
Model Manager - Handle multiple model versions and selection
============================================================
Manages pre-trained and user-trained models with versioning,
metadata tracking, and easy model selection.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import shutil


class ModelManager:
    """Manages multiple model versions and selections"""
    
    def __init__(self, models_dir=None):
        """Initialize model manager"""
        if models_dir is None:
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / 'data' / 'models_gpu_improved'
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / 'model_registry.json'
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from JSON"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'models': {},
                'last_used': None
            }
    
    def _save_registry(self):
        """Save model registry to JSON"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_pretrained_model(self):
        """Register the built-in pre-trained model"""
        # Try different possible names for pre-trained model
        pretrained_candidates = [
            self.models_dir / 'best_improved_model.h5',
            self.models_dir / 'crypto_improved_model.h5',
        ]
        
        pretrained_path = None
        pretrained_filename = None
        
        for candidate in pretrained_candidates:
            if candidate.exists():
                pretrained_path = candidate
                pretrained_filename = candidate.name
                break
        
        if not pretrained_path:
            return False
        
        # Check if already registered
        if 'pretrained' not in self.registry['models']:
            self.registry['models']['pretrained'] = {
                'name': 'Pre-trained Model',
                'filename': pretrained_filename,
                'type': 'pretrained',
                'created_date': 'Included with repository',
                'accuracy': {
                    'directional_accuracy': 56.97,
                    'trend_accuracy': 64.30,
                    'mae': 2.97
                },
                'description': 'Official pre-trained model included with the repository'
            }
            self._save_registry()
        
        return True
    
    def register_user_model(self, metrics=None):
        """Register a newly trained user model with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f'user_{timestamp}'
        
        # The training script saves as 'best_improved_model.h5'
        # We need to rename it to include timestamp
        current_model = self.models_dir / 'best_improved_model.h5'
        
        if current_model.exists():
            # Rename to versioned name
            new_filename = f'user_model_{timestamp}.h5'
            new_path = self.models_dir / new_filename
            shutil.move(str(current_model), str(new_path))
            
            # Also move the .keras file if it exists
            keras_current = self.models_dir / 'best_improved_model.keras'
            if keras_current.exists():
                keras_new = self.models_dir / f'user_model_{timestamp}.keras'
                shutil.move(str(keras_current), str(keras_new))
            
            # Register in registry
            model_info = {
                'name': f'User Model - {timestamp}',
                'filename': new_filename,
                'type': 'user_trained',
                'created_date': datetime.now().isoformat(),
                'accuracy': metrics or {},
                'description': f'Model trained on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            }
            
            self.registry['models'][model_id] = model_info
            self.registry['last_used'] = model_id
            self._save_registry()
            
            return model_id, model_info
        
        return None, None
    
    def get_all_models(self):
        """Get list of all available models with details"""
        models_list = []
        
        for model_id, model_info in self.registry['models'].items():
            model_path = self.models_dir / model_info['filename']
            
            # Check if model files exist
            h5_exists = model_path.exists()
            keras_path = self.models_dir / model_info['filename'].replace('.h5', '.keras')
            keras_exists = keras_path.exists()
            
            if h5_exists or keras_exists:
                # Get file size
                if h5_exists:
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                else:
                    size_mb = keras_path.stat().st_size / (1024 * 1024)
                
                model_data = {
                    'id': model_id,
                    'name': model_info.get('name', model_id),
                    'type': model_info.get('type', 'unknown'),
                    'created': model_info.get('created_date', 'Unknown'),
                    'description': model_info.get('description', ''),
                    'accuracy': model_info.get('accuracy', {}),
                    'size_mb': size_mb,
                    'h5_file': str(model_path) if h5_exists else None,
                    'keras_file': str(keras_path) if keras_exists else None
                }
                models_list.append(model_data)
        
        return models_list
    
    def display_model_selection_menu(self):
        """Display available models and return user selection"""
        models = self.get_all_models()
        
        if not models:
            print("[WARNING] No models found!")
            return None
        
        print("\n" + "=" * 70)
        print("  Available Models:")
        print("=" * 70 + "\n")
        
        for idx, model in enumerate(models, 1):
            print(f"  [{idx}] {model['name']}")
            print(f"      Type: {model['type']}")
            print(f"      Created: {model['created']}")
            print(f"      Size: {model['size_mb']:.2f} MB")
            
            if model['accuracy']:
                acc = model['accuracy']
                if 'directional_accuracy' in acc:
                    print(f"      Directional Accuracy: {acc['directional_accuracy']:.2f}%")
                if 'trend_accuracy' in acc:
                    print(f"      Trend Accuracy: {acc['trend_accuracy']:.2f}%")
                if 'mae' in acc:
                    print(f"      MAE: {acc['mae']:.2f}%")
            
            print(f"      {model['description']}")
            print()
        
        # Add training option
        print(f"  [{len(models) + 1}] Train a NEW model")
        print()
        
        # Get user selection
        while True:
            try:
                choice = input("  Select model number (or train new): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1]
                    self.registry['last_used'] = selected_model['id']
                    self._save_registry()
                    return selected_model
                elif choice_num == len(models) + 1:
                    return {'train_new': True}
                else:
                    print(f"  [ERROR] Please select between 1 and {len(models) + 1}")
            except ValueError:
                print(f"  [ERROR] Invalid input. Please enter a number between 1 and {len(models) + 1}")
    
    def get_model_files(self, model_id):
        """Get the actual model files for a given model ID"""
        if model_id not in self.registry['models']:
            return None, None, None
        
        model_info = self.registry['models'][model_id]
        h5_path = self.models_dir / model_info['filename']
        keras_path = self.models_dir / model_info['filename'].replace('.h5', '.keras')
        
        # Get scaler paths (use standard names)
        price_scaler = self.models_dir / 'price_scaler.pkl'
        feature_scaler = self.models_dir / 'feature_scaler.pkl'
        
        # Prefer .keras if exists, otherwise use .h5
        model_path = keras_path if keras_path.exists() else h5_path
        
        return model_path, price_scaler, feature_scaler
    
    def get_last_used_model(self):
        """Get the last used model"""
        last_id = self.registry.get('last_used')
        
        if last_id and last_id in self.registry['models']:
            models = self.get_all_models()
            for model in models:
                if model['id'] == last_id:
                    return model
        
        # Default to pretrained if available
        models = self.get_all_models()
        for model in models:
            if model['type'] == 'pretrained':
                return model
        
        return None


def initialize_model_system():
    """Initialize the model management system"""
    manager = ModelManager()
    manager.register_pretrained_model()
    return manager
