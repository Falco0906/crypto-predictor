#!/usr/bin/env python3
from src.utils.model_manager import initialize_model_system

manager = initialize_model_system()
models = manager.get_all_models()

print(f"\nTotal models found: {len(models)}\n")

for idx, model in enumerate(models, 1):
    print(f"{idx}. {model['name']}")
    print(f"   Type: {model['type']}")
    print(f"   ID: {model['id']}")
    print(f"   Accuracy: {model.get('accuracy', {})}\n")
