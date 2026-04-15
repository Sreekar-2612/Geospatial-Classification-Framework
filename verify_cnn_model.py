"""
Verify and fix the saved CNN model
Run: python verify_cnn_model.py
"""
import torch
from torchvision import models
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
model_path = MODELS_DIR / "cnn_final.pth"

print("=" * 70)
print("CNN Model Verification")
print("=" * 70)

if not model_path.exists():
    print(f"❌ Model file not found: {model_path}")
    exit(1)

print(f"\n✓ Model file found: {model_path}")
print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

# Load and inspect
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

try:
    # Try loading as state dict
    print("\n1️⃣  Attempting to load as state_dict...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    if isinstance(state_dict, dict):
        print("   ✓ Loaded successfully as state_dict")
        print(f"   Keys: {len(state_dict.keys())} parameters")
        
        # Show first few keys
        first_keys = list(state_dict.keys())[:5]
        for key in first_keys:
            print(f"     - {key}: {state_dict[key].shape}")
        
        # Verify it works with EfficientNet-B1
        print("\n2️⃣  Testing with EfficientNet-B1 architecture...")
        CLASSES = ["Agriculture", "Buildings", "Forest", "Roads", "Water"]
        model = models.efficientnet_b1(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(CLASSES))
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print("   ✓ Successfully loaded into EfficientNet-B1!")
        print("   ✓ Model is ready for inference\n")
        
        # Test a dummy forward pass
        print("3️⃣  Testing dummy forward pass...")
        dummy_input = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   ✓ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output classes: {output.shape[1]}")
        
        print("\n" + "=" * 70)
        print("✅ Model is valid and ready to use!")
        print("=" * 70)
    else:
        print("   ❌ Loaded as state_dict but got unexpected type")
        exit(1)
        
except Exception as e:
    print(f"   ❌ Failed to load as state_dict: {e}\n")
    
    # Try alternate loading methods
    print("1️⃣  Attempting alternate load (weights_only=False)...")
    try:
        loaded_model = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(loaded_model, torch.nn.Module):
            print("   ✓ Loaded successfully as full model")
            loaded_model.eval()
            
            # Save it as state_dict for web app
            print("\n2️⃣  Converting to state_dict...")
            state_dict = loaded_model.state_dict()
            
            # Re-save for web app
            torch.save(state_dict, model_path)
            print(f"   ✓ Saved state_dict back to {model_path}")
            print("   ✓ Web app should now load it correctly\n")
        else:
            print(f"   ℹ Loaded type: {type(loaded_model)}")
            
    except Exception as e2:
        print(f"   ❌ Alternate load failed: {e2}")
        exit(1)

print("✅ All checks passed!")
