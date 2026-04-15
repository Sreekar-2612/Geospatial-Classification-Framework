"""Check saved model structure"""
import torch
from pathlib import Path

model_path = Path(__file__).parent / "models" / "cnn_final.pth"

print(f"Loading {model_path}...")
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

print(f"\nState dict keys related to classifier:")
for key in sorted(state_dict.keys()):
    if 'classifier' in key:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")

print(f"\nAll final keys:")
keys_list = list(state_dict.keys())
for key in sorted(keys_list)[-5:]:
    print(f"  {key}: {state_dict[key].shape}")
