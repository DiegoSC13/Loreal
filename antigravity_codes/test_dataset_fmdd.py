import torch
from dataset import FMDDDataset, get_fmdd_sequences
from torch.utils.data import DataLoader
import argparse
import os

def test_fmdd_dataset(root_dir):
    print(f"Testing FMDDDataset with root: {root_dir}")
    
    # 1. Test sequence discovery
    sequences = get_fmdd_sequences(root_dir)
    print(f"Found {len(sequences)} sequences.")
    if not sequences:
        print("No sequences found. Check the path.")
        return

    # 2. Test RAW mode
    print("\n--- Testing RAW mode ---")
    dataset_raw = FMDDDataset(sequences, patch_size=(128, 128), mode='raw', data_scale=255.0)
    print(f"Dataset raw size (stacks): {len(dataset_raw)}")
    loader_raw = DataLoader(dataset_raw, batch_size=4, shuffle=True)
    stack, target = next(iter(loader_raw))
    print(f"RAW - Stack shape: {stack.shape}, Target shape: {target.shape}")
    assert stack.shape == (4, 5, 128, 128)

    # 3. Test SYNTHETIC mode
    print("\n--- Testing SYNTHETIC mode ---")
    gamma = 10.0
    dataset_synth = FMDDDataset(sequences, patch_size=(128, 128), mode='synthetic', gamma=gamma, data_scale=255.0)
    print(f"Dataset synth size (stacks): {len(dataset_synth)}")
    loader_synth = DataLoader(dataset_synth, batch_size=4, shuffle=True)
    stack, target = next(iter(loader_synth))
    print(f"SYNTHETIC - Stack shape: {stack.shape}, Target shape: {target.shape}")
    print(f"SYNTHETIC - Stack range: [{stack.min().item():.4f}, {stack.max().item():.4f}]")
    print(f"SYNTHETIC - Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
    
    # In synthetic mode, target should be cleaner than stack
    assert stack.shape == (4, 5, 128, 128)
    assert target.shape == (4, 1, 128, 128)
    # Check if they are derived from same base (approximately)
    # Since we added noise to target and then normalizar
    assert stack.max() <= 2.0 # Allow some headroom for Poisson
    
    print("\nFMDDDataset test PASSED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmdd_root", type=str, required=True)
    args = parser.parse_args()
    
    test_fmdd_dataset(args.fmdd_root)
