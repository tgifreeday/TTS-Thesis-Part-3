#!/usr/bin/env python3
"""
Test Dataset Loading
===================

Test script to verify that the updated dataset loader can correctly
load audio features from separate .npy files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from dataset import TTSDataset

def test_dataset_loading():
    """Test the updated dataset loader."""
    
    print("Testing dataset loading with audio features...")
    
    # Create dataset with the new file
    dataset = TTSDataset(
        data_path="data/processed/linguistic_features_with_audio_paths.json",
        file_list_path="data/processed/train_files.txt",
        base_path="data/processed"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a few samples
    for i in range(3):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Phoneme IDs shape: {sample['phoneme_ids'].shape}")
        print(f"  Linguistic features shape: {sample['linguistic_features'].shape}")
        
        # Check for audio features
        if 'mel_spectrogram' in sample:
            print(f"  Mel spectrogram shape: {sample['mel_spectrogram'].shape}")
        else:
            print("  No mel spectrogram found")
            
        if 'f0' in sample:
            print(f"  F0 shape: {sample['f0'].shape}")
        else:
            print("  No F0 found")
            
        if 'energy' in sample:
            print(f"  Energy shape: {sample['energy'].shape}")
        else:
            print("  No energy found")
            
        if 'duration' in sample:
            print(f"  Duration: {sample['duration'].item()}")
        else:
            print("  No duration found")
    
    print("\nDataset loading test completed successfully!")

if __name__ == "__main__":
    test_dataset_loading() 