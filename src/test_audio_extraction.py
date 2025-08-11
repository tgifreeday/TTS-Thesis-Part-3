#!/usr/bin/env python3
"""
Test Audio Feature Extraction
============================

Test script to verify audio feature extraction works correctly.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from extract_audio_features import AudioFeatureExtractor

def test_audio_extraction():
    """Test audio feature extraction with a small sample."""
    
    # Load a small sample of data
    with open("data/processed/linguistic_features.json", 'r') as f:
        data = json.load(f)
    
    # Take first 3 samples for testing
    test_samples = data[:3]
    
    print(f"Testing with {len(test_samples)} samples:")
    for sample in test_samples:
        print(f"  - {sample['file_id']}")
    
    # Create extractor
    audio_base_path = "/Users/s.mengari/Documents/KAN_BASELINE/ALL_wav_txt_sentence_level_cleaned"
    extractor = AudioFeatureExtractor(audio_base_path, "data/processed")
    
    # Process test samples
    processed_samples = []
    for sample in test_samples:
        try:
            processed_sample = extractor.process_sample(sample)
            processed_samples.append(processed_sample)
            print(f"✓ Successfully processed {sample['file_id']}")
            
            # Check if audio features were added
            if 'audio_features' in processed_sample:
                audio_features = processed_sample['audio_features']
                print(f"  - Mel-spec shape: {len(audio_features['mel_spectrogram'])} frames")
                print(f"  - F0 length: {len(audio_features['f0'])}")
                print(f"  - Energy length: {len(audio_features['energy'])}")
                print(f"  - Duration: {audio_features['duration']:.2f}s")
            else:
                print(f"  ✗ No audio features added")
                
        except Exception as e:
            print(f"✗ Error processing {sample['file_id']}: {e}")
    
    # Save test results
    test_output_path = "data/processed/test_audio_features.json"
    with open(test_output_path, 'w') as f:
        json.dump(processed_samples, f, indent=2)
    
    print(f"\nTest results saved to: {test_output_path}")
    print("Audio feature extraction test complete!")

if __name__ == "__main__":
    test_audio_extraction() 