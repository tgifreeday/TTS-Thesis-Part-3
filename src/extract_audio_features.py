#!/usr/bin/env python3
"""
Audio Feature Extraction Script
==============================

This script extracts audio features (mel-spectrograms, F0, energy, duration) from WAV files
and adds them to the linguistic features dataset for training.

Instead of embedding large arrays in JSON, this script saves audio features as separate .npy files
and stores only the file paths in the JSON to keep it manageable.

Usage:
    python src/extract_audio_features.py
"""

import os
import json
import librosa
import numpy as np
import parselmouth
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """Extract audio features from WAV files and save as separate .npy files."""
    
    def __init__(self, audio_base_path: str, output_path: str):
        self.audio_base_path = Path(audio_base_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different feature types
        self.features_dir = self.output_path / "audio_features"
        self.features_dir.mkdir(exist_ok=True)
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.n_mels = 80
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        
    def extract_mel_spectrogram(self, audio_path: str) -> np.ndarray:
        """Extract mel-spectrogram from audio file."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db.T  # Transpose to (time, features)
            
        except Exception as e:
            logger.error(f"Error extracting mel-spectrogram from {audio_path}: {e}")
            return np.array([])
    
    def extract_f0(self, audio_path: str) -> np.ndarray:
        """Extract F0 (pitch) contour using Parselmouth."""
        try:
            # Load audio with Parselmouth
            sound = parselmouth.Sound(audio_path)
            
            # Extract pitch
            pitch = sound.to_pitch()
            f0_values = pitch.selected_array['frequency']
            
            # Convert to numpy array and handle unvoiced frames
            f0 = np.array(f0_values)
            f0[f0 == 0] = np.nan  # Mark unvoiced frames
            
            return f0
            
        except Exception as e:
            logger.error(f"Error extracting F0 from {audio_path}: {e}")
            return np.array([])
    
    def extract_energy(self, audio_path: str) -> np.ndarray:
        """Extract energy contour from audio."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract RMS energy
            energy = librosa.feature.rms(
                y=y,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            
            return energy.flatten()
            
        except Exception as e:
            logger.error(f"Error extracting energy from {audio_path}: {e}")
            return np.array([])
    
    def extract_duration(self, audio_path: str) -> float:
        """Extract total duration of audio file."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Calculate duration
            duration = len(y) / sr
            
            return duration
            
        except Exception as e:
            logger.error(f"Error extracting duration from {audio_path}: {e}")
            return 0.0
    
    def find_audio_file(self, file_id: str) -> Optional[str]:
        """Find the corresponding audio file for a given file_id."""
        # Parse file_id (e.g., "ch34_sent_070" -> chapter="34", sentence="070")
        if not file_id.startswith("ch"):
            return None
            
        parts = file_id.split("_")
        if len(parts) != 3:
            return None
            
        chapter = parts[0][2:]  # Remove "ch" prefix
        sentence = parts[2]
        
        # Construct audio file path - now the chapter number in filename matches the folder
        audio_filename = f"ch{chapter}_sent_{sentence}.wav"
        audio_path = self.audio_base_path / f"chapter{chapter}" / audio_filename
        
        if audio_path.exists():
            return str(audio_path)
        else:
            logger.warning(f"Audio file not found: {audio_path}")
            return None
    
    def save_audio_features(self, file_id: str, mel_spec: np.ndarray, f0: np.ndarray, 
                          energy: np.ndarray, duration: float) -> Dict[str, str]:
        """Save audio features as separate .npy files and return file paths."""
        # Parse chapter number from file_id
        if file_id.startswith("ch"):
            chapter_num = file_id[2:].split("_")[0]  # Extract chapter number
            chapter_dir = f"chapter{chapter_num}"
        else:
            chapter_dir = "unknown"
        
        # Create chapter subdirectory
        chapter_path = self.features_dir / chapter_dir
        chapter_path.mkdir(exist_ok=True)
        
        # Quantize to float16 for significant size reduction (~50%)
        # This maintains sufficient precision for TTS research while reducing storage
        mel_spec_quantized = mel_spec.astype(np.float16)
        f0_quantized = f0.astype(np.float16)
        energy_quantized = energy.astype(np.float16)
        
        # Create feature file paths within chapter directory
        mel_spec_path = chapter_path / f"{file_id}_mel_spec.npy"
        f0_path = chapter_path / f"{file_id}_f0.npy"
        energy_path = chapter_path / f"{file_id}_energy.npy"
        
        # Save quantized features as .npy files
        np.save(mel_spec_path, mel_spec_quantized)
        np.save(f0_path, f0_quantized)
        np.save(energy_path, energy_quantized)
        
        # Return relative paths for JSON storage
        return {
            'mel_spectrogram_path': str(mel_spec_path.relative_to(self.output_path)),
            'f0_path': str(f0_path.relative_to(self.output_path)),
            'energy_path': str(energy_path.relative_to(self.output_path)),
            'duration': duration
        }
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample and add audio feature paths."""
        file_id = sample.get('file_id')
        if not file_id:
            logger.warning(f"Sample missing file_id: {sample}")
            return sample
        
        # Find corresponding audio file
        audio_path = self.find_audio_file(file_id)
        if not audio_path:
            logger.warning(f"No audio file found for {file_id}")
            return sample
        
        # Extract audio features
        logger.info(f"Processing {file_id}...")
        
        mel_spec = self.extract_mel_spectrogram(audio_path)
        f0 = self.extract_f0(audio_path)
        energy = self.extract_energy(audio_path)
        duration = self.extract_duration(audio_path)
        
        # Save features as separate files and get paths
        audio_feature_paths = self.save_audio_features(file_id, mel_spec, f0, energy, duration)
        
        # Add audio feature paths to sample
        sample['audio_features'] = audio_feature_paths
        
        return sample
    
    def process_dataset(self, input_path: str, output_path: str):
        """Process the entire dataset and add audio feature paths."""
        logger.info(f"Loading dataset from {input_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Processing {len(data)} samples...")
        
        processed_data = []
        successful = 0
        failed = 0
        
        for i, sample in enumerate(data):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(data)} samples processed")
            
            try:
                processed_sample = self.process_sample(sample)
                processed_data.append(processed_sample)
                successful += 1
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                failed += 1
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        # Save processed dataset
        logger.info(f"Saving processed dataset to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        logger.info("Audio feature extraction complete!")
        logger.info(f"Audio features saved to: {self.features_dir}")


def main():
    """Main function."""
    # Paths
    audio_base_path = "/Users/s.mengari/Documents/KAN_BASELINE/ALL_wav_txt_sentence_level_cleaned"
    input_path = "data/processed/linguistic_features.json"
    output_path = "data/processed/linguistic_features_with_audio_paths.json"
    
    if not os.path.exists(audio_base_path):
        logger.error(f"Audio base path does not exist: {audio_base_path}")
        return
    
    # Create extractor and execute
    extractor = AudioFeatureExtractor(audio_base_path, "data/processed")
    
    try:
        extractor.process_dataset(input_path, output_path)
        logger.info("Audio feature extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during extraction process: {e}")


if __name__ == "__main__":
    main() 