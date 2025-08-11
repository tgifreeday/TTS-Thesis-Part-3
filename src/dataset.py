#!/usr/bin/env python3
"""
Dataset for TTS Training
=======================

This module provides the TTSDataset class for loading and processing
linguistic features and audio features for TTS training.
"""

import json
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TTSDataset(Dataset):
    """Dataset for TTS training with linguistic and audio features."""
    
    def __init__(self, data_path: str, file_list_path: Optional[str] = None, 
                 base_path: str = "data/processed", input_features: Optional[List[str]] = None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON file with linguistic features
            file_list_path: Path to file containing list of file IDs to include
            base_path: Base path for loading audio feature files
            input_features: List of feature names to use (if None, uses all features)
        """
        self.data_path = data_path
        self.base_path = Path(base_path)
        
        # Load data
        self._load_data(file_list_path)
        
        # Define input features (configurable for ablation studies)
        if input_features is None:
            # Default: all features for full experiments
            self.input_features = [
                'phoneme_id', 'stress_level', 'primary_stress_pos', 'secondary_stress_pos',
                'pos_tag', 'is_content_word', 'word_position_in_sentence', 'is_sentence_initial',
                'is_sentence_final', 'syllable_count', 'word_length', 'is_compound',
                'is_loanword', 'vowel_count', 'consonant_count'
            ]
        else:
            # Use specified features for ablation studies
            self.input_features = input_features
        
        # Initialize POS tag vocabulary mapping
        self.pos_tag_vocab = self._build_pos_tag_vocabulary()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
        logger.info(f"Input features: {self.input_features}")
        logger.info(f"POS tag vocabulary size: {len(self.pos_tag_vocab)}")
    
    def _load_data(self, file_list_path: Optional[str] = None):
        """Load data from JSON file."""
        logger.info(f"Loading data from {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Load file list if provided
        if file_list_path:
            with open(file_list_path, 'r') as f:
                file_list = [line.strip() for line in f if line.strip()]
        else:
            # Use all files
            file_list = [item.get('file_id') for item in raw_data if item.get('file_id')]
        
        # Filter data based on file list
        self.data = []
        for item in raw_data:
            if item.get('file_id') in file_list:
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} samples from {len(file_list)} files")
    
    def _build_pos_tag_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary mapping for POS tags."""
        pos_tags = set()
        
        # Collect all unique POS tags from the dataset
        for sample in self.data:
            for word in sample.get('words', []):
                pos_tag = word.get('pos_tag', 'UNK')
                pos_tags.add(pos_tag)
        
        # Create vocabulary mapping
        vocab = {'UNK': 0}  # Unknown token gets index 0
        for i, pos_tag in enumerate(sorted(pos_tags), start=1):
            vocab[pos_tag] = i
        
        logger.info(f"Built POS tag vocabulary with {len(vocab)} tags: {list(vocab.keys())}")
        return vocab
    
    def _get_pos_tag_id(self, pos_tag: str) -> int:
        """Get integer ID for a POS tag using vocabulary mapping."""
        return self.pos_tag_vocab.get(pos_tag, 0)  # Default to UNK (0) if not found
    
    def _extract_phoneme_ids(self, sample: Dict[str, Any]) -> List[int]:
        """Extract phoneme IDs from sample."""
        phoneme_ids = []
        for word in sample.get('words', []):
            for phoneme in word.get('phonemes', []):
                phoneme_id = phoneme.get('phoneme_id', 0)
                phoneme_ids.append(int(phoneme_id))
        return phoneme_ids
    
    def _extract_linguistic_features(self, sample: Dict[str, Any]) -> List[List[float]]:
        """Extract linguistic features from sample."""
        features = []
        for word in sample.get('words', []):
            for phoneme in word.get('phonemes', []):
                # Extract features for each phoneme based on configured features
                phoneme_features = []
                
                for feature_name in self.input_features:
                    if feature_name == 'phoneme_id':
                        phoneme_features.append(float(phoneme.get('phoneme_id', 0)))
                    elif feature_name == 'stress_level':
                        phoneme_features.append(float(word.get('stress_level', 0.0)))
                    elif feature_name == 'primary_stress_pos':
                        phoneme_features.append(float(word.get('primary_stress_pos', 0.0)))
                    elif feature_name == 'secondary_stress_pos':
                        phoneme_features.append(float(word.get('secondary_stress_pos', 0.0)))
                    elif feature_name == 'pos_tag':
                        # ACADEMIC RIGOR: Use proper vocabulary mapping for POS tags
                        pos_tag = word.get('pos_tag', 'UNK')
                        pos_tag_id = self._get_pos_tag_id(pos_tag)
                        phoneme_features.append(float(pos_tag_id))
                    elif feature_name == 'is_content_word':
                        is_content = 1 if word.get('is_content', False) else 0
                        phoneme_features.append(float(is_content))
                    elif feature_name == 'word_position_in_sentence':
                        word_position = word.get('word_position_in_sentence', 0.0)
                        phoneme_features.append(float(word_position))
                    elif feature_name == 'is_sentence_initial':
                        is_initial = 1 if word.get('is_sentence_initial', False) else 0
                        phoneme_features.append(float(is_initial))
                    elif feature_name == 'is_sentence_final':
                        is_final = 1 if word.get('is_sentence_final', False) else 0
                        phoneme_features.append(float(is_final))
                    elif feature_name == 'syllable_count':
                        syllable_count = word.get('syllable_count', 1)
                        phoneme_features.append(float(syllable_count))
                    elif feature_name == 'word_length':
                        word_length = word.get('word_length_chars', len(word.get('phonemes', [])))
                        phoneme_features.append(float(word_length))
                    elif feature_name == 'is_compound':
                        is_compound = 1 if word.get('is_compound', False) else 0
                        phoneme_features.append(float(is_compound))
                    elif feature_name == 'is_loanword':
                        is_loanword = 1 if word.get('is_loanword', False) else 0
                        phoneme_features.append(float(is_loanword))
                    elif feature_name == 'vowel_count':
                        vowel_count = word.get('vowel_count', 0)
                        phoneme_features.append(float(vowel_count))
                    elif feature_name == 'consonant_count':
                        consonant_count = word.get('consonant_count', 0)
                        phoneme_features.append(float(consonant_count))
                    else:
                        # Unknown feature, use default value
                        phoneme_features.append(0.0)
                
                features.append(phoneme_features)
        
        return features
    
    def _load_audio_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load audio features from separate .npy files."""
        audio_features = sample.get('audio_features', {})
        if not audio_features:
            return {}
        
        loaded_features = {}
        
        # Load mel spectrogram
        mel_spec_path = audio_features.get('mel_spectrogram_path')
        if mel_spec_path:
            try:
                mel_spec_file = self.base_path / mel_spec_path
                if mel_spec_file.exists():
                    loaded_features['mel_spectrogram'] = np.load(mel_spec_file)
                else:
                    logger.warning(f"Mel spectrogram file not found: {mel_spec_file}")
            except Exception as e:
                logger.error(f"Error loading mel spectrogram: {e}")
        
        # Load F0
        f0_path = audio_features.get('f0_path')
        if f0_path:
            try:
                f0_file = self.base_path / f0_path
                if f0_file.exists():
                    loaded_features['f0'] = np.load(f0_file)
                else:
                    logger.warning(f"F0 file not found: {f0_file}")
            except Exception as e:
                logger.error(f"Error loading F0: {e}")
        
        # Load energy
        energy_path = audio_features.get('energy_path')
        if energy_path:
            try:
                energy_file = self.base_path / energy_path
                if energy_file.exists():
                    loaded_features['energy'] = np.load(energy_file)
                else:
                    logger.warning(f"Energy file not found: {energy_file}")
            except Exception as e:
                logger.error(f"Error loading energy: {e}")
        
        # Duration is stored directly in JSON
        duration = audio_features.get('duration', 0.0)
        loaded_features['duration'] = duration
        
        return loaded_features
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        sample = self.data[idx]
        
        # Extract phoneme IDs
        phoneme_ids = self._extract_phoneme_ids(sample)
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(sample)
        
        # Convert to tensors
        phoneme_ids_tensor = torch.tensor(phoneme_ids, dtype=torch.long)
        linguistic_features_tensor = torch.tensor(linguistic_features, dtype=torch.float32)
        
        # Load audio features
        audio_features = self._load_audio_features(sample)
        
        # Prepare return dictionary
        item = {
            'phoneme_ids': phoneme_ids_tensor,
            'linguistic_features': linguistic_features_tensor,
        }
        
        # Add audio features if available
        if audio_features:
            if 'mel_spectrogram' in audio_features:
                # Load quantized data and convert to float32 for training
                mel_spec = audio_features['mel_spectrogram'].astype(np.float32)
                item['mel_spectrogram'] = torch.tensor(mel_spec, dtype=torch.float32)
            
            # CRITICAL FIX: Ensure F0 and energy lengths match mel spectrogram
            mel_length = None
            if 'mel_spectrogram' in audio_features:
                mel_length = audio_features['mel_spectrogram'].shape[0]
            
            if 'f0' in audio_features:
                f0 = audio_features['f0'].astype(np.float32)
                # CRITICAL: Truncate F0 to match mel length if needed
                if mel_length is not None and f0.shape[0] != mel_length:
                    print(f"⚠️ WARNING: F0 length {f0.shape[0]} != mel length {mel_length}, truncating")
                    f0 = f0[:mel_length] if f0.shape[0] > mel_length else np.pad(f0, (0, mel_length - f0.shape[0]))
                item['f0'] = torch.tensor(f0, dtype=torch.float32)
            
            if 'energy' in audio_features:
                energy = audio_features['energy'].astype(np.float32)
                # CRITICAL: Truncate energy to match mel length if needed
                if mel_length is not None and energy.shape[0] != mel_length:
                    print(f"⚠️ WARNING: Energy length {energy.shape[0]} != mel length {mel_length}, truncating")
                    energy = energy[:mel_length] if energy.shape[0] > mel_length else np.pad(energy, (0, mel_length - energy.shape[0]))
                item['energy'] = torch.tensor(energy, dtype=torch.float32)
        
        # CRITICAL FIX: Extract TRUE per-phoneme durations from forced alignment data
        per_phoneme_durations = []
        for word in sample.get('words', []):
            for phoneme in word.get('phonemes', []):
                per_phoneme_durations.append(phoneme.get('duration', 0.0))
        
        # Handle case where no phoneme durations are found
        if not per_phoneme_durations:
            # Fallback: use uniform distribution (but this should be rare)
            num_phonemes = len(phoneme_ids)
            if num_phonemes > 0:
                total_duration = audio_features.get('duration', 1.0) if audio_features else 1.0
                per_phoneme_duration = total_duration / num_phonemes
                duration_tensor = torch.full((num_phonemes,), per_phoneme_duration, dtype=torch.float32)
            else:
                duration_tensor = torch.zeros(0, dtype=torch.float32)
        else:
            # Use the REAL per-phoneme durations from forced alignment
            duration_tensor = torch.tensor(per_phoneme_durations, dtype=torch.float32)
        
        item['duration'] = duration_tensor
        
        # CRITICAL: Pre-calculate duration_frames and duration_log for efficient training
        # Convert duration to frame counts for length regulator
        frame_rate = 22050  # Hz
        hop_length = 256  # frames per hop
        frames_per_second = frame_rate / hop_length  # ~86 frames per second
        
        duration_frames = (duration_tensor * frames_per_second).round().long()
        duration_frames = torch.clamp(duration_frames, min=0)  # Some phonemes can be 0 frames
        
        # CRITICAL FIX: Ensure duration_frames match mel spectrogram length exactly
        if audio_features and 'mel_spectrogram' in audio_features:
            mel_length = audio_features['mel_spectrogram'].shape[0]  # Time dimension
            duration_sum = duration_frames.sum().item()
            
            # If they don't match, adjust duration_frames to match mel length
            if mel_length != duration_sum:
                # Calculate adjustment factor
                adjustment_factor = mel_length / duration_sum
                duration_frames = (duration_frames.float() * adjustment_factor).round().long()
        
        item['duration_frames'] = duration_frames
        
        # Convert duration to log space for loss calculation
        duration_log = torch.log(duration_tensor + 1e-8)  # Add small epsilon to avoid log(0)
        item['duration_log'] = duration_log
        
        return item 