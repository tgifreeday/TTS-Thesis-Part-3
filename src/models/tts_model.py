"""
Main TTS model architecture combining phoneme encoding, prosody prediction, and spectrogram generation.
This class integrates all major components of the TTS pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .phoneme_encoder import PhonemeEncoder
from .prosody_predictor import ProsodyPredictor
from .spectrogram_decoder import SpectrogramDecoder

class TTSModel(nn.Module):
    """
    Main TTS model combining all components: phoneme encoder, prosody predictor, and spectrogram decoder.
    Handles the end-to-end forward pass for TTS.
    """
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Initialize phoneme encoder (e.g., Transformer-based)
        self.phoneme_encoder = PhonemeEncoder(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
        # Initialize prosody predictor (FKF architecture)
        self.prosody_predictor = ProsodyPredictor(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=len(config.prosody_features),
            num_basis=config.kan_num_basis,
            degree=config.kan_degree,
            dropout=config.kan_dropout
        )
        
        # Initialize spectrogram decoder (placeholder or actual implementation)
        # Takes concatenated phoneme and prosody features as input
        print(f"[DEBUG] TTSModel: config.hifigan_resblock_kernel_sizes = {config.hifigan_resblock_kernel_sizes}")
        self.spectrogram_decoder = SpectrogramDecoder(
            input_dim=config.hidden_dim + len(config.prosody_features),
            hidden_dim=config.hidden_dim,
            output_dim=config.n_mels,
            num_layers=config.num_layers,
            num_basis=config.kan_num_basis,
            degree=config.kan_degree,
            dropout=config.kan_dropout,
            kernel_sizes=config.hifigan_resblock_kernel_sizes
        )
        
        # Initialize weights for all submodules
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize weights for better training stability.
        Uses Xavier initialization for linear and convolutional layers, and sets LayerNorm weights/biases.
        """
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, phoneme_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the TTS model.
        Args:
            phoneme_ids: Input phoneme IDs [batch_size, seq_len]
        Returns:
            Dictionary of model outputs, including mel spectrogram, phoneme features, and prosody features.
        """
        # Get phoneme features from encoder
        phoneme_features = self.phoneme_encoder(phoneme_ids)
        
        # Get prosody features from prosody predictor
        prosody_features = self.prosody_predictor(phoneme_features['hidden_states'])
        
        # Debug: print shapes before concatenation
        print('phoneme_features[hidden_states] shape:', phoneme_features['hidden_states'].shape)
        print('prosody_features[prosody_features] shape:', prosody_features['prosody_features'].shape)
        
        # Ensure both tensors have the same number of dimensions
        if phoneme_features['hidden_states'].dim() == 3 and prosody_features['prosody_features'].dim() == 2:
            prosody_features['prosody_features'] = prosody_features['prosody_features'].unsqueeze(1)  # Add sequence dimension
        
        # Concatenate phoneme and prosody features along the last dimension
        combined_features = torch.cat([
            phoneme_features['hidden_states'],
            prosody_features['prosody_features']  # Extract the tensor from the dictionary
        ], dim=-1)
        
        # Generate mel spectrogram from combined features
        mel_spectrogram = self.spectrogram_decoder(combined_features)
        
        return {
            'mel_spectrogram': mel_spectrogram,
            'phoneme_features': phoneme_features,
            'prosody_features': prosody_features
        }
    
    def generate(self,
                phoneme_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Generate speech from phoneme IDs with optional temperature control.
        Args:
            phoneme_ids: Input phoneme IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            temperature: Temperature for sampling (higher = more random)
        Returns:
            Dictionary containing generated mel spectrogram and prosody features
        """
        self.eval()
        with torch.no_grad():
            # Get model predictions
            output = self.forward(phoneme_ids)
            
            # Apply temperature scaling if needed
            if temperature != 1.0:
                output['mel_spectrogram'] = output['mel_spectrogram'] / temperature
                output['prosody_features'] = output['prosody_features'] / temperature
            
            return output
