"""
Prosody predictor component for the TTS system using FKF (Feed-Forward, KAN, Feed-Forward) architecture.
Predicts prosody features (F0, energy, duration, voice quality) from phoneme representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from components.kan_blocks import BsplineKAN, MultiScaleKAN

class ProsodyPredictor(nn.Module):
    """
    Predicts prosody features from phoneme representations using FKF (Feed-Forward, KAN, Feed-Forward) architecture.
    The output includes F0, energy, duration, and voice quality features, each predicted by a dedicated submodule.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_basis: int = 8,
                 degree: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initial feed-forward layer: projects input to hidden_dim
        self.input_ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # KAN layer for non-linear transformation
        self.kan_layer = BsplineKAN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_basis=num_basis,
            degree=degree,
            dropout=dropout
        )
        
        # Final feed-forward layer: further transforms hidden representation
        self.output_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature-specific predictors with KAN architecture
        # Each predicts a different set of prosody features
        self.f0_predictor = ProsodyFeaturePredictor(hidden_dim, 4, num_basis, degree, dropout)  # F0 (mean, std, min, max)
        self.energy_predictor = ProsodyFeaturePredictor(hidden_dim, 2, num_basis, degree, dropout)  # Energy (mean, std)
        self.duration_predictor = ProsodyFeaturePredictor(hidden_dim, 1, num_basis, degree, dropout)  # Duration
        self.voice_quality_predictor = ProsodyFeaturePredictor(hidden_dim, 3, num_basis, degree, dropout)  # Jitter, Shimmer, HNR
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the FKF prosody predictor.
        Args:
            hidden_states: Phoneme hidden states [batch_size, seq_len, input_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
        Returns:
            Dictionary of predicted prosody features (all, and per-feature)
        """
        # Apply initial feed-forward layer
        x = self.input_ff(hidden_states)
        
        # Ensure input is 2D [batch_size, hidden_dim]
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove the extra dimension if present
        
        # Apply KAN transformation
        kan_out = self.kan_layer(x)
        
        # Apply final feed-forward layer
        x = self.output_ff(kan_out)
        
        # Predict individual features
        f0 = self.f0_predictor(x)  # [batch_size, seq_len, 4] - mean, std, min, max
        energy = self.energy_predictor(x)  # [batch_size, seq_len, 2] - mean, std
        duration = self.duration_predictor(x)  # [batch_size, seq_len, 1] - duration
        voice_quality = self.voice_quality_predictor(x)  # [batch_size, seq_len, 3] - jitter, shimmer, HNR
        
        # Combine all features into a single tensor for convenience
        prosody_features = torch.cat([
            f0, energy, duration, voice_quality
        ], dim=-1)  # [batch_size, seq_len, 10]
        
        return {
            'prosody_features': prosody_features,
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'voice_quality': voice_quality
        }

class ProsodyFeaturePredictor(nn.Module):
    """
    Predicts individual prosody features using a KAN layer and layer normalization.
    Used for F0, energy, duration, and voice quality sub-predictions.
    """
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 num_basis: int = 8,
                 degree: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # KAN layer for feature prediction
        self.kan = BsplineKAN(
            in_features=input_dim,
            out_features=output_dim,
            num_basis=num_basis,
            degree=degree,
            dropout=dropout
        )
        
        # Layer normalization for output stability
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature predictor.
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
        Returns:
            Predicted features [batch_size, seq_len, output_dim]
        """
        # Ensure input is 2D [batch_size, input_dim]
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove the extra dimension if present
        
        # Apply KAN transformation
        features = self.kan(x)
        
        # Apply layer normalization
        features = self.layer_norm(features)
        
        return features
