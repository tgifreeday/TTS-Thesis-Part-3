"""
Spectrogram decoder component for the TTS system.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from components.kan_blocks import MultiScaleKAN, BsplineKAN
import os

class SpectrogramDecoder(nn.Module):
    """Generates mel spectrograms from combined features using KAN architecture."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 kernel_sizes: List[int],
                 num_layers: int = 4,
                 num_basis: int = 8,
                 degree: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_sizes = kernel_sizes # <--- STORE THE ARGUMENT
        
        # CRITICAL FIX: Only print debug info if DEBUG environment variable is set
        if os.environ.get("DEBUG", False):
            print(f"[DEBUG] SpectrogramDecoder loaded from: {os.path.abspath(__file__)}")
            print(f"[DEBUG] SpectrogramDecoder.__init__ received kernel_sizes: {kernel_sizes}")
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-scale KAN layers for feature extraction
        self.feature_layers = nn.ModuleList([
            MultiScaleKAN(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_sizes=self.kernel_sizes,
                num_basis=num_basis,
                degree=degree,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Upsampling layers with KAN-based feature refinement
        self.upsampling_layers = nn.ModuleList([
            nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(
                    hidden_dim,
                    hidden_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                'kan': MultiScaleKAN(
                    in_channels=hidden_dim // 2,
                    out_channels=hidden_dim // 2,
                    kernel_sizes=self.kernel_sizes,
                    num_basis=num_basis,
                    degree=degree,
                    dropout=dropout
                ),
                'norm': nn.LayerNorm((hidden_dim // 2,)),
                'activation': nn.GELU()
            })
            for _ in range(2)
        ])
        
        # Final KAN layer for spectrogram generation
        self.final_kan = BsplineKAN(
            in_features=hidden_dim // 4,
            out_features=output_dim,
            num_basis=num_basis,
            degree=degree,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                target_length: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through the spectrogram decoder.
        
        Args:
            features: Combined features [batch_size, seq_len, input_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            target_length: Optional target length for truncation
            
        Returns:
            Generated mel spectrogram [batch_size, seq_len, output_dim]
        """
        # Project input
        x = self.input_proj(features)
        
        # Reshape for convolutional processing
        batch_size, seq_len, hidden_dim = x.shape
        x = x.transpose(1, 2).unsqueeze(-1)  # [batch_size, hidden_dim, seq_len, 1]
        
        # Apply multi-scale KAN layers
        for feature_layer in self.feature_layers:
            # Apply KAN transformation
            kan_out = feature_layer(x)
            
            # Apply residual connection and layer norm
            x = x + kan_out
            x = self.layer_norm(x.transpose(1, -1)).transpose(1, -1)
            
            # Apply dropout
            x = self.dropout(x)
        
        # Apply upsampling layers with KAN refinement
        for upsampling_layer in self.upsampling_layers:
            # Upsample
            x = upsampling_layer['upsample'](x)
            
            # Apply KAN refinement
            x = x + upsampling_layer['kan'](x)
            
            # Apply normalization and activation
            x = upsampling_layer['norm'](x)
            x = upsampling_layer['activation'](x)
            
            # Apply dropout
            x = self.dropout(x)
        
        # Reshape for final KAN layer
        x = x.squeeze(-1).transpose(1, 2)  # [batch_size, seq_len, hidden_dim // 4]
        
        # Generate spectrogram
        spectrogram = self.final_kan(x)
        
        # CRITICAL FIX: Truncate to target length if provided (handles upsampling expansion)
        if target_length is not None and spectrogram.shape[1] != target_length:
            if os.environ.get("DEBUG", False):
                print(f"⚠️ WARNING: Spectrogram length {spectrogram.shape[1]} != target {target_length}, truncating")
            # Truncate to target length
            spectrogram = spectrogram[:, :target_length, :]
        
        return spectrogram
