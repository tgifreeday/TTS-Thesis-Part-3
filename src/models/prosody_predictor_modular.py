"""
Modular prosody predictor components for the TTS system.
Supports MLP, Transformer, and KAN-FKF architectures for baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from components.kan_blocks import BsplineKAN, MultiScaleKAN


class BaseProsodyPredictor(nn.Module):
    """
    Base class for all prosody predictors.
    Ensures consistent interface across different architectures.
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the prosody predictor.
        Args:
            hidden_states: Phoneme hidden states [batch_size, seq_len, input_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
        Returns:
            Dictionary of predicted prosody features
        """
        raise NotImplementedError("Subclasses must implement forward method")


class MLPPredictor(BaseProsodyPredictor):
    """
    Simple feed-forward network baseline for prosody prediction.
    Uses standard MLP architecture with ReLU activations and dropout.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.1,
                 activation: str = "relu",
                 batch_norm: bool = True):
        # CRITICAL FIX: Use input_dim as output_dim for consistency with config
        super().__init__(input_dim, input_dim)
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if batch_norm else nn.Identity(),  # CRITICAL FIX: Use LayerNorm for stability
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer for output
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        
        self.mlp = nn.Sequential(*layers)
        
        # Feature-specific predictors
        # CRITICAL FIX: Predict single values per phoneme to match ground-truth contours
        self.f0_predictor = nn.Linear(hidden_dims[-1], 1)  # F0 (single value per phoneme)
        self.energy_predictor = nn.Linear(hidden_dims[-1], 1)  # Energy (single value per phoneme)
        self.duration_predictor = nn.Linear(hidden_dims[-1], 1)  # Duration
        self.voice_quality_predictor = nn.Linear(hidden_dims[-1], 3)  # Jitter, Shimmer, HNR
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MLP prosody predictor.
        """
        # Reshape for batch norm: (batch_size, seq_len, features) -> (batch_size * seq_len, features)
        batch_size, seq_len, features = hidden_states.shape
        x = hidden_states.view(-1, features)
        
        # Apply MLP transformation
        x = self.mlp(x)
        
        # Reshape back: (batch_size * seq_len, features) -> (batch_size, seq_len, features)
        x = x.view(batch_size, seq_len, -1)
        
        # CRITICAL FIX: Predict per-phoneme features (not utterance-level)
        # All predictors must output sequence-level predictions for valid comparison
        f0 = self.f0_predictor(x)  # [batch_size, seq_len, 1]
        energy = self.energy_predictor(x)  # [batch_size, seq_len, 1]
        duration = self.duration_predictor(x)  # [batch_size, seq_len, 1]
        voice_quality = self.voice_quality_predictor(x)  # [batch_size, seq_len, 3]
        
        # Combine all features
        prosody_features = torch.cat([f0, energy, duration, voice_quality], dim=-1)
        
        return {
            'prosody_features': prosody_features,
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'voice_quality': voice_quality
        }


class TransformerPredictor(BaseProsodyPredictor):
    """
    Transformer-based prosody predictor using multi-head attention.
    State-of-the-art attention-based architecture for comparison.
    """
    
    def __init__(self, 
                 input_dim: int,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 layer_norm_eps: float = 1e-6):
        super().__init__(input_dim, d_model)
        
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout, activation, layer_norm_eps)
            for _ in range(n_layers)
        ])
        
        # Feature-specific predictors
        # CRITICAL FIX: Predict single values per phoneme to match ground-truth contours
        self.f0_predictor = nn.Linear(d_model, 1)  # F0 (single value per phoneme)
        self.energy_predictor = nn.Linear(d_model, 1)  # Energy (single value per phoneme)
        self.duration_predictor = nn.Linear(d_model, 1)
        self.voice_quality_predictor = nn.Linear(d_model, 3)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Transformer prosody predictor.
        """
        # Project input to d_model
        x = self.input_projection(hidden_states)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # Predict individual features
        f0 = self.f0_predictor(x)
        energy = self.energy_predictor(x)
        duration = self.duration_predictor(x)
        voice_quality = self.voice_quality_predictor(x)
        
        # Combine all features
        prosody_features = torch.cat([f0, energy, duration, voice_quality], dim=-1)
        
        return {
            'prosody_features': prosody_features,
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'voice_quality': voice_quality
        }


class TransformerLayer(nn.Module):
    """
    Single transformer layer with self-attention and feed-forward network.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, 
                 activation: str, layer_norm_eps: float):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer layer.
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class KANPredictor(BaseProsodyPredictor):
    """
    KAN-FKF (Feed-Forward, KAN, Feed-Forward) prosody predictor.
    Uses B-spline KAN blocks for interpretable function approximation.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 kan_config: Dict[str, Any],
                 fkf_config: Dict[str, Any]):
        super().__init__(input_dim, output_dim)
        self.kan_config = kan_config
        self.fkf_config = fkf_config
        self.lambda_smooth = kan_config.get('lambda_smooth', 0.0)
        # Define layers (example composition)
        self.ff1 = nn.Linear(input_dim, fkf_config.get('ff1_dim', hidden_dim))
        self.kan = BsplineKAN(
            in_features=fkf_config.get('ff1_dim', hidden_dim),
            out_features=hidden_dim,
            num_basis=kan_config.get('num_basis', 8),
            degree=kan_config.get('degree', 3),
            use_linear=kan_config.get('use_linear', True),
            dropout=kan_config.get('dropout', 0.1)
        )
        # Set smoothness regularization weight
        self.kan.lambda_smooth = self.lambda_smooth
        self.ff2 = nn.Linear(hidden_dim, fkf_config.get('ff2_dim', hidden_dim))
        self.out_linear = nn.Linear(fkf_config.get('ff2_dim', hidden_dim), output_dim)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the KAN-FKF prosody predictor.
        """
        # Apply initial feed-forward layer
        x = self.ff1(hidden_states)
        
        # CRITICAL FIX: Remove utterance-level prediction
        # Keep sequence dimension for per-phoneme predictions
        # Input should be [batch_size, seq_len, hidden_dim]
        
        # CRITICAL FIX: Handle 3D input for per-phoneme predictions
        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape for KAN processing: [batch_size * seq_len, hidden_dim]
        x_reshaped = x.view(-1, hidden_dim)
        
        # Apply KAN transformation
        kan_out = self.kan(x_reshaped)  # [batch_size * seq_len, hidden_dim]
        
        # Apply final feed-forward layer
        x_processed = self.ff2(kan_out)  # [batch_size * seq_len, hidden_dim]
        
        # Reshape back to sequence: [batch_size, seq_len, hidden_dim]
        x = x_processed.view(batch_size, seq_len, -1)
        
        # Predict individual features (per-phoneme)
        f0 = self.out_linear(x)  # [batch_size, seq_len, 1]
        energy = self.out_linear(x)  # [batch_size, seq_len, 1]
        duration = self.out_linear(x)  # [batch_size, seq_len, 1]
        voice_quality = self.out_linear(x)  # [batch_size, seq_len, 3]
        
        # Combine all features
        prosody_features = torch.cat([f0, energy, duration, voice_quality], dim=-1)
        
        return {
            'prosody_features': prosody_features,
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'voice_quality': voice_quality
        }


class KANFeaturePredictor(nn.Module):
    """
    KAN-based predictor for individual prosody features.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 num_basis: int = 8,
                 degree: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.kan = BsplineKAN(
            in_features=input_dim,
            out_features=output_dim,
            num_basis=num_basis,
            degree=degree,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN feature predictor.
        """
        # CRITICAL FIX: Handle 3D input for per-phoneme predictions
        if x.dim() == 3:
            batch_size, seq_len, input_dim = x.shape
            # Reshape for KAN processing: [batch_size * seq_len, input_dim]
            x_reshaped = x.view(-1, input_dim)
            
            # Apply KAN transformation
            features = self.kan(x_reshaped)  # [batch_size * seq_len, output_dim]
            
            # Apply layer normalization
            features = self.layer_norm(features)
            
            # Reshape back to sequence: [batch_size, seq_len, output_dim]
            features = features.view(batch_size, seq_len, -1)
        else:
            # Handle 2D input (fallback)
            features = self.kan(x)
            features = self.layer_norm(features)
        
        return features


# Factory function for creating prosody predictors
def create_prosody_predictor(predictor_type: str, **kwargs) -> BaseProsodyPredictor:
    """
    Factory function to create prosody predictors based on type.
    
    Args:
        predictor_type: Type of predictor ('mlp', 'transformer', 'kan_fkf')
        **kwargs: Configuration parameters for the predictor
    
    Returns:
        Configured prosody predictor
    """
    if predictor_type == "mlp":
        return MLPPredictor(**kwargs)
    elif predictor_type == "transformer":
        return TransformerPredictor(**kwargs)
    elif predictor_type == "kan_fkf":
        return KANPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}") 