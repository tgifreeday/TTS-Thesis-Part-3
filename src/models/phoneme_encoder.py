"""
Phoneme encoder component for the TTS system.
Encodes phoneme sequences into hidden representations using a Transformer architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

class PhonemeEncoder(nn.Module):
    """
    Encodes phoneme sequences into hidden representations using embedding, positional encoding, and Transformer layers.
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Token embedding layer: maps phoneme IDs to dense vectors
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding: injects sequence order information
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers: capture contextual dependencies between phonemes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization for output stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self,
                phoneme_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the phoneme encoder.
        Args:
            phoneme_ids: Input phoneme IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len] (1=attend, 0=ignore)
            return_attention: Whether to return attention weights
        Returns:
            Dictionary containing:
                - hidden_states: Encoded phoneme representations
                - attention_weights: Optional attention weights
        """
        # Get sequence length
        seq_len = phoneme_ids.size(1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(phoneme_ids)
        
        # Convert attention mask to transformer format
        # 1 for tokens to attend to, 0 for tokens to ignore
        transformer_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        transformer_mask = (1.0 - transformer_mask) * -10000.0
        
        # Get token embeddings
        x = self.token_embedding(phoneme_ids)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        if return_attention:
            # Custom forward pass to get attention weights
            x, attention_weights = self._forward_with_attention(x, transformer_mask)
        else:
            x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
            attention_weights = None
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Prepare output
        output = {
            'hidden_states': x
        }
        
        if return_attention and attention_weights is not None:
            output['attention_weights'] = attention_weights
        
        return output
    
    def _forward_with_attention(self,
                              x: torch.Tensor,
                              mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom forward pass to extract attention weights from each Transformer layer.
        Args:
            x: Input tensor
            mask: Attention mask
        Returns:
            Tuple of (output tensor, attention weights)
        """
        attention_weights = []
        
        # Process through each layer
        for layer in self.transformer.layers:
            # Self attention
            attn_output, attn_weights = layer.self_attn(
                x, x, x,
                key_padding_mask=mask,
                need_weights=True
            )
            attention_weights.append(attn_weights)
            
            # Add & Norm
            x = x + layer.dropout1(attn_output)
            x = layer.norm1(x)
            
            # Feed forward
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x)))
            )
            
            # Add & Norm
            x = x + layer.dropout2(ff_output)
            x = layer.norm2(x)
        
        # Average attention weights across layers
        attention_weights = torch.stack(attention_weights).mean(dim=0)
        
        return x, attention_weights

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Adds sinusoidal position information to input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine and cosine functions to even/odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but moves with the model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
