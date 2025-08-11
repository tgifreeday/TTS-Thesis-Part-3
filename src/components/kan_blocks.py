"""
Core KAN building blocks for the TTS system.

Note on Performance vs. Correctness:
The _compute_basis methods in both BsplineKAN and ConvKAN use Python for loops for clarity and correctness.
While this is not the most performant implementation, it ensures the mathematical operations are correct
and easy to verify. Future optimizations could include:
1. Vectorizing the basis computation using tensor operations
2. Implementing custom CUDA kernels for the B-spline calculations
3. Using torch.jit.script for JIT compilation of the basis computation
4. Pre-computing and caching basis functions for common input ranges

For now, we prioritize correctness over performance to ensure the model learns the intended
mathematical relationships. The current implementation is suitable for research and development.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class BsplineKAN(nn.Module):
    """
    B-spline based Kolmogorov-Arnold Network (KAN) layer.
    This layer applies a learnable B-spline transformation to each input feature, enabling flexible, data-adaptive nonlinearities.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 num_basis: int = 8,
                 degree: int = 3,
                 use_linear: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        self.degree = degree
        self.use_linear = use_linear
        
        # Control points for B-spline [out_features, in_features, num_basis + degree]
        # These are the learnable parameters that define the shape of each B-spline basis function.
        self.control_points = nn.Parameter(
            torch.randn(out_features, in_features, num_basis + degree)
        )
        
        # Linear projection as fallback (optional, for skip connection or residual learning)
        if use_linear:
            self.linear = nn.Linear(in_features, out_features)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for output stability
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Initialize control points and linear weights
        self._initialize_control_points()
        
        # Smoothness regularization weight (set by caller via attribute if desired)
        self.lambda_smooth = 0.0
    
    def _initialize_control_points(self):
        """Initialize control points and linear weights using Xavier initialization for better training stability."""
        nn.init.xavier_uniform_(self.control_points)
        if self.use_linear:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
    
    def smoothness_penalty(self) -> torch.Tensor:
        """Compute L2 penalty on adjacent control-point differences to encourage smooth splines."""
        if self.lambda_smooth <= 0:
            return torch.tensor(0.0, device=self.control_points.device, dtype=self.control_points.dtype)
        diffs = self.control_points[..., 1:] - self.control_points[..., :-1]
        return self.lambda_smooth * (diffs.pow(2).mean())
    
    def _compute_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions for each input feature.
        Args:
            x: Input tensor [batch_size, in_features]
        Returns:
            Basis tensor [batch_size, in_features, num_basis + degree]
        """
        # Ensure x is 2D [batch_size, in_features]
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor [batch_size, in_features], got shape {x.shape}")
        
        # Create knot vector (cached for efficiency)
        if not hasattr(self, 'knots'):
            # Uniformly spaced knots in [0, 1]
            self.register_buffer('knots', torch.linspace(0, 1, self.num_basis + self.degree + 1))
        
        # Initialize basis tensor [batch_size, in_features, num_basis + degree]
        batch_size = x.shape[0]
        basis = torch.zeros(batch_size, self.in_features, self.num_basis + self.degree, device=x.device)
        
        # Compute basis functions for each input feature
        # This is a direct implementation of the Cox-de Boor recursion formula for B-splines
        for i in range(self.in_features):
            # Degree 0 basis functions (piecewise constant)
            for j in range(self.num_basis + self.degree):
                basis[:, i, j] = ((x[:, i] >= self.knots[j]) & 
                                (x[:, i] < self.knots[j + 1])).float()
            
            # Higher degree basis functions (recursively build up)
            for d in range(1, self.degree + 1):
                for j in range(self.num_basis + self.degree - d):
                    # Compute alpha and beta for the recursion
                    denom1 = self.knots[j + d] - self.knots[j]
                    denom2 = self.knots[j + d + 1] - self.knots[j + 1]
                    
                    # Avoid division by zero
                    alpha = torch.where(denom1 != 0, 
                                      (x[:, i] - self.knots[j]) / denom1, 
                                      torch.zeros_like(x[:, i]))
                    beta = torch.where(denom2 != 0, 
                                     (self.knots[j + d + 1] - x[:, i]) / denom2, 
                                     torch.zeros_like(x[:, i]))
                    
                    # Recursively update basis values
                    basis[:, i, j] = alpha * basis[:, i, j] + beta * basis[:, i, j + 1]
        
        return basis
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KAN layer.
        Args:
            x: Input tensor [batch_size, in_features]
        Returns:
            Output tensor [batch_size, out_features]
        """
        # Compute B-spline basis [batch_size, in_features, num_basis + degree]
        basis = self._compute_basis(x)
        
        # Compute spline output using Einstein summation:
        # 'bik,oik->bo' where:
        # b: batch_size, i: in_features, k: num_basis+degree, o: out_features
        # This applies each set of control points to the corresponding basis functions
        spline_out = torch.einsum('bik,oik->bo', basis, self.control_points)
        
        # Add linear projection if enabled (residual connection)
        if self.use_linear:
            linear_out = self.linear(x)
            out = spline_out + linear_out
        else:
            out = spline_out
        
        # Apply dropout and layer normalization for regularization and stability
        out = self.dropout(out)
        out = self.layer_norm(out)
        
        return out

class ConvKAN(nn.Module):
    """
    Convolutional Kolmogorov-Arnold Network layer.
    Applies KAN logic to image patches, enabling spatially-aware nonlinearities.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 num_basis: int = 8,
                 degree: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 use_conv: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_basis = num_basis
        self.degree = degree
        self.stride = stride
        self.padding = padding
        self.use_conv = use_conv
        
        # Control points for each position in the kernel
        # [out_channels, in_channels, kernel_size * kernel_size, num_basis + degree]
        self.control_points = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size * kernel_size, num_basis + degree)
        )
        
        # Standard convolution as fallback
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                 stride=stride, padding=padding)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # Initialize control points
        self._initialize_control_points()
    
    def _initialize_control_points(self):
        """Initialize control points using Xavier initialization."""
        nn.init.xavier_uniform_(self.control_points)
        if self.use_conv:
            nn.init.xavier_uniform_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
    
    def _compute_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions for each input channel and kernel position.
        
        Note: This implementation uses Python for loops for clarity and correctness.
        While not the most performant, it ensures the mathematical operations are correct
        and easy to verify. Future optimizations could include vectorizing these operations.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Basis tensor [batch_size, in_channels, kernel_size * kernel_size, num_basis + degree]
        """
        # Unfold input for convolution
        unfolded = F.unfold(x, kernel_size=self.kernel_size, 
                          stride=self.stride, padding=self.padding)
        
        # Reshape for basis computation
        batch_size = x.shape[0]
        unfolded = unfolded.reshape(batch_size, self.in_channels, -1, 
                                  self.kernel_size * self.kernel_size)
        
        # CRITICAL FIX: Use adaptive normalization instead of hardcoded [-1, 1] assumption
        # Apply sigmoid to map any input distribution to [0, 1] range for B-splines
        unfolded = torch.sigmoid(unfolded)
        
        # Create knot vector (cached)
        if not hasattr(self, 'knots'):
            self.register_buffer('knots', torch.linspace(0, 1, self.num_basis + self.degree + 1))
        
        # Initialize basis tensor
        basis = torch.zeros(batch_size, self.in_channels, 
                          self.kernel_size * self.kernel_size,
                          self.num_basis + self.degree, device=x.device)
        
        # Compute basis functions for each channel and kernel position
        for c in range(self.in_channels):
            for k in range(self.kernel_size * self.kernel_size):
                # Degree 0 basis functions
                for j in range(self.num_basis + self.degree):
                    basis[:, c, k, j] = ((unfolded[:, c, :, k] >= self.knots[j]) & 
                                       (unfolded[:, c, :, k] < self.knots[j + 1])).float()
                
                # Higher degree basis functions
                for d in range(1, self.degree + 1):
                    for j in range(self.num_basis + self.degree - d):
                        # Compute alpha and beta efficiently
                        denom1 = self.knots[j + d] - self.knots[j]
                        denom2 = self.knots[j + d + 1] - self.knots[j + 1]
                        
                        alpha = torch.where(denom1 != 0, 
                                          (unfolded[:, c, :, k] - self.knots[j]) / denom1, 
                                          torch.zeros_like(unfolded[:, c, :, k]))
                        beta = torch.where(denom2 != 0, 
                                         (self.knots[j + d + 1] - unfolded[:, c, :, k]) / denom2, 
                                         torch.zeros_like(unfolded[:, c, :, k]))
                        
                        basis[:, c, k, j] = alpha * basis[:, c, k, j] + beta * basis[:, c, k, j + 1]
        
        return basis
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional KAN layer.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor [batch_size, out_channels, out_height, out_width]
        """
        # Compute B-spline basis
        basis = self._compute_basis(x)
        
        # Compute spline output using correct einsum operation
        # 'bckn,ockn->bco' where:
        # b: batch_size
        # c: in_channels
        # k: kernel_size * kernel_size
        # n: num_basis + degree
        # o: out_channels
        spline_out = torch.einsum('bckn,ockn->bco', basis, self.control_points)
        
        # Reshape to output dimensions
        out_height = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        spline_out = spline_out.reshape(x.shape[0], self.out_channels, out_height, out_width)
        
        # Add standard convolution output if enabled
        if self.use_conv:
            conv_out = self.conv(x)
            out = spline_out + conv_out
        else:
            out = spline_out
        
        # Apply dropout and batch normalization
        out = self.dropout(out)
        out = self.batch_norm(out)
        
        return out

class MultiScaleKAN(nn.Module):
    """Multi-scale feature extraction using ConvKANs with different kernel sizes."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: List[int] = [3, 5, 7],
                 num_basis: int = 8,
                 degree: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        num_kernels = len(kernel_sizes)
        print(f"DEBUG [MultiScaleKAN]: Initializing with out_channels={out_channels}, num_kernels={num_kernels}, kernel_sizes={kernel_sizes}")
        if out_channels % num_kernels != 0:
            print(f"--> ERROR: Mismatch found! {out_channels} is not divisible by {num_kernels}.")
        if out_channels % len(kernel_sizes) != 0:
            raise ValueError("out_channels must be divisible by the number of kernel sizes")
        
        channels_per_conv = out_channels // len(kernel_sizes)
        self.conv_layers = nn.ModuleList([
            ConvKAN(
                in_channels=in_channels,
                out_channels=channels_per_conv,
                kernel_size=k,
                num_basis=num_basis,
                degree=degree,
                padding=(k - 1) // 2,
                dropout=dropout
            )
            for k in kernel_sizes
        ])
        
        # Feature fusion layer
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        self.fusion_norm = nn.BatchNorm2d(out_channels)
        self.fusion_dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale KAN layers."""
        # Process through each scale
        features = [conv(x) for conv in self.conv_layers]
        
        # Concatenate features
        out = torch.cat(features, dim=1)
        
        # Fuse features
        out = self.fusion(out)
        out = self.fusion_norm(out)
        out = self.fusion_dropout(out)
        
        return out
