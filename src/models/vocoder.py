"""
HiFi-GAN vocoder integration for the TTS system.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import os

class HiFiGANVocoder(nn.Module):
    """HiFi-GAN vocoder for converting mel spectrograms to waveform."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define self.device at the beginning of the constructor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained HiFi-GAN model
        self.model = self._load_model()
        
        # If the model loaded successfully, move it to the device and set to eval mode
        if self.model:
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def _load_model(self) -> Optional[nn.Module]:
        """Load pre-trained HiFi-GAN model, returning None if not found."""
        checkpoint_path = self.config.hifigan_checkpoint
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f'HiFi-GAN checkpoint not found at {checkpoint_path}. Skipping vocoder initialization.')
            return None
            
        try:
            # Use map_location='cpu' for initial loading
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Initialize model architecture
            model = HiFiGANGenerator(self.config)
            
            # Load state dict
            model.load_state_dict(checkpoint['generator'])
            
            self.logger.info(f'Successfully loaded HiFi-GAN model from {checkpoint_path}')
            return model
            
        except Exception as e:
            self.logger.error(f'Failed to load HiFi-GAN model: {e}')
            return None
    
    def _init_model(self) -> nn.Module:
        """Initialize HiFi-GAN model architecture."""
        return nn.ModuleDict({
            'generator': nn.ModuleDict({
                'upsample': nn.ModuleList([
                    nn.Sequential(
                        nn.ConvTranspose1d(
                            self.config.hifigan_upsample_initial_channel // (2 ** i),
                            self.config.hifigan_upsample_initial_channel // (2 ** (i + 1)),
                            kernel_size=16,
                            stride=8,
                            padding=4
                        ),
                        nn.LeakyReLU(0.1)
                    )
                    for i in range(3)
                ]),
                'resblocks': nn.ModuleList([
                    ResBlock(
                        self.config.hifigan_upsample_initial_channel // 8,
                        self.config.hifigan_resblock_kernel_sizes,
                        self.config.hifigan_resblock_dilation_sizes
                    )
                    for _ in range(3)
                ]),
                'conv_post': nn.Conv1d(
                    self.config.hifigan_upsample_initial_channel // 8,
                    1,
                    kernel_size=7,
                    stride=1,
                    padding=3
                )
            })
        })
    
    @torch.no_grad()
    def generate(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel spectrogram.
        
        Args:
            mel_spectrogram: Mel spectrogram [batch_size, n_mel, time]
            
        Returns:
            Generated waveform [batch_size, time]
        """
        # Move input to device
        mel_spectrogram = mel_spectrogram.to(self.device)
        
        # Generate waveform
        waveform = self.model['generator'](mel_spectrogram)
        
        return waveform.squeeze(1)
    
    def save_audio(self, waveform: torch.Tensor, path: str):
        """
        Save generated waveform to file.
        
        Args:
            waveform: Generated waveform [batch_size, time]
            path: Output file path
        """
        # Convert to numpy
        waveform = waveform.cpu().numpy()
        
        # Save using torchaudio
        torchaudio.save(
            path,
            torch.from_numpy(waveform).unsqueeze(1),
            self.config.sample_rate
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform."""
        if self.model is None:
            raise RuntimeError("HiFi-GAN model not loaded. Cannot perform forward pass.")
            
        with torch.no_grad():
            return self.model(mel)

class ResBlock(nn.Module):
    """Residual block for HiFi-GAN."""
    def __init__(self, channels: int, kernel_sizes: Tuple[int, ...], 
                 dilation_sizes: Tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=k,
                    dilation=d,
                    padding=(k * d) // 2
                ),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=1)
            )
            for k, d in zip(kernel_sizes, dilation_sizes)
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=k,
                    dilation=d,
                    padding=(k * d) // 2
                ),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=1)
            )
            for k, d in zip(kernel_sizes, dilation_sizes)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x)
            xt = c2(xt)
            x = xt + x
        return x 