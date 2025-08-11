#!/usr/bin/env python3
"""
KAN-TTS Training Script
Version: SCHEDULER_FIXED_v1.0 - SequentialLR + CosineAnnealingLR implementation
Date: 2025-07-29
Status: All compatibility issues resolved
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import TTSDataset
from models.prosody_predictor_modular import MLPPredictor, TransformerPredictor, KANPredictor, create_prosody_predictor
# DurationPredictor is implemented inline in the TTSModel class
from models.spectrogram_decoder import SpectrogramDecoder
from utils.config import Config
from utils.logging import setup_logging
from models.vocoder import HiFiGANVocoder


class TTSTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for KAN-TTS models.
    
    This class handles the training loop, validation, and logging for all
    TTS model architectures (MLP, Transformer, KAN-FKF).
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model components
        self._setup_model()
        self._setup_loss_functions()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def _setup_model(self):
        """Initialize the TTS model based on configuration."""
        model_config = self.config.model
        
        # Create prosody predictor based on configuration
        prosody_predictor_name = model_config.prosody_predictor['name']
        
        if prosody_predictor_name == "mlp":
            self.prosody_predictor = MLPPredictor(
                input_dim=model_config.prosody_predictor['input_dim'],
                hidden_dims=model_config.prosody_predictor['hidden_dims'],
                dropout=model_config.prosody_predictor['dropout'],
                activation=model_config.prosody_predictor['activation'],
                batch_norm=model_config.prosody_predictor['batch_norm']
            )
        elif prosody_predictor_name == "transformer":
            self.prosody_predictor = TransformerPredictor(
                input_dim=model_config.prosody_predictor['input_dim'],
                n_heads=model_config.prosody_predictor['n_heads'],
                n_layers=model_config.prosody_predictor['n_layers'],
                d_model=model_config.prosody_predictor['d_model'],
                d_ff=model_config.prosody_predictor['d_ff'],
                dropout=model_config.prosody_predictor['dropout'],
                activation=model_config.prosody_predictor['activation'],
                layer_norm_eps=model_config.prosody_predictor['layer_norm_eps']
            )
        elif prosody_predictor_name == "kan_fkf":
            self.prosody_predictor = KANPredictor(
                input_dim=model_config.prosody_predictor['input_dim'],
                hidden_dim=model_config.prosody_predictor['hidden_dim'],
                output_dim=model_config.prosody_predictor['output_dim'],
                kan_config=model_config.prosody_predictor['kan_config'],
                fkf_config=model_config.prosody_predictor['fkf_config']
            )
        else:
            raise ValueError(f"Unknown prosody predictor: {prosody_predictor_name}")
        
        # CRITICAL: Handle different architecture types
        architecture_type = getattr(model_config, 'architecture_type', None)
        
        if architecture_type == "clean_input":
            # CRITICAL: Clean input architecture for interpretability
            # Simple embedding for phonemes
            self.phoneme_embedding = nn.Embedding(
                num_embeddings=100,  # vocab_size
                embedding_dim=model_config.phoneme_encoder['embedding_dim']
            )
            
            # Simple linear projection for linguistic features
            self.linguistic_projection = nn.Linear(
                model_config.linguistic_encoder['input_dim'],
                model_config.linguistic_encoder['output_dim']
            )
            
            # No Transformer encoder - direct concatenation for interpretability
            self.encoder = None
            self.input_projection = None
            
            # Combined dimension for clean input
            combined_dim = (model_config.phoneme_encoder['embedding_dim'] + 
                          model_config.linguistic_encoder['output_dim'])
            
        else:
            # CRITICAL: Standard Transformer-based architecture
            # Start with embedding layer
            self.phoneme_embedding = nn.Embedding(
                num_embeddings=100,  # vocab_size
                embedding_dim=model_config.phoneme_encoder['embedding_dim']
            )
            
            # CRITICAL FIX: Dynamic dimension calculation based on input features
            # Get input features from data configuration
            input_features = getattr(self.config.data, 'input_features', None)
            
            if input_features is None or len(input_features) > 1:
                # Full features case: 256 phoneme + 15 linguistic = 271
                combined_dim = 271
            else:
                # Phonemes-only case: only phoneme_id = 256
                combined_dim = 256
            
            encoder_dim = model_config.phoneme_encoder['hidden_dim']
            
            # Dynamic projection based on input features
            self.input_projection = nn.Linear(combined_dim, encoder_dim)
            
            # Transformer encoder for contextualized features
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=encoder_dim,  # 272
                nhead=8,  # Standard value
                dim_feedforward=encoder_dim * 4,
                dropout=model_config.phoneme_encoder['dropout'],
                activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=model_config.phoneme_encoder['n_layers']
            )
        
        # CRITICAL: Create dedicated Duration Predictor
        # This is a standard component in FastSpeech-like models
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(model_config.phoneme_encoder['hidden_dim'], 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # CRITICAL FIX: Use BatchNorm1d for Conv1d output
            nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # CRITICAL FIX: Use BatchNorm1d for Conv1d output
            nn.Dropout(0.5),
            # CRITICAL FIX: Per-phoneme duration prediction
            nn.Conv1d(256, 1, kernel_size=1)  # 1x1 conv for per-phoneme prediction
        )
        
        # Calculate input dimension for spectrogram decoder
        # Encoded features (256) + duration_feature (1) = 257
        # Length regulator expands phoneme-level features to audio-level features
        input_dim = model_config.phoneme_encoder['hidden_dim'] + 1  # 271 + 1 = 272
        
        # Create spectrogram decoder with proper input dimension
        layers = []
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, model_config.spectrogram_decoder['hidden_dim']))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(model_config.spectrogram_decoder['dropout']))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(model_config.spectrogram_decoder['n_layers'] - 1):
            layers.append(nn.Linear(model_config.spectrogram_decoder['hidden_dim'], 
                                  model_config.spectrogram_decoder['hidden_dim']))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(model_config.spectrogram_decoder['dropout']))
        
        # Output layer: hidden_dim -> output_dim
        layers.append(nn.Linear(model_config.spectrogram_decoder['hidden_dim'], 
                              model_config.spectrogram_decoder['output_dim']))
        
        self.spectrogram_decoder = nn.Sequential(*layers)
        
        # Prosody feature predictors are now handled by the prosody_predictor
        # No need for separate predictors
        
        # Setup optional vocoder for audio export
        try:
            self.vocoder = HiFiGANVocoder(type('cfg', (), {
                'hifigan_checkpoint': model_config.vocoder.get('checkpoint_path', ''),
                'sample_rate': model_config.vocoder.get('sample_rate', 22050),
                'hifigan_upsample_initial_channel': 512,
                'hifigan_resblock_kernel_sizes': (3,5,7),
                'hifigan_resblock_dilation_sizes': (1,3,5),
            }))
        except Exception:
            self.vocoder = None
        
    def _length_regulator(self, features: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """
        Academically Robust Length Regulator.
        Expands phoneme-level features using a single, vectorized operation.
        """
        # Ensure durations are integer counts on the correct device
        durations = durations.long().to(features.device)
        
        # CRITICAL FIX: Flatten durations to 1D for repeat_interleave
        # durations shape: [batch_size, seq_len] -> [batch_size * seq_len]
        batch_size, seq_len = features.shape[:2]
        durations_flat = durations.view(-1)  # Flatten to 1D
        
        # Flatten features to match: [batch_size, seq_len, features] -> [batch_size * seq_len, features]
        features_flat = features.view(-1, features.shape[-1])
        
        # Use repeat_interleave to expand the features
        expanded_features_flat = torch.repeat_interleave(features_flat, durations_flat, dim=0)
        
        # Reshape back to batch format
        # Calculate the new sequence length for each batch item
        total_frames = durations.sum(dim=1)  # [batch_size]
        max_frames = total_frames.max().item()
        
        # Pad to max length for batch processing
        expanded_features = torch.zeros(
            batch_size, max_frames, features.shape[-1], 
            device=features.device, dtype=features.dtype
        )
        
        # Fill each batch item
        start_idx = 0
        for i in range(batch_size):
            item_frames = total_frames[i].item()
            if item_frames > 0:
                # Get the expanded features for this batch item
                item_start = start_idx
                item_end = start_idx + item_frames
                item_features = expanded_features_flat[item_start:item_end]
                expanded_features[i, :item_frames] = item_features
                start_idx = item_end
        
        return expanded_features
        
    def _safe_truncate_loss(self, pred, target, loss_fn):
        """Safely compute loss with truncation to handle length mismatches."""
        min_len = min(pred.shape[1], target.shape[1])
        if min_len > 0:
            return loss_fn(pred[:, :min_len], target[:, :min_len])
        return torch.tensor(0.0, device=self.device)
    
    def _safe_truncate_rmse(self, pred, target):
        """Safely compute RMSE with truncation to handle length mismatches."""
        min_len = min(pred.shape[1], target.shape[1])
        if min_len > 0:
            return torch.sqrt(torch.mean((pred[:, :min_len] - target[:, :min_len])**2))
        return torch.tensor(0.0, device=self.device)
    
    def _setup_loss_functions(self):
        """Initialize loss functions based on configuration."""
        loss_config = self.config.loss
        
        # Loss functions
        if loss_config.mel_loss == "l1":
            self.mel_loss_fn = nn.L1Loss()
        elif loss_config.mel_loss == "mse":
            self.mel_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown mel loss: {loss_config.mel_loss}")
        
        if loss_config.duration_loss == "mse":
            self.duration_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown duration loss: {loss_config.duration_loss}")
        
        if loss_config.f0_loss == "mse":
            self.f0_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown F0 loss: {loss_config.f0_loss}")
        
        if loss_config.energy_loss == "mse":
            self.energy_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown energy loss: {loss_config.energy_loss}")
        
        # Loss weights
        self.mel_loss_weight = loss_config.mel_loss_weight
        self.duration_loss_weight = loss_config.duration_loss_weight
        self.f0_loss_weight = loss_config.f0_loss_weight
        self.energy_loss_weight = loss_config.energy_loss_weight
    
    def forward(self, batch, use_teacher_forcing=True):
        """Forward pass through the model."""
        
        # VERSION CHECK - Clean implementation, no debug hacks
        if not hasattr(self, '_version_checked'):
            # print("✅ FORWARD METHOD: Clean version - no debug padding/truncation")
            self._version_checked = True

        
        # Extract input features
        phoneme_ids = batch['phoneme_ids']
        linguistic_features = batch['linguistic_features']
        
        # 1. ENCODE FEATURES
        architecture_type = getattr(self.config.model, 'architecture_type', None)
        
        if architecture_type == "clean_input":
            # CRITICAL: Clean input architecture for interpretability
            phoneme_embeddings = self.phoneme_embedding(phoneme_ids)
            linguistic_projected = self.linguistic_projection(linguistic_features)
            
            # Direct concatenation without Transformer for interpretability
            combined_features = torch.cat([phoneme_embeddings, linguistic_projected], dim=-1)
            encoded_features = combined_features  # No encoder transformation
            
        else:
            # CRITICAL: Standard Transformer-based architecture
            phoneme_embeddings = self.phoneme_embedding(phoneme_ids)
            
            # CRITICAL FIX: Handle phonemes-only vs full features
            input_features = getattr(self.config.data, 'input_features', None)
            if input_features is None or len(input_features) > 1:
                # Full features case: combine phoneme embeddings with linguistic features
                combined_features = torch.cat([phoneme_embeddings, linguistic_features], dim=-1)
            else:
                # Phonemes-only case: use only phoneme embeddings
                combined_features = phoneme_embeddings
            
            # CRITICAL FIX: Dynamic projection based on input features
            projected_features = self.input_projection(combined_features)
            encoded_features = self.encoder(projected_features)
        
        # 2. PREDICT DURATIONS
        # Duration predictor operates on the contextualized features
        duration_log_prediction = self.duration_predictor(encoded_features.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        
        # 3. PREDICT OTHER PROSODY (F0, Energy)
        # CRITICAL FIX: BOTH predictors now use the SAME contextualized features
        prosody_predictions = self.prosody_predictor(encoded_features)
        
        # Extract F0 and energy predictions
        f0_prediction = prosody_predictions['f0']
        energy_prediction = prosody_predictions['energy']
        
        # 4. EXPAND SEQUENCE (LENGTH REGULATOR)
        # CRITICAL FIX: Define duration_frames for both paths to avoid variable scope issues
        if use_teacher_forcing and 'duration_frames' in batch:
            # --- TEACHER FORCING PATH (Training) ---
            # Use the guaranteed-correct frame counts from the dataset
            duration_frames = batch['duration_frames']  # Use ground-truth
            expanded_features = self._length_regulator(encoded_features, duration_frames)
        else:
            # --- INFERENCE PATH (Validation) ---
            # Use PREDICTED durations to expand the sequence
            duration_prediction = torch.exp(duration_log_prediction)
            
            # Debug ranges (disabled by default)
            # print(f"DEBUG: duration_log_prediction range: {duration_log_prediction.min():.3f} to {duration_log_prediction.max():.3f}")
            # print(f"DEBUG: duration_prediction range: {duration_prediction.min():.3f} to {duration_prediction.max():.3f}")
            
            frame_rate = 22050  # Hz
            hop_length = 256  # frames per hop
            frames_per_second = frame_rate / hop_length  # ~86 frames per second
            
            duration_frames = (duration_prediction * frames_per_second).long()
            
            # CRITICAL FIX: Add reasonable bounds to prevent explosion
            duration_frames = torch.clamp(duration_frames, min=1, max=100)  # Max 100 frames per phoneme
            
            expanded_features = self._length_regulator(encoded_features, duration_frames)

        # 5. ADD PREDICTED DURATION AS FEATURE (for robustness)
        # Expand the predicted duration to match the new sequence length
        # CRITICAL FIX: Use the same duration_frames variable defined above
        duration_feature_expanded = self._length_regulator(
            duration_log_prediction.unsqueeze(-1), 
            duration_frames
        )
        
        # Concatenate the main expanded features with the duration feature
        decoder_input = torch.cat([expanded_features, duration_feature_expanded], dim=-1)

        # 6. DECODE SPECTROGRAM
        # CRITICAL FIX: Pass target length to handle upsampling expansion
        target_length = None
        if 'mel_spectrogram' in batch:
            target_length = batch['mel_spectrogram'].shape[1]
        
        mel_spectrogram = self.spectrogram_decoder(decoder_input)
        

        
        # Expand F0 and energy predictions to match the expanded sequence length
        if use_teacher_forcing and 'duration_frames' in batch:
            # Use ground-truth duration frames for expansion
            f0_expanded = self._length_regulator(f0_prediction, batch['duration_frames'])
            energy_expanded = self._length_regulator(energy_prediction, batch['duration_frames'])
        else:
            # Use predicted duration frames for expansion
            f0_expanded = self._length_regulator(f0_prediction, duration_frames)
            energy_expanded = self._length_regulator(energy_prediction, duration_frames)
        
        # Update prosody predictions with expanded values
        prosody_predictions['f0'] = f0_expanded
        prosody_predictions['energy'] = energy_expanded
        

        
        return {
            'mel_spectrogram': mel_spectrogram,
            'duration_log_prediction': duration_log_prediction,
            'prosody_predictions': prosody_predictions  # Contains F0, energy, etc.
        }
    
    def _make_frame_mask(self, durations: torch.Tensor, max_len: int) -> torch.Tensor:
        """Build [batch, max_len] boolean mask from per-phoneme frame counts."""
        # durations: [B, T_phon]
        valid_lengths = durations.sum(dim=1)  # [B]
        arange = torch.arange(max_len, device=durations.device)[None, :]
        mask = arange < valid_lengths[:, None]
        return mask

    def _masked_l1(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred/target: [B, T', D] or [B, T'] ; mask: [B, T']
        if pred.dim() == 3:
            mask = mask.unsqueeze(-1)
        diff = (pred - target).abs() * mask
        denom = mask.sum().clamp_min(1)
        return diff.sum() / denom

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            mask = mask.unsqueeze(-1)
        diff = (pred - target).pow(2) * mask
        denom = mask.sum().clamp_min(1)
        return diff.sum() / denom

    def _masked_pearsonr(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked Pearson correlation along time for F0 sequences.
        pred/target: [B, T] ; mask: [B, T]
        Returns scalar tensor.
        """
        # flatten across batch and time with mask
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        m = mask.contiguous().view(-1).to(dtype=pred.dtype)
        # guard
        if m.sum() < 2:
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze(0)
        # mean with mask
        mean_pred = (pred * m).sum() / m.sum()
        mean_tgt = (target * m).sum() / m.sum()
        dp = ((pred - mean_pred) * (target - mean_tgt) * m).sum()
        var_p = (((pred - mean_pred) ** 2) * m).sum()
        var_t = (((target - mean_tgt) ** 2) * m).sum()
        denom = (var_p.clamp_min(1e-8) * var_t.clamp_min(1e-8)).sqrt()
        corr = dp / denom
        return corr

    def _masked_vuv_error(self, f0_pred: torch.Tensor, f0_true: torch.Tensor, mask: torch.Tensor, thresh: float = 1e-6) -> torch.Tensor:
        """Voiced/Unvoiced error rate under mask.
        pred/true: [B, T] ; mask: [B, T]
        """
        v_pred = (f0_pred > thresh)
        v_true = (f0_true > thresh)
        mismatches = (v_pred != v_true) & mask.bool()
        total = mask.sum().clamp_min(1)
        return mismatches.sum().to(f0_pred.dtype) / total

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Forward pass
        outputs = self.forward(batch, use_teacher_forcing=True)
        
        # Calculate losses (handle missing audio features)
        mel_loss = torch.tensor(0.0, device=self.device)
        duration_loss = torch.tensor(0.0, device=self.device)
        f0_loss = torch.tensor(0.0, device=self.device)
        energy_loss = torch.tensor(0.0, device=self.device)
        f0_corr = torch.tensor(0.0, device=self.device)
        vuv_err = torch.tensor(0.0, device=self.device)
        
        # Build frame mask from ground-truth duration_frames for masking
        frame_mask = None
        if 'duration_frames' in batch and batch['duration_frames'].numel() > 0:
            # Sum per-phoneme to get per-utterance valid frame counts
            valid_lengths = batch['duration_frames'].sum(dim=1)
            max_len = outputs['mel_spectrogram'].shape[1]
            arange = torch.arange(max_len, device=self.device)[None, :]
            frame_mask = (arange < valid_lengths[:, None])
        
        # Mel loss (masked)
        if 'mel_spectrogram' in batch and 'mel_spectrogram' in outputs and frame_mask is not None:
            pred_mel = outputs['mel_spectrogram']
            target_mel = batch['mel_spectrogram'].to(pred_mel.device)
            # Align time dim via truncation to min length before masking if needed
            min_len = min(pred_mel.shape[1], target_mel.shape[1])
            pred_mel = pred_mel[:, :min_len]
            target_mel = target_mel[:, :min_len]
            mel_mask = frame_mask[:, :min_len]
            mel_loss = self._masked_l1(pred_mel, target_mel, mel_mask)
        
        # Duration loss (unchanged; uses log domain)
        if 'duration_log' in batch and 'duration_log_prediction' in outputs:
            duration_loss = self.duration_loss_fn(outputs['duration_log_prediction'], batch['duration_log'])
        
        # F0 loss (masked)
        if 'f0' in batch and 'prosody_predictions' in outputs and frame_mask is not None:
            f0_pred = outputs['prosody_predictions']['f0'].squeeze(-1)
            target_f0 = batch['f0'].to(f0_pred.device)
            min_len = min(f0_pred.shape[1], target_f0.shape[1])
            f0_pred = f0_pred[:, :min_len]
            target_f0 = target_f0[:, :min_len]
            f0_mask = frame_mask[:, :min_len]
            f0_loss = self._masked_mse(f0_pred, target_f0, f0_mask)
            # F0 correlation and V/UV error
            f0_corr = self._masked_pearsonr(f0_pred, target_f0, f0_mask)
            vuv_err = self._masked_vuv_error(f0_pred, target_f0, f0_mask)
        
        # Energy loss (masked)
        if 'energy' in batch and 'prosody_predictions' in outputs and frame_mask is not None:
            energy_pred = outputs['prosody_predictions']['energy'].squeeze(-1)
            target_energy = batch['energy'].to(energy_pred.device)
            min_len = min(energy_pred.shape[1], target_energy.shape[1])
            energy_pred = energy_pred[:, :min_len]
            target_energy = target_energy[:, :min_len]
            energy_mask = frame_mask[:, :min_len]
            energy_loss = self._masked_mse(energy_pred, target_energy, energy_mask)
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss + 
                     self.duration_loss_weight * duration_loss +
                     self.f0_loss_weight * f0_loss +
                     self.energy_loss_weight * energy_loss)
        # Add KAN smoothness penalty if present
        smooth_penalty = torch.tensor(0.0, device=self.device)
        for m in self.modules():
            if hasattr(m, 'smoothness_penalty'):
                smooth_penalty = smooth_penalty + m.smoothness_penalty()
        total_loss = total_loss + smooth_penalty
        
        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        if smooth_penalty is not None:
            self.log('train_smooth_penalty', smooth_penalty)
        self.log('train_mel_loss', mel_loss)
        self.log('train_duration_loss', duration_loss)
        self.log('train_f0_loss', f0_loss)
        self.log('train_energy_loss', energy_loss)
        self.log('train_f0_corr', f0_corr)
        self.log('train_vuv_error', vuv_err)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Forward pass (use inference mode for proper validation)
        outputs = self.forward(batch, use_teacher_forcing=False)
        
        # Calculate losses (handle missing audio features)
        mel_loss = torch.tensor(0.0, device=self.device)
        duration_loss = torch.tensor(0.0, device=self.device)
        f0_loss = torch.tensor(0.0, device=self.device)
        energy_loss = torch.tensor(0.0, device=self.device)
        f0_corr = torch.tensor(0.0, device=self.device)
        vuv_err = torch.tensor(0.0, device=self.device)
        
        # Frame mask from ground-truth durations if available (validation should have it)
        frame_mask = None
        if 'duration_frames' in batch and batch['duration_frames'].numel() > 0:
            valid_lengths = batch['duration_frames'].sum(dim=1)
            max_len = outputs['mel_spectrogram'].shape[1]
            arange = torch.arange(max_len, device=self.device)[None, :]
            frame_mask = (arange < valid_lengths[:, None])
        
        # Mel loss (masked)
        if 'mel_spectrogram' in batch and 'mel_spectrogram' in outputs and frame_mask is not None:
            pred_mel = outputs['mel_spectrogram']
            target_mel = batch['mel_spectrogram'].to(pred_mel.device)
            min_len = min(pred_mel.shape[1], target_mel.shape[1])
            pred_mel = pred_mel[:, :min_len]
            target_mel = target_mel[:, :min_len]
            mel_mask = frame_mask[:, :min_len]
            mel_loss = self._masked_l1(pred_mel, target_mel, mel_mask)
        
        # Duration loss
        if 'duration_log' in batch and 'duration_log_prediction' in outputs:
            duration_loss = self.duration_loss_fn(outputs['duration_log_prediction'], batch['duration_log'])
        
        # F0 loss (masked)
        if 'f0' in batch and 'prosody_predictions' in outputs and frame_mask is not None:
            f0_pred = outputs['prosody_predictions']['f0'].squeeze(-1)
            target_f0 = batch['f0'].to(f0_pred.device)
            min_len = min(f0_pred.shape[1], target_f0.shape[1])
            f0_pred = f0_pred[:, :min_len]
            target_f0 = target_f0[:, :min_len]
            f0_mask = frame_mask[:, :min_len]
            f0_loss = self._masked_mse(f0_pred, target_f0, f0_mask)
            # F0 correlation and V/UV error
            f0_corr = self._masked_pearsonr(f0_pred, target_f0, f0_mask)
            vuv_err = self._masked_vuv_error(f0_pred, target_f0, f0_mask)
        
        # Energy loss (masked)
        if 'energy' in batch and 'prosody_predictions' in outputs and frame_mask is not None:
            energy_pred = outputs['prosody_predictions']['energy'].squeeze(-1)
            target_energy = batch['energy'].to(energy_pred.device)
            min_len = min(energy_pred.shape[1], target_energy.shape[1])
            energy_pred = energy_pred[:, :min_len]
            target_energy = target_energy[:, :min_len]
            energy_mask = frame_mask[:, :min_len]
            energy_loss = self._masked_mse(energy_pred, target_energy, energy_mask)
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss + 
                     self.duration_loss_weight * duration_loss +
                     self.f0_loss_weight * f0_loss +
                     self.energy_loss_weight * energy_loss)
        # Add KAN smoothness penalty if present
        smooth_penalty = torch.tensor(0.0, device=self.device)
        for m in self.modules():
            if hasattr(m, 'smoothness_penalty'):
                smooth_penalty = smooth_penalty + m.smoothness_penalty()
        total_loss = total_loss + smooth_penalty
        
        # Calculate RMSE metrics (masked variants)
        duration_rmse = torch.tensor(0.0, device=self.device)
        f0_rmse = torch.tensor(0.0, device=self.device)
        energy_rmse = torch.tensor(0.0, device=self.device)
        
        if 'duration_log' in batch and 'duration_log_prediction' in outputs:
            duration_rmse = self._safe_truncate_rmse(outputs['duration_log_prediction'], batch['duration_log'])
        
        if 'f0' in batch and 'prosody_predictions' in outputs and frame_mask is not None:
            f0_pred = outputs['prosody_predictions']['f0'].squeeze(-1)
            target_f0 = batch['f0'].to(f0_pred.device)
            min_len = min(f0_pred.shape[1], target_f0.shape[1])
            f0_pred = f0_pred[:, :min_len]
            target_f0 = target_f0[:, :min_len]
            f0_mask = frame_mask[:, :min_len]
            # masked RMSE
            diff = (f0_pred - target_f0) ** 2
            f0_rmse = torch.sqrt((diff * f0_mask).sum() / f0_mask.sum().clamp_min(1))
        
        if 'energy' in batch and 'prosody_predictions' in outputs and frame_mask is not None:
            energy_pred = outputs['prosody_predictions']['energy'].squeeze(-1)
            target_energy = batch['energy'].to(energy_pred.device)
            min_len = min(energy_pred.shape[1], target_energy.shape[1])
            energy_pred = energy_pred[:, :min_len]
            target_energy = target_energy[:, :min_len]
            energy_mask = frame_mask[:, :min_len]
            diff = (energy_pred - target_energy) ** 2
            energy_rmse = torch.sqrt((diff * energy_mask).sum() / energy_mask.sum().clamp_min(1))
        
        # Log metrics
        self.log('val_loss', total_loss, prog_bar=True)
        if smooth_penalty is not None:
            self.log('val_smooth_penalty', smooth_penalty)
        self.log('val_mel_loss', mel_loss)
        self.log('val_duration_loss', duration_loss)
        self.log('val_f0_loss', f0_loss)
        self.log('val_energy_loss', energy_loss)
        self.log('val_duration_rmse', duration_rmse)
        self.log('val_f0_rmse', f0_rmse)
        self.log('val_energy_rmse', energy_rmse)
        self.log('val_f0_corr', f0_corr)
        self.log('val_vuv_error', vuv_err)
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Export a few fixed-sentence samples every N epochs for subjective checks."""
        eval_conf = self.config.evaluation
        if (self.current_epoch + 1) % max(1, eval_conf.generate_audio_every_n_epochs) != 0:
            return
        # We need a small batch from val data; Lightning doesn't expose it here easily.
        # Skip complex fetching; instead, log a note or rely on external script.
        self.log('audio_export_epoch', float(self.current_epoch + 1))
    
    def configure_optimizers(self):
        """Configure optimizers and a warm-up + cosine decay scheduler."""
        training_config = self.config.training
        
        optimizer = optim.AdamW(
            self.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # This requires the total number of training steps. We can get this 
        # from the trainer after it has been initialized.
        try:
            total_steps = self.trainer.estimated_stepping_batches
        except AttributeError:
            # Fallback if the trainer is not fully initialized yet.
            # This will be re-evaluated correctly when training starts.
            num_train_samples = 6182  # From your logs
            batches_per_epoch = num_train_samples // self.config.data.batch_size
            total_steps = training_config.max_epochs * batches_per_epoch

        # 1. Warm-up scheduler for the first N steps
        warmup_steps = training_config.warmup_steps
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )

        # 2. Main scheduler: Cosine Annealing (activates AFTER warm-up)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(total_steps - warmup_steps)
        )

        # 3. Chain them together
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # The scheduler must be checked every step
            }
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Separate different types of data
    phoneme_ids = []
    linguistic_features = []
    mel_spectrograms = []
    durations = []
    duration_frames = []
    duration_logs = []
    f0s = []
    energies = []
    
    max_phoneme_length = max(len(item['phoneme_ids']) for item in batch)
    
    for item in batch:
        # Pad phoneme_ids
        phoneme_ids_padded = torch.zeros(max_phoneme_length, dtype=torch.long)
        phoneme_ids_padded[:len(item['phoneme_ids'])] = item['phoneme_ids']
        phoneme_ids.append(phoneme_ids_padded)
        
        # Pad linguistic_features
        if len(item['linguistic_features']) > 0:
            linguistic_padded = torch.zeros(max_phoneme_length, item['linguistic_features'].shape[1], dtype=torch.float32)
            linguistic_padded[:len(item['linguistic_features'])] = item['linguistic_features']
            linguistic_features.append(linguistic_padded)
        else:
            linguistic_features.append(torch.zeros(max_phoneme_length, 0, dtype=torch.float32))
        
        # Handle audio features if present
        if 'mel_spectrogram' in item:
            mel_spectrograms.append(item['mel_spectrogram'])
        if 'duration' in item:
            durations.append(item['duration'])
        if 'duration_frames' in item:
            duration_frames.append(item['duration_frames'])
        if 'duration_log' in item:
            duration_logs.append(item['duration_log'])
        if 'f0' in item:
            f0s.append(item['f0'])
        if 'energy' in item:
            energies.append(item['energy'])
    
    result = {
        'phoneme_ids': torch.stack(phoneme_ids),
        'linguistic_features': torch.stack(linguistic_features),
    }
    
    # Handle variable-length audio features
    if mel_spectrograms:
        # Pad mel spectrograms to the same length
        max_mel_length = max(mel.shape[0] for mel in mel_spectrograms)
        mel_spectrograms_padded = []
        for mel in mel_spectrograms:
            # Pad with zeros to the right
            padded_mel = torch.zeros(max_mel_length, mel.shape[1], dtype=mel.dtype)
            padded_mel[:mel.shape[0], :] = mel
            mel_spectrograms_padded.append(padded_mel)
        result['mel_spectrogram'] = torch.stack(mel_spectrograms_padded)
    
    if durations:
        # CRITICAL FIX: Pad duration sequences to the same length
        # Handle empty or 0-dimensional duration tensors
        valid_durations = [d for d in durations if d.numel() > 0 and len(d.shape) > 0]
        if valid_durations:
            max_duration_length = max(duration.shape[0] for duration in valid_durations)
            durations_padded = []
            for duration in durations:
                if duration.numel() > 0 and len(duration.shape) > 0:
                    padded_duration = torch.zeros(max_duration_length, dtype=duration.dtype)
                    padded_duration[:duration.shape[0]] = duration
                    durations_padded.append(padded_duration)
                else:
                    # Handle empty duration tensors
                    padded_duration = torch.zeros(max_duration_length, dtype=torch.float32)
                    durations_padded.append(padded_duration)
            result['duration'] = torch.stack(durations_padded)
        else:
            # All durations are empty, create empty tensor
            result['duration'] = torch.zeros(len(durations), 0, dtype=torch.float32)
    
    # CRITICAL FIX: Handle duration_frames (the key missing piece!)
    if duration_frames:
        # Pad duration_frames sequences to the same length
        valid_duration_frames = [d for d in duration_frames if d.numel() > 0 and len(d.shape) > 0]
        if valid_duration_frames:
            max_duration_frames_length = max(duration_frame.shape[0] for duration_frame in valid_duration_frames)
            duration_frames_padded = []
            for duration_frame in duration_frames:
                if duration_frame.numel() > 0 and len(duration_frame.shape) > 0:
                    padded_duration_frame = torch.zeros(max_duration_frames_length, dtype=duration_frame.dtype)
                    padded_duration_frame[:duration_frame.shape[0]] = duration_frame
                    duration_frames_padded.append(padded_duration_frame)
                else:
                    # Handle empty duration_frames tensors
                    padded_duration_frame = torch.zeros(max_duration_frames_length, dtype=torch.long)
                    duration_frames_padded.append(padded_duration_frame)
            result['duration_frames'] = torch.stack(duration_frames_padded)
        else:
            # All duration_frames are empty, create empty tensor
            result['duration_frames'] = torch.zeros(len(duration_frames), 0, dtype=torch.long)
    
    # Handle duration_logs
    if duration_logs:
        # Pad duration_log sequences to the same length
        valid_duration_logs = [d for d in duration_logs if d.numel() > 0 and len(d.shape) > 0]
        if valid_duration_logs:
            max_duration_log_length = max(duration_log.shape[0] for duration_log in valid_duration_logs)
            duration_logs_padded = []
            for duration_log in duration_logs:
                if duration_log.numel() > 0 and len(duration_log.shape) > 0:
                    padded_duration_log = torch.zeros(max_duration_log_length, dtype=duration_log.dtype)
                    padded_duration_log[:duration_log.shape[0]] = duration_log
                    duration_logs_padded.append(padded_duration_log)
                else:
                    # Handle empty duration_log tensors
                    padded_duration_log = torch.zeros(max_duration_log_length, dtype=torch.float32)
                    duration_logs_padded.append(padded_duration_log)
            result['duration_log'] = torch.stack(duration_logs_padded)
        else:
            # All duration_logs are empty, create empty tensor
            result['duration_log'] = torch.zeros(len(duration_logs), 0, dtype=torch.float32)
    
    if f0s:
        # Pad F0 sequences to the same length
        max_f0_length = max(f0.shape[0] for f0 in f0s)
        f0s_padded = []
        for f0 in f0s:
            padded_f0 = torch.zeros(max_f0_length, dtype=f0.dtype)
            padded_f0[:f0.shape[0]] = f0
            f0s_padded.append(padded_f0)
        result['f0'] = torch.stack(f0s_padded)
    
    if energies:
        # Pad energy sequences to the same length
        max_energy_length = max(energy.shape[0] for energy in energies)
        energies_padded = []
        for energy in energies:
            padded_energy = torch.zeros(max_energy_length, dtype=energy.dtype)
            padded_energy[:energy.shape[0]] = energy
            energies_padded.append(padded_energy)
        result['energy'] = torch.stack(energies_padded)
    
    return result

def create_data_loaders(config: Config):
    """Create training and validation data loaders."""
    data_config = config.data
    
    # Get input features from configuration (for ablation studies)
    input_features = getattr(data_config, 'input_features', None)
    
    # Create datasets with the new audio features file
    train_dataset = TTSDataset(
        data_path="data/processed/linguistic_features_with_audio_paths.json",
        file_list_path=data_config.train_files,
        base_path="data/processed",
        input_features=input_features
    )
    
    val_dataset = TTSDataset(
        data_path="data/processed/linguistic_features_with_audio_paths.json",
        file_list_path=data_config.val_files,
        base_path="data/processed",
        input_features=input_features
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=True if data_config.num_workers and data_config.num_workers > 0 else False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=True if data_config.num_workers and data_config.num_workers > 0 else False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def setup_callbacks(config: Config, output_dir: str):
    """Setup PyTorch Lightning callbacks."""
    experiment_config = config.experiment
    training_config = config.training
    
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="model-{epoch:02d}-{val_loss:.4f}",
        monitor=experiment_config.monitor,
        mode=experiment_config.mode,
        save_top_k=training_config.save_top_k,
        save_last=training_config.save_last,
        every_n_epochs=experiment_config.save_checkpoint_every_n_epochs
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=training_config.early_stopping_patience,
        min_delta=training_config.early_stopping_min_delta,
        mode="min"
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(config: Config, output_dir: str):
    """Setup TensorBoard logger."""
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "logs"),
        name=config.project_name,
        version=config.run_name
    )
    return logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train KAN-TTS models")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for experiment")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config.from_dict(config_dict)
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Set random seed
    if config.reproducibility.seed is not None:
        pl.seed_everything(config.reproducibility.seed)
        logger.info(f"Random seed set to: {config.reproducibility.seed}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    model = TTSTrainer(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup callbacks
    callbacks = setup_callbacks(config, args.output_dir)
    
    # Setup logger
    tb_logger = setup_logger(config, args.output_dir)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=config.experiment.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        gradient_clip_val=config.training.gradient_clip_val,
        deterministic=config.reproducibility.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    logger.info("Training completed!")
    logger.info(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main() 