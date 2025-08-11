import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load PyTorch checkpoint."""
    return torch.load(checkpoint_path, map_location='cpu')

def calculate_model_size(checkpoint: Dict) -> Dict:
    """Calculate model size statistics."""
    total_params = 0
    trainable_params = 0
    
    for name, param in checkpoint['state_dict'].items():
        param_count = np.prod(param.shape)
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def parse_training_log(log_path: str) -> Dict:
    """Parse training log for metrics."""
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'duration_rmse': [],
        'f0_rmse': [],
        'energy_rmse': [],
        'convergence_epoch': None,
        'best_val_loss': float('inf'),
        'training_time_hours': 0
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'val_loss=' in line:
                val_loss = float(line.split('val_loss=')[1].split()[0])
                metrics['val_loss'].append(val_loss)
                if val_loss < metrics['best_val_loss']:
                    metrics['best_val_loss'] = val_loss
            
            if 'train_loss=' in line:
                train_loss = float(line.split('train_loss=')[1].split()[0])
                metrics['train_loss'].append(train_loss)
            
            if 'duration_rmse=' in line:
                rmse = float(line.split('duration_rmse=')[1].split()[0])
                metrics['duration_rmse'].append(rmse)
            
            if 'f0_rmse=' in line:
                rmse = float(line.split('f0_rmse=')[1].split()[0])
                metrics['f0_rmse'].append(rmse)
            
            if 'energy_rmse=' in line:
                rmse = float(line.split('energy_rmse=')[1].split()[0])
                metrics['energy_rmse'].append(rmse)
            
            if 'Training completed in' in line:
                time_str = line.split('Training completed in')[1].strip()
                hours = float(time_str.split('hours')[0].strip())
                metrics['training_time_hours'] = hours
    
    # Calculate convergence epoch (when validation loss stabilizes)
    if len(metrics['val_loss']) > 10:
        window_size = 5
        val_losses = np.array(metrics['val_loss'])
        rolling_std = pd.Series(val_losses).rolling(window=window_size).std()
        convergence_idx = np.where(rolling_std < 0.01)[0]
        if len(convergence_idx) > 0:
            metrics['convergence_epoch'] = convergence_idx[0]
    
    # Calculate final metrics
    metrics['final_metrics'] = {
        'duration_rmse': np.mean(metrics['duration_rmse'][-5:]),
        'f0_rmse': np.mean(metrics['f0_rmse'][-5:]),
        'energy_rmse': np.mean(metrics['energy_rmse'][-5:]),
        'best_val_loss': metrics['best_val_loss'],
        'convergence_epoch': metrics['convergence_epoch'],
        'training_time_hours': metrics['training_time_hours']
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for experiment run')
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Find best checkpoint
    checkpoint_path = list(model_dir.glob('checkpoints/best_*.ckpt'))[0]
    checkpoint = load_checkpoint(str(checkpoint_path))
    
    # Calculate metrics
    model_size = calculate_model_size(checkpoint)
    training_metrics = parse_training_log(str(model_dir / 'training.log'))
    
    # Combine all metrics
    metrics = {
        'model_statistics': model_size,
        'training_metrics': training_metrics['final_metrics'],
        'learning_curves': {
            'train_loss': training_metrics['train_loss'],
            'val_loss': training_metrics['val_loss'],
            'duration_rmse': training_metrics['duration_rmse'],
            'f0_rmse': training_metrics['f0_rmse'],
            'energy_rmse': training_metrics['energy_rmse']
        }
    }
    
    # Save metrics
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved to: {args.output_file}")

if __name__ == '__main__':
    main() 