import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import scipy.stats as stats

def load_metrics(metrics_file: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)

def calculate_statistics(values: List[float]) -> Dict:
    """Calculate mean, std, and confidence intervals."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    
    # 95% confidence interval
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
    
    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'n_samples': n
    }

def aggregate_metrics(base_dir: str) -> Dict:
    """Aggregate metrics across multiple runs."""
    base_dir = Path(base_dir)
    metrics_files = list(base_dir.glob('seed_*/metrics.json'))
    
    if not metrics_files:
        raise ValueError(f"No metrics files found in {base_dir}")
    
    # Collect metrics from all runs
    all_metrics = {
        'duration_rmse': [],
        'f0_rmse': [],
        'energy_rmse': [],
        'best_val_loss': [],
        'convergence_epoch': [],
        'training_time_hours': [],
        'total_parameters': [],
        'model_size_mb': []
    }
    
    learning_curves = {
        'train_loss': [],
        'val_loss': [],
        'duration_rmse': [],
        'f0_rmse': [],
        'energy_rmse': []
    }
    
    for metrics_file in metrics_files:
        metrics = load_metrics(str(metrics_file))
        
        # Collect final metrics
        all_metrics['duration_rmse'].append(metrics['training_metrics']['duration_rmse'])
        all_metrics['f0_rmse'].append(metrics['training_metrics']['f0_rmse'])
        all_metrics['energy_rmse'].append(metrics['training_metrics']['energy_rmse'])
        all_metrics['best_val_loss'].append(metrics['training_metrics']['best_val_loss'])
        all_metrics['convergence_epoch'].append(metrics['training_metrics']['convergence_epoch'])
        all_metrics['training_time_hours'].append(metrics['training_metrics']['training_time_hours'])
        all_metrics['total_parameters'].append(metrics['model_statistics']['total_parameters'])
        all_metrics['model_size_mb'].append(metrics['model_statistics']['model_size_mb'])
        
        # Collect learning curves
        for key in learning_curves:
            learning_curves[key].append(metrics['learning_curves'][key])
    
    # Calculate statistics for each metric
    aggregated = {
        'metrics_statistics': {
            metric: calculate_statistics(values)
            for metric, values in all_metrics.items()
        },
        'learning_curves': {
            key: {
                'mean': np.mean(curves, axis=0).tolist(),
                'std': np.std(curves, axis=0).tolist()
            }
            for key, curves in learning_curves.items()
        }
    }
    
    # Add summary statistics
    aggregated['summary'] = {
        'n_runs': len(metrics_files),
        'mean_convergence_epoch': float(np.mean(all_metrics['convergence_epoch'])),
        'mean_training_time': float(np.mean(all_metrics['training_time_hours'])),
        'total_training_time': float(np.sum(all_metrics['training_time_hours'])),
        'parameter_count': int(all_metrics['total_parameters'][0]),  # Should be same for all runs
        'model_size_mb': float(all_metrics['model_size_mb'][0])
    }
    
    return aggregated

def main():
    parser = argparse.ArgumentParser(description='Aggregate metrics across multiple runs')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing seed_* subdirectories')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file')
    args = parser.parse_args()
    
    # Aggregate metrics
    aggregated_metrics = aggregate_metrics(args.base_dir)
    
    # Save aggregated metrics
    with open(args.output_file, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    print(f"✅ Aggregated metrics saved to: {args.output_file}")
    
    # Print summary
    print("\n📊 Summary Statistics:")
    print(f"Number of runs: {aggregated_metrics['summary']['n_runs']}")
    print(f"Mean convergence epoch: {aggregated_metrics['summary']['mean_convergence_epoch']:.1f}")
    print(f"Mean training time: {aggregated_metrics['summary']['mean_training_time']:.1f} hours")
    print(f"Total training time: {aggregated_metrics['summary']['total_training_time']:.1f} hours")
    print(f"Model size: {aggregated_metrics['summary']['model_size_mb']:.1f} MB")
    print(f"Parameter count: {aggregated_metrics['summary']['parameter_count']:,}")

if __name__ == '__main__':
    main() 