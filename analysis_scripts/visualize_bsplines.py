import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
import seaborn as sns

def load_model(checkpoint_path: str) -> Dict:
    """Load model checkpoint."""
    return torch.load(checkpoint_path, map_location='cpu')

def extract_bspline_parameters(state_dict: Dict, layer_name: str = 'kan_layer') -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract B-spline control points and knots from model state dict."""
    control_points = None
    for key, value in state_dict.items():
        if f'{layer_name}.control_points' in key:
            control_points = value
    
    if control_points is None:
        raise ValueError(f"Could not find B-spline parameters for layer {layer_name}")
    
    # Create uniform knot vector (as done in the model)
    num_basis = control_points.shape[-1]
    knots = torch.linspace(0, 1, num_basis)
    
    return control_points, knots

def evaluate_bspline(control_points: torch.Tensor, knots: torch.Tensor, x: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """Evaluate B-spline at given points."""
    # Implementation of Cox-de Boor recursion formula
    n = len(knots) - 1
    d = degree
    
    # Initialize basis functions array
    N = torch.zeros((len(x), n+1, d+1))
    
    # Initialize degree 0
    for j in range(n):
        N[:, j, 0] = ((x >= knots[j]) & (x < knots[j+1])).float()
    N[:, -1, 0] = (x == knots[-1]).float()
    
    # Build up degrees
    for k in range(1, d+1):
        for j in range(n-k+1):
            # First term
            denom1 = knots[j+k] - knots[j]
            numer1 = (x - knots[j])
            term1 = torch.where(denom1 != 0,
                              (numer1/denom1) * N[:, j, k-1],
                              torch.zeros_like(x))
            
            # Second term
            denom2 = knots[j+k+1] - knots[j+1]
            numer2 = (knots[j+k+1] - x)
            term2 = torch.where(denom2 != 0,
                              (numer2/denom2) * N[:, j+1, k-1],
                              torch.zeros_like(x))
            
            N[:, j, k] = term1 + term2
    
    # Evaluate spline
    y = torch.zeros((len(x), control_points.shape[0]))
    for i in range(control_points.shape[0]):
        y[:, i] = torch.sum(N[:, :, -1] * control_points[i], dim=1)
    
    return y

def plot_feature_transformations(control_points: torch.Tensor,
                               knots: torch.Tensor,
                               feature_names: List[str],
                               output_dir: str,
                               degree: int = 3):
    """Plot learned feature transformations."""
    x = torch.linspace(0, 1, 1000)
    y = evaluate_bspline(control_points, knots, x, degree)
    
    n_features = min(len(feature_names), control_points.shape[0])
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5*n_rows))
    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i+1)
        plt.plot(x.numpy(), y[:, i].numpy(), 'b-', label='Learned Transform')
        plt.plot(knots.numpy(), control_points[i].numpy(), 'r.', label='Control Points')
        plt.title(feature_names[i])
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_transformations.png", dpi=300, bbox_inches='tight')
    plt.close()

def calculate_feature_importance(control_points: torch.Tensor) -> np.ndarray:
    """Calculate feature importance based on control point variation."""
    # Calculate variance of control points for each feature
    importance = torch.var(control_points, dim=1)
    # Normalize to sum to 1
    importance = importance / importance.sum()
    return importance.numpy()

def plot_feature_importance(importance: np.ndarray,
                          feature_names: List[str],
                          output_dir: str):
    """Plot feature importance as a bar chart."""
    plt.figure(figsize=(12, 6))
    
    # Sort by importance
    idx = np.argsort(importance)[::-1]
    importance = importance[idx]
    feature_names = [feature_names[i] for i in idx]
    
    # Plot
    sns.barplot(x=importance, y=feature_names)
    plt.title('Feature Importance Based on B-spline Transformations')
    plt.xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize B-spline transformations')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for plots')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to model config file')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and config
    checkpoint = load_model(args.checkpoint)
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Get feature names from config
    feature_names = config['data']['input_features']
    
    # Extract B-spline parameters
    control_points, knots = extract_bspline_parameters(checkpoint['state_dict'])
    
    # Create visualizations
    plot_feature_transformations(control_points, knots, feature_names, str(output_dir))
    
    importance = calculate_feature_importance(control_points)
    plot_feature_importance(importance, feature_names, str(output_dir))
    
    # Save feature importance scores
    importance_dict = {
        name: float(score)
        for name, score in zip(feature_names, importance)
    }
    with open(output_dir / 'feature_importance.json', 'w') as f:
        json.dump(importance_dict, f, indent=2)
    
    print(f"✅ Visualizations saved to: {output_dir}")
    print("\nTop 5 most important features:")
    for name, score in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {score:.4f}")

if __name__ == '__main__':
    main() 