import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import numpy as np
from datetime import datetime

def load_aggregated_metrics(model_dir: str) -> dict:
    """Load aggregated metrics for a model."""
    metrics_file = Path(model_dir) / 'aggregated_metrics.json'
    with open(metrics_file, 'r') as f:
        return json.load(f)

def create_learning_curves(metrics: dict, output_dir: str) -> str:
    """Create learning curves plot and return the file path."""
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    epochs = range(1, len(metrics['learning_curves']['train_loss']['mean']) + 1)
    plt.plot(epochs, metrics['learning_curves']['train_loss']['mean'], label='Train Loss')
    plt.plot(epochs, metrics['learning_curves']['val_loss']['mean'], label='Val Loss')
    plt.fill_between(epochs,
                    np.array(metrics['learning_curves']['train_loss']['mean']) - np.array(metrics['learning_curves']['train_loss']['std']),
                    np.array(metrics['learning_curves']['train_loss']['mean']) + np.array(metrics['learning_curves']['train_loss']['std']),
                    alpha=0.2)
    plt.fill_between(epochs,
                    np.array(metrics['learning_curves']['val_loss']['mean']) - np.array(metrics['learning_curves']['val_loss']['std']),
                    np.array(metrics['learning_curves']['val_loss']['mean']) + np.array(metrics['learning_curves']['val_loss']['std']),
                    alpha=0.2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot RMSE metrics
    plt.subplot(2, 1, 2)
    for metric in ['duration_rmse', 'f0_rmse', 'energy_rmse']:
        plt.plot(epochs, metrics['learning_curves'][metric]['mean'], label=metric.replace('_', ' ').title())
        plt.fill_between(epochs,
                        np.array(metrics['learning_curves'][metric]['mean']) - np.array(metrics['learning_curves'][metric]['std']),
                        np.array(metrics['learning_curves'][metric]['mean']) + np.array(metrics['learning_curves'][metric]['std']),
                        alpha=0.2)
    plt.title('RMSE Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'learning_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(plot_path)

def create_comparison_plots(all_metrics: dict, output_dir: str) -> str:
    """Create comparison plots across models and return the file path."""
    plt.figure(figsize=(15, 10))
    
    # Prepare data
    models = list(all_metrics.keys())
    metrics = ['duration_rmse', 'f0_rmse', 'energy_rmse']
    
    # Plot mean metrics with error bars
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        means = [all_metrics[model]['metrics_statistics'][m]['mean'] for m in metrics]
        stds = [all_metrics[model]['metrics_statistics'][m]['std'] for m in metrics]
        plt.bar(x + i*width, means, width, label=model,
               yerr=stds, capsize=5)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Comparison Across Models')
    plt.xticks(x + width*(len(models)-1)/2, [m.replace('_', ' ').title() for m in metrics])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = Path(output_dir) / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(plot_path)

def create_pdf_report(base_dir: str, output_file: str):
    """Generate comprehensive PDF report."""
    # Load metrics for all models
    base_dir = Path(base_dir)
    all_metrics = {}
    for model_dir in base_dir.glob('*'):
        if model_dir.is_dir() and (model_dir / 'aggregated_metrics.json').exists():
            all_metrics[model_dir.name] = load_aggregated_metrics(str(model_dir))
    
    # Create plots
    plots_dir = base_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    comparison_plot = create_comparison_plots(all_metrics, str(plots_dir))
    
    # Create PDF
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph("KAN-TTS Experiment Results", title_style))
    elements.append(Spacer(1, 12))
    
    # Date and summary
    elements.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Model Comparison Table
    table_data = [['Model', 'Parameters', 'Size (MB)', 'Conv. Epoch', 'Train Time (h)']]
    for model, metrics in all_metrics.items():
        table_data.append([
            model,
            f"{metrics['summary']['parameter_count']:,}",
            f"{metrics['summary']['model_size_mb']:.1f}",
            f"{metrics['summary']['mean_convergence_epoch']:.1f}",
            f"{metrics['summary']['mean_training_time']:.1f}"
        ])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Add comparison plot
    elements.append(Paragraph("Model Performance Comparison", styles['Heading2']))
    elements.append(Image(comparison_plot, width=7*inch, height=5*inch))
    elements.append(Spacer(1, 20))
    
    # Detailed Results per Model
    elements.append(Paragraph("Detailed Results per Model", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    for model, metrics in all_metrics.items():
        elements.append(Paragraph(f"Model: {model}", styles['Heading3']))
        elements.append(Spacer(1, 12))
        
        # Create learning curves for this model
        learning_curves_plot = create_learning_curves(metrics, str(plots_dir))
        elements.append(Image(learning_curves_plot, width=7*inch, height=5*inch))
        elements.append(Spacer(1, 12))
        
        # Metrics table
        metrics_data = [['Metric', 'Mean', 'Std Dev', '95% CI']]
        for metric, stats in metrics['metrics_statistics'].items():
            if metric not in ['total_parameters', 'model_size_mb']:
                metrics_data.append([
                    metric.replace('_', ' ').title(),
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"({stats['ci_lower']:.4f}, {stats['ci_upper']:.4f})"
                ])
        
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(elements)

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive PDF report')
    parser.add_argument('--base_dir', type=str, required=True,
                      help='Base directory containing model subdirectories')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Output PDF file path')
    args = parser.parse_args()
    
    create_pdf_report(args.base_dir, args.output_file)
    print(f"✅ Report generated: {args.output_file}")

if __name__ == '__main__':
    main() 