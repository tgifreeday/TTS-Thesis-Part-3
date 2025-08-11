"""
Logging utilities for KAN-TTS thesis project.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    """
    Setup logging configuration for the experiment.
    
    Args:
        output_dir: Directory to save log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(logs_dir / "training.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log files will be saved to: {logs_dir}")
    logger.info(f"Log level set to: {level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Logger specifically for experiment tracking.
    """
    
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.logger = get_logger(f"experiment.{experiment_name}")
        
        # Create experiment-specific log file
        self.log_file = self.output_dir / "logs" / f"{experiment_name}.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler for experiment-specific logging
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_config(self, config: dict):
        """Log configuration parameters."""
        self.logger.info("=== EXPERIMENT CONFIGURATION ===")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=== END CONFIGURATION ===")
    
    def log_training_start(self, model_params: int, num_train_samples: int, num_val_samples: int):
        """Log training start information."""
        self.logger.info("=== TRAINING START ===")
        self.logger.info(f"Model parameters: {model_params:,}")
        self.logger.info(f"Training samples: {num_train_samples:,}")
        self.logger.info(f"Validation samples: {num_val_samples:,}")
        self.logger.info("=== END TRAINING START ===")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Log epoch information."""
        self.logger.info(
            f"Epoch {epoch:03d} - Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - LR: {lr:.2e}"
        )
    
    def log_training_end(self, best_val_loss: float, total_epochs: int):
        """Log training end information."""
        self.logger.info("=== TRAINING END ===")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info("=== END TRAINING END ===")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error information."""
        self.logger.error(f"ERROR in {context}: {str(error)}")
        self.logger.error(f"Error type: {type(error).__name__}")
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if step is not None:
            self.logger.info(f"Metric {metric_name} at step {step}: {value:.6f}")
        else:
            self.logger.info(f"Metric {metric_name}: {value:.6f}")


def log_experiment_summary(output_dir: str, config: dict, results: dict):
    """
    Log a summary of the experiment.
    
    Args:
        output_dir: Experiment output directory
        config: Configuration used
        results: Results dictionary
    """
    summary_file = Path(output_dir) / "experiment_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=== KAN-TTS EXPERIMENT SUMMARY ===\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nRESULTS:\n")
        f.write("-" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n=== END SUMMARY ===\n")
    
    print(f"Experiment summary saved to: {summary_file}") 