"""
Configuration utilities for KAN-TTS thesis project.
"""

import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str
    train_files: str
    val_files: str
    input_features: list
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool
    val_split: float
    random_seed: int


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sampling_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    f_min: float
    f_max: float
    f0_min: float
    f0_max: float
    f0_bin_size: int
    energy_threshold: float
    duration_min: float
    duration_max: float


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    prosody_predictor: Dict[str, Any]
    phoneme_encoder: Dict[str, Any]
    spectrogram_decoder: Dict[str, Any]
    vocoder: Dict[str, Any]
    architecture_type: Optional[str] = None  # "transformer", "clean_input", etc.
    linguistic_encoder: Optional[Dict[str, Any]] = None  # For clean input architecture


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    optimizer: str
    learning_rate: float
    weight_decay: float
    betas: list
    eps: float
    scheduler: str
    max_epochs: int
    max_steps: Optional[int]
    early_stopping_patience: int
    early_stopping_min_delta: float
    gradient_clip_val: float
    gradient_clip_norm: Optional[float]
    use_amp: bool
    precision: str
    save_top_k: int
    save_last: bool
    save_every_n_epochs: int
    val_check_interval: float
    num_sanity_val_steps: int
    # Optional/with defaults
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    warmup_steps: Optional[int] = None  # Added for learning rate warm-up


@dataclass
class LossConfig:
    """Loss function configuration."""
    mel_loss_weight: float
    duration_loss_weight: float
    f0_loss_weight: float
    energy_loss_weight: float
    mel_loss: str
    duration_loss: str
    f0_loss: str
    energy_loss: str
    use_kl_loss: bool
    kl_loss_weight: float


@dataclass
class ExperimentConfig:
    """Experiment logging and checkpointing configuration."""
    log_every_n_steps: int
    log_every_n_epochs: int
    log_gradients: bool
    log_learning_rate: bool
    log_histograms: bool
    log_audio_every_n_epochs: int
    log_audio_samples: int
    save_checkpoint_every_n_epochs: int
    save_best_model: bool
    monitor: str
    mode: str
    output_dir: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list
    generate_audio_every_n_epochs: int
    num_eval_samples: int
    test_sentences: list


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    gpus: int
    accelerator: str
    devices: int
    accumulate_grad_batches: int
    sync_batchnorm: bool
    precision: str
    deterministic: bool


@dataclass
class DebugConfig:
    """Debug configuration."""
    debug_mode: bool
    fast_dev_run: bool
    enable_profiler: bool
    profiler_filename: str
    track_grad_norm: bool
    log_memory_usage: bool
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None


@dataclass
class ReproducibilityConfig:
    """Reproducibility configuration."""
    deterministic: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    seed: Optional[int] = None


@dataclass
class Config:
    """Main configuration class."""
    project_name: str
    run_name: str
    experiment_id: str
    description: str
    
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    experiment: ExperimentConfig
    evaluation: EvaluationConfig
    hardware: HardwareConfig
    debug: DebugConfig
    reproducibility: ReproducibilityConfig
    
    # Optional model-specific configurations
    mlp: Optional[Dict[str, Any]] = None
    transformer: Optional[Dict[str, Any]] = None
    kan: Optional[Dict[str, Any]] = None
    ablation: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config instance from dictionary."""
        return cls(
            project_name=config_dict.get('project_name', 'KAN-TTS-Thesis'),
            run_name=config_dict.get('run_name', 'default'),
            experiment_id=config_dict.get('experiment_id', 'default'),
            description=config_dict.get('description', ''),
            
            data=DataConfig(**config_dict.get('data', {})),
            audio=AudioConfig(**config_dict.get('audio', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            hardware=HardwareConfig(**config_dict.get('hardware', {})),
            debug=DebugConfig(**config_dict.get('debug', {})),
            reproducibility=ReproducibilityConfig(**config_dict.get('reproducibility', {})),
            
            mlp=config_dict.get('mlp'),
            transformer=config_dict.get('transformer'),
            kan=config_dict.get('kan'),
            ablation=config_dict.get('ablation')
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> bool:
        """Validate configuration."""
        # Check required files exist
        if not Path(self.data.dataset_path).exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data.dataset_path}")
        
        if not Path(self.data.train_files).exists():
            raise FileNotFoundError(f"Train files not found: {self.data.train_files}")
        
        if not Path(self.data.val_files).exists():
            raise FileNotFoundError(f"Val files not found: {self.data.val_files}")
        
        # Validate model configuration
        if self.model.prosody_predictor.name not in ['mlp', 'transformer', 'kan_fkf']:
            raise ValueError(f"Unknown prosody predictor: {self.model.prosody_predictor.name}")
        
        # Validate training configuration
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.training.max_epochs <= 0:
            raise ValueError("Max epochs must be positive")
        
        # Validate hardware configuration
        if self.hardware.devices <= 0:
            raise ValueError("Number of devices must be positive")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'project_name': self.project_name,
            'run_name': self.run_name,
            'experiment_id': self.experiment_id,
            'description': self.description,
            'data': self.data.__dict__,
            'audio': self.audio.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'loss': self.loss.__dict__,
            'experiment': self.experiment.__dict__,
            'evaluation': self.evaluation.__dict__,
            'hardware': self.hardware.__dict__,
            'debug': self.debug.__dict__,
            'reproducibility': self.reproducibility.__dict__,
            'mlp': self.mlp,
            'transformer': self.transformer,
            'kan': self.kan,
            'ablation': self.ablation
        }
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(experiment_id='{self.experiment_id}', run_name='{self.run_name}')" 