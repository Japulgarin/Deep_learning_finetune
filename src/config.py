"""
Configuration management for training.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = 'simple'  # 'simple' or 'pretrained'
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    pretrained_path: Optional[str] = None
    freeze_base: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    device: str = 'cpu'  # 'cpu' or 'cuda'
    
    # Fine-tuning specific
    use_progressive_unfreezing: bool = False
    unfreezing_phases: List[tuple] = field(default_factory=lambda: [(5, True), (10, False)])


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: Optional[str] = None
    n_samples: int = 1000  # For generated data
    test_size: float = 0.2
    val_size: float = 0.1
    target_column: str = 'price'


@dataclass
class Config:
    """Main configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    
    # Experiment name
    experiment_name: str = 'airbnb_finetune'
