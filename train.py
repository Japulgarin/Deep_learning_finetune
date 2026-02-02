"""
Main training script for Airbnb listings price prediction.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

from src.data import AirbnbDataLoader
from src.models import create_model
from src.utils import Trainer, FineTuner, evaluate_model, calculate_metrics
from src.config import Config, ModelConfig, TrainingConfig, DataConfig


def train(config: Config):
    """
    Main training function.
    
    Args:
        config: Configuration object
    """
    print("=" * 50)
    print("Airbnb Listings Fine-Tuning")
    print("=" * 50)
    
    # Set device
    device = config.training.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\n=== Loading Data ===")
    data_loader = AirbnbDataLoader(config.data.data_path)
    
    if config.data.data_path is None:
        print(f"Generating sample data with {config.data.n_samples} samples")
        df = data_loader.create_sample_data(config.data.n_samples)
    else:
        import pandas as pd
        df = pd.read_csv(config.data.data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_data = data_loader.preprocess_data(
        df,
        target_column=config.data.target_column,
        test_size=config.data.test_size,
        val_size=config.data.val_size
    )
    
    # Create dataloaders
    dataloaders = data_loader.create_dataloaders(
        preprocessed_data,
        batch_size=config.training.batch_size
    )
    
    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Test samples: {len(dataloaders['test'].dataset)}")
    
    # Create model
    print("\n=== Creating Model ===")
    input_dim = preprocessed_data['train'][0].shape[1]
    print(f"Input dimensions: {input_dim}")
    
    model = create_model(
        input_dim=input_dim,
        model_type=config.model.model_type,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        pretrained_path=config.model.pretrained_path,
        freeze_base=config.model.freeze_base
    )
    
    print(f"Model type: {config.model.model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Create trainer
    if config.model.model_type == 'pretrained' and config.training.use_progressive_unfreezing:
        trainer = FineTuner(model, criterion, optimizer, device, scheduler)
    else:
        trainer = Trainer(model, criterion, optimizer, device, scheduler)
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"{config.experiment_name}_best.pth"
    )
    
    # Train
    print("\n=== Training ===")
    if config.training.use_progressive_unfreezing and isinstance(trainer, FineTuner):
        history = trainer.progressive_unfreezing(
            dataloaders['train'],
            dataloaders['val'],
            phases=config.training.unfreezing_phases,
            save_path=checkpoint_path
        )
    else:
        history = trainer.fit(
            dataloaders['train'],
            dataloaders['val'],
            epochs=config.training.epochs,
            save_path=checkpoint_path,
            early_stopping_patience=config.training.early_stopping_patience
        )
    
    # Evaluate on test set
    print("\n=== Evaluation ===")
    trainer.load_checkpoint(checkpoint_path)
    predictions, targets = evaluate_model(model, dataloaders['test'], device)
    metrics = calculate_metrics(targets, predictions)
    
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    os.makedirs(config.results_dir, exist_ok=True)
    results_path = os.path.join(config.results_dir, f"{config.experiment_name}_results.txt")
    with open(results_path, 'w') as f:
        f.write("Test Set Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_path}")
    print(f"Model saved to {checkpoint_path}")
    print("\nTraining complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Airbnb price prediction model')
    parser.add_argument('--model-type', type=str, default='simple', 
                       choices=['simple', 'pretrained'],
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data CSV file')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples to generate (if no data path)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--experiment-name', type=str, default='airbnb_finetune',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        model=ModelConfig(model_type=args.model_type),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        ),
        data=DataConfig(
            data_path=args.data_path,
            n_samples=args.n_samples
        ),
        experiment_name=args.experiment_name
    )
    
    train(config)


if __name__ == '__main__':
    main()
