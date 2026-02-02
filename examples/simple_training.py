"""
Simple example script for training an Airbnb price prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import AirbnbDataLoader
from src.models import create_model
from src.utils import Trainer, evaluate_model, calculate_metrics


def main():
    """Simple training example."""
    print("=" * 50)
    print("Airbnb Price Prediction - Quick Start Example")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # 1. Load data
    print("\n1. Loading data...")
    data_loader = AirbnbDataLoader()
    df = data_loader.create_sample_data(n_samples=1000)
    print(f"   Generated {len(df)} samples")
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessed_data = data_loader.preprocess_data(df)
    dataloaders = data_loader.create_dataloaders(preprocessed_data, batch_size=32)
    print(f"   Train: {len(dataloaders['train'].dataset)} samples")
    print(f"   Val: {len(dataloaders['val'].dataset)} samples")
    print(f"   Test: {len(dataloaders['test'].dataset)} samples")
    
    # 3. Create model
    print("\n3. Creating model...")
    input_dim = preprocessed_data['train'][0].shape[1]
    model = create_model(input_dim=input_dim, model_type='simple')
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Setup training
    print("\n4. Setting up training...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, criterion, optimizer, device)
    
    # 5. Train
    print("\n5. Training model...")
    history = trainer.fit(
        dataloaders['train'],
        dataloaders['val'],
        epochs=20,
        save_path='checkpoints/simple_model.pth',
        early_stopping_patience=5
    )
    
    # 6. Evaluate
    print("\n6. Evaluating on test set...")
    predictions, targets = evaluate_model(model, dataloaders['test'], device)
    metrics = calculate_metrics(targets, predictions)
    
    print("\n" + "=" * 50)
    print("Test Set Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:10s}: ${value:.2f}" if metric in ['MAE', 'RMSE'] else f"{metric:10s}: {value:.4f}")
    
    print("\n✓ Training complete!")
    print("  Model saved to: checkpoints/simple_model.pth")


if __name__ == '__main__':
    main()
