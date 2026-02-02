"""
Fine-tuning example with transfer learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import AirbnbDataLoader
from src.models import create_model
from src.utils import FineTuner, evaluate_model, calculate_metrics


def main():
    """Fine-tuning example with progressive unfreezing."""
    print("=" * 50)
    print("Airbnb Fine-Tuning - Transfer Learning Example")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # 1. Load data
    print("\n1. Loading data...")
    data_loader = AirbnbDataLoader()
    df = data_loader.create_sample_data(n_samples=1500)
    print(f"   Generated {len(df)} samples")
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessed_data = data_loader.preprocess_data(df)
    dataloaders = data_loader.create_dataloaders(preprocessed_data, batch_size=32)
    
    # 3. Create model with pretrained base
    print("\n3. Creating model with pretrained base...")
    input_dim = preprocessed_data['train'][0].shape[1]
    
    # For this example, we'll first train a base model
    print("   Training base model first...")
    base_model = create_model(input_dim=input_dim, model_type='simple')
    base_criterion = nn.MSELoss()
    base_optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    base_trainer = FineTuner(base_model, base_criterion, base_optimizer, device)
    
    # Quick pre-training
    base_trainer.fit(
        dataloaders['train'],
        dataloaders['val'],
        epochs=10,
        save_path='checkpoints/base_model.pth',
        early_stopping_patience=5
    )
    print("   Base model trained!")
    
    # 4. Create fine-tuning model
    print("\n4. Creating fine-tuning model...")
    ft_model = create_model(
        input_dim=input_dim,
        model_type='pretrained',
        freeze_base=True
    )
    
    print(f"   Total parameters: {sum(p.numel() for p in ft_model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in ft_model.parameters() if p.requires_grad):,}")
    
    # 5. Setup fine-tuning
    print("\n5. Setting up fine-tuner...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ft_model.parameters(), lr=0.001)
    finetuner = FineTuner(ft_model, criterion, optimizer, device)
    
    # 6. Fine-tune with progressive unfreezing
    print("\n6. Fine-tuning with progressive unfreezing...")
    print("   Phase 1: Training head only (base frozen)")
    print("   Phase 2: Training full model (base unfrozen)")
    
    history = finetuner.progressive_unfreezing(
        dataloaders['train'],
        dataloaders['val'],
        phases=[
            (5, True),   # 5 epochs with frozen base
            (10, False)  # 10 epochs with unfrozen base
        ],
        save_path='checkpoints/finetuned_model.pth'
    )
    
    # 7. Evaluate
    print("\n7. Evaluating fine-tuned model...")
    predictions, targets = evaluate_model(ft_model, dataloaders['test'], device)
    metrics = calculate_metrics(targets, predictions)
    
    print("\n" + "=" * 50)
    print("Fine-Tuned Model Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:10s}: ${value:.2f}" if metric in ['MAE', 'RMSE'] else f"{metric:10s}: {value:.4f}")
    
    print("\n✓ Fine-tuning complete!")
    print("  Base model saved to: checkpoints/base_model.pth")
    print("  Fine-tuned model saved to: checkpoints/finetuned_model.pth")


if __name__ == '__main__':
    main()
