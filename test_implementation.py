"""
Test script to verify the implementation works correctly.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '/home/runner/work/Deep_learning_finetune/Deep_learning_finetune')

from src.data import AirbnbDataLoader
from src.models import create_model, AirbnbPricePredictor, PretrainedAirbnbModel
from src.utils import Trainer, evaluate_model, calculate_metrics


def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading...")
    data_loader = AirbnbDataLoader()
    df = data_loader.create_sample_data(n_samples=100)
    assert len(df) == 100, "Data generation failed"
    assert 'price' in df.columns, "Price column missing"
    print("✓ Data loading works")
    
    preprocessed_data = data_loader.preprocess_data(df)
    assert 'train' in preprocessed_data, "Training data missing"
    assert 'val' in preprocessed_data, "Validation data missing"
    assert 'test' in preprocessed_data, "Test data missing"
    print("✓ Data preprocessing works")
    
    dataloaders = data_loader.create_dataloaders(preprocessed_data, batch_size=16)
    batch = next(iter(dataloaders['train']))
    assert len(batch) == 2, "Batch format incorrect"
    print("✓ DataLoader creation works")
    
    return preprocessed_data, dataloaders


def test_models():
    """Test model creation."""
    print("\nTesting models...")
    
    # Test simple model
    model_simple = create_model(input_dim=12, model_type='simple')
    assert isinstance(model_simple, AirbnbPricePredictor), "Simple model creation failed"
    print("✓ Simple model creation works")
    
    # Test pretrained model
    model_pretrained = create_model(input_dim=12, model_type='pretrained', freeze_base=True)
    assert isinstance(model_pretrained, PretrainedAirbnbModel), "Pretrained model creation failed"
    print("✓ Pretrained model creation works")
    
    # Test forward pass
    x = torch.randn(4, 12)
    output = model_simple(x)
    assert output.shape == (4, 1), "Forward pass failed"
    print("✓ Model forward pass works")
    
    return model_simple


def test_training(model, dataloaders):
    """Test training loop."""
    print("\nTesting training...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, criterion, optimizer, device='cpu')
    
    # Train for 2 epochs
    history = trainer.fit(
        dataloaders['train'],
        dataloaders['val'],
        epochs=2,
        early_stopping_patience=10
    )
    
    assert len(history['train_losses']) == 2, "Training history incorrect"
    assert len(history['val_losses']) == 2, "Validation history incorrect"
    print("✓ Training loop works")
    
    return trainer


def test_evaluation(trainer, dataloaders):
    """Test evaluation."""
    print("\nTesting evaluation...")
    
    predictions, targets = evaluate_model(trainer.model, dataloaders['test'], device='cpu')
    assert len(predictions) == len(targets), "Predictions shape mismatch"
    print("✓ Model evaluation works")
    
    metrics = calculate_metrics(targets, predictions)
    assert 'MAE' in metrics, "MAE metric missing"
    assert 'RMSE' in metrics, "RMSE metric missing"
    assert 'R2' in metrics, "R2 metric missing"
    print("✓ Metrics calculation works")
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Implementation Tests")
    print("=" * 60)
    
    try:
        # Test data pipeline
        preprocessed_data, dataloaders = test_data_loading()
        
        # Test models
        model = test_models()
        
        # Test training
        trainer = test_training(model, dataloaders)
        
        # Test evaluation
        test_evaluation(trainer, dataloaders)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
