"""
Training utilities for fine-tuning Airbnb models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer class for model training and fine-tuning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[object] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on ('cpu' or 'cuda')
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_features, batch_targets in tqdm(dataloader, desc="Training"):
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_features)
            loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_features)
                loss = self.criterion(predictions, batch_targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 10
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_path: Path to save the best model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self.save_checkpoint(save_path)
                    print(f"Model saved to {save_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {path}")


class FineTuner(Trainer):
    """
    Fine-tuner class for transfer learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[object] = None,
        freeze_base: bool = True
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model: Model with pretrained base
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
            freeze_base: Whether to start with frozen base layers
        """
        super().__init__(model, criterion, optimizer, device, scheduler)
        self.freeze_base = freeze_base
        
    def progressive_unfreezing(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        phases: list = [(5, True), (5, False)],
        save_path: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Fine-tune with progressive unfreezing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            phases: List of (epochs, freeze_base) tuples
            save_path: Path to save the model
            
        Returns:
            Training history
        """
        for phase_idx, (epochs, freeze) in enumerate(phases):
            print(f"\n=== Phase {phase_idx + 1}: {'Frozen' if freeze else 'Unfrozen'} base ===")
            
            # Freeze/unfreeze base
            if hasattr(self.model, 'freeze_base_layers') and freeze:
                self.model.freeze_base_layers()
            elif hasattr(self.model, 'unfreeze_base_layers') and not freeze:
                self.model.unfreeze_base_layers()
            
            # Train for this phase
            self.fit(
                train_loader,
                val_loader,
                epochs=epochs,
                save_path=save_path,
                early_stopping_patience=epochs  # No early stopping within phases
            )
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
