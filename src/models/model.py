"""
Neural network models for Airbnb price prediction.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class AirbnbPricePredictor(nn.Module):
    """
    Neural network for predicting Airbnb listing prices.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super(AirbnbPricePredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.model(x)


class PretrainedAirbnbModel(nn.Module):
    """
    Model with pretrained base that can be fine-tuned.
    """
    
    def __init__(
        self,
        input_dim: int,
        pretrained_path: Optional[str] = None,
        freeze_base: bool = False
    ):
        """
        Initialize the model with optional pretrained weights.
        
        Args:
            input_dim: Number of input features
            pretrained_path: Path to pretrained model weights
            freeze_base: Whether to freeze the base layers during fine-tuning
        """
        super(PretrainedAirbnbModel, self).__init__()
        
        # Base network (can be pretrained)
        self.base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fine-tuning head
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)
        
        # Freeze base if requested
        if freeze_base:
            self.freeze_base_layers()
    
    def load_pretrained(self, path: str):
        """Load pretrained weights."""
        state_dict = torch.load(path, map_location='cpu')
        self.base.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {path}")
    
    def freeze_base_layers(self):
        """Freeze the base layers for fine-tuning."""
        for param in self.base.parameters():
            param.requires_grad = False
        print("Base layers frozen for fine-tuning")
    
    def unfreeze_base_layers(self):
        """Unfreeze the base layers."""
        for param in self.base.parameters():
            param.requires_grad = True
        print("Base layers unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output predictions
        """
        features = self.base(x)
        output = self.head(features)
        return output


def create_model(
    input_dim: int,
    model_type: str = 'simple',
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        input_dim: Number of input features
        model_type: Type of model ('simple' or 'pretrained')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type == 'simple':
        # Filter kwargs for simple model
        simple_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['hidden_dims', 'dropout']}
        return AirbnbPricePredictor(input_dim, **simple_kwargs)
    elif model_type == 'pretrained':
        # Filter kwargs for pretrained model
        pretrained_kwargs = {k: v for k, v in kwargs.items() 
                            if k in ['pretrained_path', 'freeze_base']}
        return PretrainedAirbnbModel(input_dim, **pretrained_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
