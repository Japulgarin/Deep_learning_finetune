"""
Data loading and preprocessing module for Airbnb listings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader


class AirbnbListingsDataset(Dataset):
    """PyTorch Dataset for Airbnb listings data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features: Numpy array of features
            targets: Numpy array of target values
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class AirbnbDataLoader:
    """Data loader for Airbnb listings."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing Airbnb listings
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create sample Airbnb listings data for demonstration.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with sample listings
        """
        np.random.seed(42)
        
        data = {
            'property_type': np.random.choice(['Apartment', 'House', 'Condo', 'Loft'], n_samples),
            'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_samples),
            'accommodates': np.random.randint(1, 10, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'beds': np.random.randint(1, 8, n_samples),
            'minimum_nights': np.random.randint(1, 30, n_samples),
            'number_of_reviews': np.random.randint(0, 200, n_samples),
            'review_scores_rating': np.random.uniform(3.5, 5.0, n_samples),
            'latitude': np.random.uniform(40.5, 40.9, n_samples),
            'longitude': np.random.uniform(-74.1, -73.7, n_samples),
            'availability_365': np.random.randint(0, 365, n_samples),
        }
        
        # Generate price based on features (with some noise)
        base_price = (
            data['accommodates'] * 20 +
            data['bedrooms'] * 30 +
            data['bathrooms'] * 25 +
            data['review_scores_rating'] * 10 +
            np.random.normal(0, 20, n_samples)
        )
        data['price'] = np.maximum(base_price, 30)  # Minimum price of $30
        
        return pd.DataFrame(data)
    
    def preprocess_data(
        self, 
        df: pd.DataFrame,
        target_column: str = 'price',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict:
        """
        Preprocess the data for model training.
        
        Args:
            df: DataFrame with Airbnb listings
            target_column: Name of the target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Dictionary with preprocessed train, validation, and test datasets
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column].values
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy
        X = X.values
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Reshape targets
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def create_dataloaders(
        self,
        preprocessed_data: Dict,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders from preprocessed data.
        
        Args:
            preprocessed_data: Dictionary with train, val, test data
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the training data
            
        Returns:
            Dictionary of DataLoaders for train, val, and test
        """
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            X, y = preprocessed_data[split]
            dataset = AirbnbListingsDataset(X, y)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(shuffle and split == 'train')
            )
        
        return dataloaders
