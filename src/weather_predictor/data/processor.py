from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    """PyTorch Dataset for weather data."""
    
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class WeatherDataProcessor:
    """Process weather data for model training and prediction."""
    
    def __init__(self, sequence_length: int = 24):
        """Initialize the data processor.
        
        Args:
            sequence_length (int): Number of time steps to use for prediction
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.feature_columns = [
            'temperature', 'dewpoint', 'windSpeed', 'windDirection',
            'barometricPressure', 'relativeHumidity', 'precipitation'
        ]
        
    def process_observations(self, observations: List[Dict]) -> pd.DataFrame:
        """Process raw observation data into a pandas DataFrame.
        
        Args:
            observations (List[Dict]): Raw observation data from the API
            
        Returns:
            pd.DataFrame: Processed observations
        """
        processed_data = []
        
        for obs in observations:
            properties = obs['properties']
            row = {
                'timestamp': datetime.fromisoformat(properties['timestamp']),
                'temperature': float(properties['temperature']['value']),
                'dewpoint': float(properties['dewpoint']['value']),
                'windSpeed': float(properties['windSpeed']['value']),
                'windDirection': float(properties['windDirection']['value']),
                'barometricPressure': float(properties['barometricPressure']['value']),
                'relativeHumidity': float(properties['relativeHumidity']['value']),
                'precipitation': float(properties.get('precipitation', {'value': 0})['value'] or 0)
            }
            processed_data.append(row)
            
        df = pd.DataFrame(processed_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
        
    def create_sequences(
        self,
        data: pd.DataFrame,
        target_hours: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training/prediction.
        
        Args:
            data (pd.DataFrame): Processed weather data
            target_hours (int): Number of hours ahead to predict
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets arrays
        """
        features = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - target_hours + 1):
            seq = data[i:(i + self.sequence_length)]
            target = data.iloc[i + self.sequence_length + target_hours - 1]['temperature']
            
            features.append(seq[self.feature_columns].values)
            targets.append(target)
            
        return np.array(features), np.array(targets)
        
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_hours: int = 24,
        train_split: float = 0.8,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for model training.
        
        Args:
            data (pd.DataFrame): Processed weather data
            target_hours (int): Number of hours ahead to predict
            train_split (float): Fraction of data to use for training
            batch_size (int): Batch size for DataLoader
            
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation DataLoaders
        """
        # Scale features
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(data[self.feature_columns]),
            columns=self.feature_columns,
            index=data.index
        )
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, target_hours)
        
        # Split into train/val
        train_size = int(len(X) * train_split)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Create datasets and dataloaders
        train_dataset = WeatherDataset(X_train, y_train)
        val_dataset = WeatherDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
        
    def prepare_prediction_data(
        self,
        data: pd.DataFrame
    ) -> torch.Tensor:
        """Prepare data for making predictions.
        
        Args:
            data (pd.DataFrame): Recent weather data
            
        Returns:
            torch.Tensor: Processed input for the model
        """
        # Scale features
        scaled_data = pd.DataFrame(
            self.scaler.transform(data[self.feature_columns]),
            columns=self.feature_columns,
            index=data.index
        )
        
        # Take the last sequence
        sequence = scaled_data.iloc[-self.sequence_length:][self.feature_columns].values
        
        # Convert to tensor
        return torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension 