import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime

def test_process_observations(data_processor, sample_weather_data):
    """Test processing of raw weather observations."""
    df = data_processor.process_observations(sample_weather_data["features"])
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert all(col in df.columns for col in data_processor.feature_columns)
    assert df.index.name == "timestamp"
    assert df.index[0] == datetime.fromisoformat("2024-02-10T00:00:00+00:00")

def test_create_sequences(data_processor):
    """Test creation of sequences for training."""
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
    data = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 100),
        'dewpoint': np.random.normal(15, 3, 100),
        'windSpeed': np.random.normal(5, 2, 100),
        'windDirection': np.random.uniform(0, 360, 100),
        'barometricPressure': np.random.normal(1013, 5, 100),
        'relativeHumidity': np.random.uniform(60, 90, 100),
        'precipitation': np.random.exponential(1, 100)
    }, index=dates)
    
    X, y = data_processor.create_sequences(data, target_hours=24)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[1] == data_processor.sequence_length
    assert X.shape[2] == len(data_processor.feature_columns)
    assert len(y.shape) == 1

def test_prepare_data(data_processor):
    """Test data preparation for model training."""
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="H")
    data = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 1000),
        'dewpoint': np.random.normal(15, 3, 1000),
        'windSpeed': np.random.normal(5, 2, 1000),
        'windDirection': np.random.uniform(0, 360, 1000),
        'barometricPressure': np.random.normal(1013, 5, 1000),
        'relativeHumidity': np.random.uniform(60, 90, 1000),
        'precipitation': np.random.exponential(1, 1000)
    }, index=dates)
    
    train_loader, val_loader = data_processor.prepare_data(
        data,
        target_hours=24,
        train_split=0.8,
        batch_size=32
    )
    
    # Check DataLoader properties
    assert isinstance(train_loader.dataset[0][0], torch.Tensor)
    assert isinstance(train_loader.dataset[0][1], torch.Tensor)
    assert train_loader.batch_size == 32
    assert not val_loader.shuffle

def test_prepare_prediction_data(data_processor):
    """Test data preparation for making predictions."""
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=24, freq="H")
    data = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 24),
        'dewpoint': np.random.normal(15, 3, 24),
        'windSpeed': np.random.normal(5, 2, 24),
        'windDirection': np.random.uniform(0, 360, 24),
        'barometricPressure': np.random.normal(1013, 5, 24),
        'relativeHumidity': np.random.uniform(60, 90, 24),
        'precipitation': np.random.exponential(1, 24)
    }, index=dates)
    
    prediction_input = data_processor.prepare_prediction_data(data)
    
    assert isinstance(prediction_input, torch.Tensor)
    assert prediction_input.shape == (1, 24, len(data_processor.feature_columns))

def test_weather_dataset():
    """Test WeatherDataset class."""
    features = torch.randn(100, 24, 7)
    targets = torch.randn(100)
    dataset = data_processor.WeatherDataset(features, targets)
    
    assert len(dataset) == 100
    sample_features, sample_target = dataset[0]
    assert sample_features.shape == (24, 7)
    assert isinstance(sample_target, torch.Tensor)

def test_invalid_data_processing(data_processor):
    """Test handling of invalid data."""
    with pytest.raises(KeyError):
        data_processor.process_observations([{"properties": {}}])

def test_insufficient_sequence_data(data_processor):
    """Test handling of insufficient data for sequences."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="H")
    data = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 10),
        'dewpoint': np.random.normal(15, 3, 10),
        'windSpeed': np.random.normal(5, 2, 10),
        'windDirection': np.random.uniform(0, 360, 10),
        'barometricPressure': np.random.normal(1013, 5, 10),
        'relativeHumidity': np.random.uniform(60, 90, 10),
        'precipitation': np.random.exponential(1, 10)
    }, index=dates)
    
    X, y = data_processor.create_sequences(data, target_hours=24)
    assert len(X) == 0  # Should return empty arrays when insufficient data 