import pytest
import torch
import torch.nn as nn
import os
from pathlib import Path

def test_weather_lstm_initialization(model):
    """Test WeatherLSTM model initialization."""
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.fc, nn.Sequential)
    assert model.hidden_size == 32
    assert model.num_layers == 1

def test_weather_lstm_forward(model, device):
    """Test forward pass through the LSTM model."""
    batch_size = 16
    seq_length = 24
    input_size = 7
    
    # Create sample input
    x = torch.randn(batch_size, seq_length, input_size).to(device)
    hidden = model.init_hidden(batch_size, device)
    
    # Forward pass
    output, final_hidden = model(x, hidden)
    
    assert output.shape == (batch_size, 1)
    assert len(final_hidden) == 2
    assert final_hidden[0].shape == (1, batch_size, 32)  # Hidden state
    assert final_hidden[1].shape == (1, batch_size, 32)  # Cell state

def test_weather_lstm_init_hidden(model, device):
    """Test hidden state initialization."""
    batch_size = 16
    hidden = model.init_hidden(batch_size, device)
    
    assert len(hidden) == 2
    assert hidden[0].shape == (1, batch_size, 32)
    assert hidden[1].shape == (1, batch_size, 32)
    assert hidden[0].device == device
    assert hidden[1].device == device

def test_weather_predictor_initialization(predictor, model, device):
    """Test WeatherPredictor initialization."""
    assert predictor.model == model
    assert predictor.device == device
    assert isinstance(predictor.criterion, nn.MSELoss)
    assert isinstance(predictor.optimizer, torch.optim.Adam)
    assert predictor.optimizer.defaults['lr'] == 0.001

def test_weather_predictor_train_epoch(predictor, device):
    """Test training for one epoch."""
    # Create sample training data
    batch_size = 16
    seq_length = 24
    input_size = 7
    num_batches = 10
    
    dataset = torch.utils.data.TensorDataset(
        torch.randn(batch_size * num_batches, seq_length, input_size),
        torch.randn(batch_size * num_batches)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Train for one epoch
    loss = predictor.train_epoch(train_loader)
    
    assert isinstance(loss, float)
    assert loss > 0  # Loss should be positive

def test_weather_predictor_validate(predictor, device):
    """Test model validation."""
    # Create sample validation data
    batch_size = 16
    seq_length = 24
    input_size = 7
    num_batches = 5
    
    dataset = torch.utils.data.TensorDataset(
        torch.randn(batch_size * num_batches, seq_length, input_size),
        torch.randn(batch_size * num_batches)
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Validate
    val_loss, val_mae = predictor.validate(val_loader)
    
    assert isinstance(val_loss, float)
    assert isinstance(val_mae, float)
    assert val_loss > 0
    assert val_mae > 0

def test_weather_predictor_predict(predictor, device):
    """Test making predictions."""
    # Create sample input sequence
    seq_length = 24
    input_size = 7
    input_sequence = torch.randn(1, seq_length, input_size)
    
    # Make prediction
    prediction = predictor.predict(input_sequence)
    
    assert isinstance(prediction, float)

def test_weather_predictor_save_load(predictor, tmp_path):
    """Test model saving and loading."""
    # Save model
    save_path = tmp_path / "test_model.pth"
    predictor.save_model(save_path)
    
    # Check if file exists
    assert save_path.exists()
    
    # Load model
    predictor.load_model(save_path)
    
    # Make sure the model can still make predictions
    seq_length = 24
    input_size = 7
    input_sequence = torch.randn(1, seq_length, input_size)
    prediction = predictor.predict(input_sequence)
    assert isinstance(prediction, float)

def test_gradient_flow(predictor, device):
    """Test gradient flow through the model."""
    # Create sample batch
    batch_size = 4
    seq_length = 24
    input_size = 7
    
    data = torch.randn(batch_size, seq_length, input_size).to(device)
    target = torch.randn(batch_size).to(device)
    
    # Forward pass
    predictor.model.train()
    predictor.optimizer.zero_grad()
    hidden = predictor.model.init_hidden(batch_size, device)
    
    output, _ = predictor.model(data, hidden)
    loss = predictor.criterion(output.squeeze(), target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist and are not zero
    for name, param in predictor.model.named_parameters():
        assert param.grad is not None
        assert torch.any(param.grad != 0).item()

def test_model_overfitting(predictor, device):
    """Test if model can overfit to a small dataset."""
    # Create a very small dataset
    batch_size = 2
    seq_length = 24
    input_size = 7
    
    data = torch.randn(batch_size, seq_length, input_size).to(device)
    target = torch.randn(batch_size).to(device)
    
    dataset = torch.utils.data.TensorDataset(data, target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Train for several epochs
    initial_loss = None
    final_loss = None
    
    for epoch in range(50):
        loss = predictor.train_epoch(loader)
        if initial_loss is None:
            initial_loss = loss
        final_loss = loss
    
    # Loss should decrease significantly
    assert final_loss < initial_loss
    assert final_loss < 0.1  # Should be able to overfit to tiny dataset 