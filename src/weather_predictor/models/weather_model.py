import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

class WeatherLSTM(nn.Module):
    """LSTM-based weather prediction model."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """Initialize the model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Initial hidden state
            
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                Predictions and final hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use only the last output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        predictions = self.fc(last_output)
        
        return predictions, hidden
        
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initial hidden state
        """
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )

class WeatherPredictor:
    """Weather prediction model manager."""
    
    def __init__(
        self,
        model: WeatherLSTM,
        device: torch.device,
        learning_rate: float = 0.001
    ):
        """Initialize the predictor.
        
        Args:
            model (WeatherLSTM): The LSTM model
            device (torch.device): Device to run the model on
            learning_rate (float): Learning rate for optimization
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            hidden = self.model.init_hidden(data.size(0), self.device)
            
            output, _ = self.model(data, hidden)
            loss = self.criterion(output.squeeze(), target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Tuple[float, float]: Average loss and MAE for validation set
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                hidden = self.model.init_hidden(data.size(0), self.device)
                
                output, _ = self.model(data, hidden)
                loss = self.criterion(output.squeeze(), target)
                mae = torch.mean(torch.abs(output.squeeze() - target))
                
                total_loss += loss.item()
                total_mae += mae.item()
                
        return total_loss / len(val_loader), total_mae / len(val_loader)
        
    def predict(self, input_sequence: torch.Tensor) -> float:
        """Make a prediction.
        
        Args:
            input_sequence (torch.Tensor): Input sequence of shape (1, seq_len, input_size)
            
        Returns:
            float: Predicted temperature
        """
        self.model.eval()
        
        with torch.no_grad():
            input_sequence = input_sequence.to(self.device)
            hidden = self.model.init_hidden(1, self.device)
            
            output, _ = self.model(input_sequence, hidden)
            return output.item()
            
    def save_model(self, path: str):
        """Save the model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path: str):
        """Load the model.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 