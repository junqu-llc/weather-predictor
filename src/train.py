import argparse
import logging
import torch
from datetime import datetime, timedelta
import os
from pathlib import Path

from api.weather_service import WeatherService
from data.processor import WeatherDataProcessor
from models.weather_model import WeatherLSTM, WeatherPredictor

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def train(args):
    """Main training function.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize services
    weather_service = WeatherService(user_agent=args.user_agent)
    data_processor = WeatherDataProcessor(sequence_length=args.sequence_length)
    
    # Get training data
    logger.info("Collecting historical weather data...")
    
    # Get grid coordinates for the location
    point_data = weather_service.get_point_data(args.latitude, args.longitude)
    grid_id = point_data['properties']['gridId']
    grid_x = point_data['properties']['gridX']
    grid_y = point_data['properties']['gridY']
    
    # Get nearby stations
    stations = weather_service.get_stations(grid_id, grid_x, grid_y)
    station_id = stations[0]['properties']['stationIdentifier']
    
    # Get historical observations
    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.history_days)
    
    observations = weather_service.get_station_observations(
        station_id,
        start=start_time,
        end=end_time
    )
    
    # Process data
    logger.info("Processing weather data...")
    data = data_processor.process_observations(observations)
    
    # Prepare data for training
    train_loader, val_loader = data_processor.prepare_data(
        data,
        target_hours=args.forecast_hours,
        batch_size=args.batch_size
    )
    
    # Initialize model
    input_size = len(data_processor.feature_columns)
    model = WeatherLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    predictor = WeatherPredictor(
        model=model,
        device=device,
        learning_rate=args.learning_rate
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = predictor.train_epoch(train_loader)
        val_loss, val_mae = predictor.validate(val_loader)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val MAE: {val_mae:.4f}°C"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_dir = Path("models/checkpoints")
            model_dir.mkdir(parents=True, exist_ok=True)
            predictor.save_model(model_dir / "best_model.pth")
            
    logger.info("Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Train weather prediction model")
    
    # Data collection arguments
    parser.add_argument("--latitude", type=float, required=True,
                      help="Latitude of the location")
    parser.add_argument("--longitude", type=float, required=True,
                      help="Longitude of the location")
    parser.add_argument("--history-days", type=int, default=365,
                      help="Number of days of historical data to use")
    parser.add_argument("--user-agent", type=str, required=True,
                      help="User agent string for API requests")
    
    # Model arguments
    parser.add_argument("--hidden-size", type=int, default=128,
                      help="Number of hidden units in LSTM")
    parser.add_argument("--num-layers", type=int, default=2,
                      help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--sequence-length", type=int, default=24,
                      help="Number of time steps to use for prediction")
    parser.add_argument("--forecast-hours", type=int, default=24,
                      help="Number of hours ahead to predict")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                      help="Learning rate")
    
    args = parser.parse_args()
    
    setup_logging()
    train(args)

if __name__ == "__main__":
    main() 