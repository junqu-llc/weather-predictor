# Weather Predictor

A machine learning-based weather prediction system that utilizes the National Weather Service API (api.weather.gov) and PyTorch to forecast weather conditions.

## Project Overview

This project aims to create an accurate weather prediction model by combining historical weather data from the National Weather Service with deep learning techniques implemented in PyTorch. The system will predict various weather parameters for specified locations within the United States.

## Technical Requirements

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- requests
- pandas
- numpy
- scikit-learn
- matplotlib
- python-dotenv (for environment variable management)

### API Requirements
- National Weather Service API (api.weather.gov)
  - No API key required
  - Rate limiting compliance required

## Functional Requirements

### Data Collection
1. Fetch historical weather data from api.weather.gov
   - Temperature
   - Precipitation
   - Wind speed and direction
   - Humidity
   - Atmospheric pressure
2. Support data collection for multiple locations via latitude/longitude
3. Implement proper error handling for API requests
4. Store collected data in structured format (CSV/SQLite)

### Data Processing
1. Clean and preprocess weather data
2. Handle missing values appropriately
3. Normalize/scale features for model input
4. Create sliding window sequences for time-series prediction
5. Split data into training, validation, and test sets

### Model Architecture
1. Implement a deep learning model using PyTorch with:
   - LSTM/GRU layers for sequence processing
   - Multiple fully connected layers
   - Dropout for regularization
2. Support multi-variable prediction
3. Include proper loss functions for regression tasks
4. Implement model checkpointing and early stopping

### Training Pipeline
1. Create data loaders for batch processing
2. Implement training loop with validation
3. Support GPU acceleration when available
4. Log training metrics and visualizations
5. Save and load model checkpoints

### Prediction System
1. Load trained models
2. Make predictions for specified time horizons
3. Support batch prediction
4. Provide confidence intervals for predictions
5. Visualize prediction results

### Performance Requirements
1. Model accuracy metrics:
   - Mean Absolute Error (MAE) < 2°C for temperature
   - Root Mean Square Error (RMSE) tracking
   - R-squared score > 0.8
2. API response handling:
   - Timeout handling
   - Rate limiting compliance
   - Retry mechanism for failed requests

## Project Structure
```
weather-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── checkpoints/
│   └── final/
├── src/
│   ├── data/
│   │   ├── collector.py
│   │   └── processor.py
│   ├── models/
│   │   ├── architecture.py
│   │   └── training.py
│   └── utils/
│       ├── api.py
│       └── visualization.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and configure if needed
5. Run data collection scripts
6. Train the model
7. Make predictions

## Future Enhancements
- Support for additional weather parameters
- Integration with more weather data sources
- Web interface for predictions
- Model ensemble approaches
- Real-time prediction updates
- Automated retraining pipeline

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
