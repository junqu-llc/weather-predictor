import pytest
import torch
import json
from pathlib import Path
from datetime import datetime, timedelta

from weather_predictor.api.weather_service import WeatherService
from weather_predictor.data.processor import WeatherDataProcessor
from weather_predictor.models.weather_model import WeatherLSTM, WeatherPredictor

@pytest.fixture
def sample_weather_data():
    """Sample weather observation data for testing."""
    return {
        "features": [
            {
                "properties": {
                    "timestamp": "2024-02-10T00:00:00+00:00",
                    "temperature": {"value": 20.5},
                    "dewpoint": {"value": 15.2},
                    "windSpeed": {"value": 5.1},
                    "windDirection": {"value": 180},
                    "barometricPressure": {"value": 1013.2},
                    "relativeHumidity": {"value": 75},
                    "precipitation": {"value": 0.0}
                }
            },
            {
                "properties": {
                    "timestamp": "2024-02-10T01:00:00+00:00",
                    "temperature": {"value": 19.8},
                    "dewpoint": {"value": 14.9},
                    "windSpeed": {"value": 4.8},
                    "windDirection": {"value": 185},
                    "barometricPressure": {"value": 1013.0},
                    "relativeHumidity": {"value": 78},
                    "precipitation": {"value": 0.2}
                }
            }
        ]
    }

@pytest.fixture
def mock_api_response(monkeypatch):
    """Mock API responses for testing."""
    def mock_get(*args, **kwargs):
        class MockResponse:
            def __init__(self, json_data, status_code=200):
                self.json_data = json_data
                self.status_code = status_code

            def json(self):
                return self.json_data

            def raise_for_status(self):
                if self.status_code != 200:
                    raise Exception(f"API error: {self.status_code}")

        if "points/" in args[0]:
            return MockResponse({
                "properties": {
                    "gridId": "TEST",
                    "gridX": 50,
                    "gridY": 50
                }
            })
        elif "stations" in args[0]:
            return MockResponse({
                "features": [{
                    "properties": {
                        "stationIdentifier": "TEST_STATION"
                    }
                }]
            })
        elif "observations" in args[0]:
            return MockResponse({
                "features": [
                    {
                        "properties": {
                            "timestamp": "2024-02-10T00:00:00+00:00",
                            "temperature": {"value": 20.5},
                            "dewpoint": {"value": 15.2},
                            "windSpeed": {"value": 5.1},
                            "windDirection": {"value": 180},
                            "barometricPressure": {"value": 1013.2},
                            "relativeHumidity": {"value": 75},
                            "precipitation": {"value": 0.0}
                        }
                    }
                ]
            })

        return MockResponse(None, 404)

    monkeypatch.setattr("requests.Session.get", mock_get)

@pytest.fixture
def weather_service():
    """WeatherService instance for testing."""
    return WeatherService(user_agent="TestApp/1.0")

@pytest.fixture
def data_processor():
    """WeatherDataProcessor instance for testing."""
    return WeatherDataProcessor(sequence_length=24)

@pytest.fixture
def device():
    """PyTorch device for testing."""
    return torch.device("cpu")

@pytest.fixture
def model(device):
    """WeatherLSTM model instance for testing."""
    input_size = 7  # Number of features
    return WeatherLSTM(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        dropout=0.1
    ).to(device)

@pytest.fixture
def predictor(model, device):
    """WeatherPredictor instance for testing."""
    return WeatherPredictor(
        model=model,
        device=device,
        learning_rate=0.001
    ) 