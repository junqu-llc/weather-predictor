import pytest
import requests
import subprocess
import time
import os
import signal
from pathlib import Path
import json

@pytest.fixture(scope="module")
def streamlit_server():
    """Start Streamlit server for testing."""
    # Start the Streamlit server
    process = subprocess.Popen(
        ["streamlit", "run", "src/weather_predictor/frontend/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait for server to start
    time.sleep(5)
    
    yield process
    
    # Cleanup: Kill the server process
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

def test_api_health(streamlit_server):
    """Test if the Streamlit server is running."""
    response = requests.get("http://localhost:8501/healthz")
    assert response.status_code == 200

def test_weather_service():
    """Test the WeatherService API integration."""
    from weather_predictor.api.weather_service import WeatherService
    
    weather_service = WeatherService(user_agent="weather_predictor_test/1.0")
    
    # Test getting point data
    point_data = weather_service.get_point_data(32.7157, -117.1611)  # San Diego coordinates
    assert point_data["properties"]["gridId"] is not None
    assert point_data["properties"]["gridX"] is not None
    assert point_data["properties"]["gridY"] is not None
    
    # Test getting forecast
    grid_id = point_data["properties"]["gridId"]
    grid_x = point_data["properties"]["gridX"]
    grid_y = point_data["properties"]["gridY"]
    
    forecast = weather_service.get_forecast(grid_id, grid_x, grid_y)
    assert forecast["properties"]["periods"] is not None
    assert len(forecast["properties"]["periods"]) > 0

def test_weather_service_error_handling():
    """Test error handling in WeatherService."""
    from weather_predictor.api.weather_service import WeatherService
    
    weather_service = WeatherService(user_agent="weather_predictor_test/1.0")
    
    # Test invalid coordinates
    with pytest.raises(Exception):
        weather_service.get_point_data(1000, 1000)  # Invalid coordinates
    
    # Test invalid grid points
    with pytest.raises(Exception):
        weather_service.get_forecast("INVALID", 0, 0)

def test_geocoding():
    """Test the geocoding functionality."""
    from geopy.geocoders import Nominatim
    
    geolocator = Nominatim(user_agent="weather_predictor_test/1.0")
    location = geolocator.geocode("San Diego")
    
    assert location is not None
    assert location.latitude is not None
    assert location.longitude is not None
    assert abs(location.latitude - 32.7157) < 0.1  # Approximate match
    assert abs(location.longitude - (-117.1611)) < 0.1  # Approximate match

def test_geocoding_error_handling():
    """Test error handling in geocoding."""
    from weather_predictor.frontend.app import get_coordinates
    
    # Test invalid city name
    result = get_coordinates("NonexistentCity12345")
    assert result is None

def test_forecast_plot():
    """Test the forecast plotting functionality."""
    import plotly.graph_objects as go
    from weather_predictor.frontend.app import create_forecast_plot
    
    # Create sample forecast data
    forecast_data = {
        "properties": {
            "periods": [
                {
                    "startTime": "2024-02-10T00:00:00+00:00",
                    "temperature": 72,
                    "shortForecast": "Clear"
                },
                {
                    "startTime": "2024-02-10T01:00:00+00:00",
                    "temperature": 70,
                    "shortForecast": "Partly Cloudy"
                }
            ]
        }
    }
    
    # Create plot
    fig = create_forecast_plot(forecast_data)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # One trace for temperature
    assert fig.data[0].mode == "lines+markers"

def test_forecast_plot_empty_data():
    """Test plotting functionality with empty data."""
    import plotly.graph_objects as go
    from weather_predictor.frontend.app import create_forecast_plot
    
    # Create empty forecast data
    forecast_data = {
        "properties": {
            "periods": []
        }
    }
    
    # Create plot
    fig = create_forecast_plot(forecast_data)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # Should still create an empty trace
    assert len(fig.data[0].x) == 0  # No data points

def test_weather_alerts():
    """Test weather alerts functionality."""
    from weather_predictor.api.weather_service import WeatherService
    
    weather_service = WeatherService(user_agent="weather_predictor_test/1.0")
    
    # Test getting alerts for a specific area
    alerts = weather_service.get_alerts(area="CA")
    assert isinstance(alerts, list)  # Should always return a list, even if empty 