import pytest
from datetime import datetime
import requests

def test_weather_service_initialization(weather_service):
    """Test WeatherService initialization."""
    assert weather_service.session.headers["User-Agent"] == "TestApp/1.0"
    assert weather_service.session.headers["Accept"] == "application/geo+json"

def test_get_point_data(weather_service, mock_api_response):
    """Test getting point data for a location."""
    data = weather_service.get_point_data(32.7157, -117.1611)
    assert data["properties"]["gridId"] == "TEST"
    assert data["properties"]["gridX"] == 50
    assert data["properties"]["gridY"] == 50

def test_get_stations(weather_service, mock_api_response):
    """Test getting nearby weather stations."""
    stations = weather_service.get_stations("TEST", 50, 50)
    assert len(stations) == 1
    assert stations[0]["properties"]["stationIdentifier"] == "TEST_STATION"

def test_get_station_observations(weather_service, mock_api_response):
    """Test getting station observations."""
    observations = weather_service.get_station_observations(
        "TEST_STATION",
        start=datetime(2024, 2, 10),
        end=datetime(2024, 2, 11)
    )
    assert len(observations) == 1
    assert observations[0]["properties"]["temperature"]["value"] == 20.5

def test_get_alerts(weather_service, mock_api_response):
    """Test getting weather alerts."""
    alerts = weather_service.get_alerts(area="CA")
    assert isinstance(alerts, list)

def test_invalid_coordinates(weather_service):
    """Test handling of invalid coordinates."""
    with pytest.raises(requests.exceptions.HTTPError):
        weather_service.get_point_data(1000, 1000)  # Invalid coordinates

def test_invalid_station(weather_service):
    """Test handling of invalid station ID."""
    with pytest.raises(requests.exceptions.HTTPError):
        weather_service.get_station_observations("INVALID_STATION") 