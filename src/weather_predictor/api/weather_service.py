from typing import Dict, List, Optional, Union
import requests
from datetime import datetime, timedelta
import logging

class WeatherService:
    """Interface for the National Weather Service API."""
    
    BASE_URL = "https://api.weather.gov"
    
    def __init__(self, user_agent: str):
        """Initialize the weather service.
        
        Args:
            user_agent (str): Required user agent string for API identification
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/geo+json"
        })
        self.logger = logging.getLogger(__name__)

    def get_point_data(self, latitude: float, longitude: float) -> Dict:
        """Get grid point data for a specific lat/lon location.
        
        Args:
            latitude (float): Location latitude
            longitude (float): Location longitude
            
        Returns:
            Dict: Point metadata including grid coordinates
        """
        endpoint = f"{self.BASE_URL}/points/{latitude},{longitude}"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()

    def get_forecast(self, grid_id: str, grid_x: int, grid_y: int, hourly: bool = False) -> Dict:
        """Get forecast data for a specific grid point.
        
        Args:
            grid_id (str): NWS grid ID (e.g., 'SGX')
            grid_x (int): Grid X coordinate
            grid_y (int): Grid Y coordinate
            hourly (bool): If True, get hourly forecast instead of daily
            
        Returns:
            Dict: Forecast data
        """
        forecast_type = "forecast/hourly" if hourly else "forecast"
        endpoint = f"{self.BASE_URL}/gridpoints/{grid_id}/{grid_x},{grid_y}/{forecast_type}"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()

    def get_stations(self, grid_id: str, grid_x: int, grid_y: int) -> List[Dict]:
        """Get observation stations near a grid point.
        
        Args:
            grid_id (str): NWS grid ID
            grid_x (int): Grid X coordinate
            grid_y (int): Grid Y coordinate
            
        Returns:
            List[Dict]: List of nearby observation stations
        """
        endpoint = f"{self.BASE_URL}/gridpoints/{grid_id}/{grid_x},{grid_y}/stations"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()["features"]

    def get_station_observations(
        self, 
        station_id: str, 
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict]:
        """Get observations from a specific weather station.
        
        Args:
            station_id (str): Station identifier
            start (datetime, optional): Start time for observations
            end (datetime, optional): End time for observations
            
        Returns:
            List[Dict]: List of observations
        """
        endpoint = f"{self.BASE_URL}/stations/{station_id}/observations"
        params = {}
        
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()
            
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()["features"]

    def get_alerts(
        self,
        area: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        active: bool = True
    ) -> List[Dict]:
        """Get weather alerts for a specific area.
        
        Args:
            area (str, optional): State/territory code
            region (str, optional): Marine region code
            zone (str, optional): Zone ID
            active (bool): If True, only get active alerts
            
        Returns:
            List[Dict]: List of alerts
        """
        if active:
            endpoint = f"{self.BASE_URL}/alerts/active"
        else:
            endpoint = f"{self.BASE_URL}/alerts"
            
        params = {}
        if area:
            params["area"] = area
        if region:
            params["region"] = region
        if zone:
            params["zone"] = zone
            
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()["features"] 