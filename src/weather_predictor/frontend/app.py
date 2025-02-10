import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pandas as pd

from weather_predictor.api.weather_service import WeatherService
from weather_predictor.data.processor import WeatherDataProcessor
from weather_predictor.models.weather_model import WeatherPredictor

def get_coordinates(city_name: str) -> tuple[float, float]:
    """Get latitude and longitude for a city name.
    
    Args:
        city_name (str): Name of the city
        
    Returns:
        tuple[float, float]: Latitude and longitude
    """
    geolocator = Nominatim(user_agent="weather_predictor/1.0")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"Could not find coordinates for {city_name}")
            return None
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        return None

def create_forecast_plot(forecast_data: dict) -> go.Figure:
    """Create a plotly figure for the forecast.
    
    Args:
        forecast_data (dict): Forecast data from the API
        
    Returns:
        go.Figure: Plotly figure
    """
    periods = forecast_data["properties"]["periods"]
    times = [p["startTime"] for p in periods]
    temps = [p["temperature"] for p in periods]
    
    fig = go.Figure()
    
    # Add temperature line
    fig.add_trace(go.Scatter(
        x=times,
        y=temps,
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#FF9900', width=2),
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title="Temperature Forecast",
        xaxis_title="Time",
        yaxis_title="Temperature (°F)",
        hovermode='x unified',
        template="plotly_dark"
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Weather Predictor",
        page_icon="🌤️",
        layout="wide"
    )
    
    st.title("🌤️ Weather Predictor")
    st.write("Enter a city name to get the weather forecast!")
    
    # Initialize weather service
    weather_service = WeatherService(user_agent="weather_predictor/1.0")
    
    # City input
    city = st.text_input("City name", "San Diego")
    
    if st.button("Get Forecast"):
        with st.spinner("Getting forecast..."):
            # Get coordinates
            coords = get_coordinates(city)
            if coords:
                lat, lon = coords
                
                try:
                    # Get grid point data
                    point_data = weather_service.get_point_data(lat, lon)
                    grid_id = point_data["properties"]["gridId"]
                    grid_x = point_data["properties"]["gridX"]
                    grid_y = point_data["properties"]["gridY"]
                    
                    # Get forecast
                    forecast = weather_service.get_forecast(grid_id, grid_x, grid_y)
                    
                    # Display current conditions
                    current = forecast["properties"]["periods"][0]
                    st.header(f"Current Conditions in {city}")
                    
                    # Create columns for current conditions
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Temperature", f"{current['temperature']}°F")
                    
                    with col2:
                        st.metric("Wind", current["windSpeed"])
                    
                    with col3:
                        st.metric("Forecast", current["shortForecast"])
                    
                    # Display detailed forecast
                    st.header("Detailed Forecast")
                    st.plotly_chart(create_forecast_plot(forecast), use_container_width=True)
                    
                    # Display forecast details
                    for period in forecast["properties"]["periods"][:6]:
                        with st.expander(f"{period['name']} - {period['shortForecast']}"):
                            st.write(f"**Temperature:** {period['temperature']}°F")
                            st.write(f"**Wind:** {period['windSpeed']} {period['windDirection']}")
                            st.write(f"**Detailed Forecast:** {period['detailedForecast']}")
                    
                    # Get alerts if any
                    try:
                        area = point_data["properties"]["forecastZone"].split("/")[-1]
                        alerts = weather_service.get_alerts(area=area)
                        if alerts:
                            st.header("⚠️ Weather Alerts")
                            for alert in alerts:
                                with st.expander(alert["properties"]["event"]):
                                    st.write(f"**Severity:** {alert['properties']['severity']}")
                                    st.write(f"**Description:** {alert['properties']['description']}")
                    except Exception as e:
                        st.warning("Unable to fetch weather alerts at this time. This does not affect the forecast data.")
                        
                except Exception as e:
                    st.error(f"Error getting forecast: {str(e)}")
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "Data provided by [National Weather Service](https://www.weather.gov/). "
        "Created with ❤️ using Streamlit."
    )

if __name__ == "__main__":
    main() 