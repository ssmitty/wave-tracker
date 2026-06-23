"""Open-Meteo weather and geocoding client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from wavevision.models import WeatherSnapshot


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_LABELS = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    51: "light drizzle",
    61: "rain",
    80: "showers",
    95: "thunderstorm",
}

CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    country: str | None = None

    @property
    def label(self) -> str:
        if self.country:
            return f"{self.name}, {self.country}"
        return self.name


def search_locations(query: str) -> list[Location]:
    if not query.strip():
        return []
    response = requests.get(
        GEOCODE_URL,
        params={"name": query, "count": 5, "language": "en", "format": "json"},
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        Location(
            name=item["name"],
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
            country=item.get("country"),
        )
        for item in payload.get("results", [])
    ]


def fetch_current_weather(
    latitude: float,
    longitude: float,
    location_name: str | None = None,
    beach_faces_deg: float | None = None,
) -> WeatherSnapshot | None:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m,wind_direction_10m,weather_code",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    }
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=5)
        response.raise_for_status()
        current = response.json().get("current", {})
    except requests.RequestException:
        return None

    wind_deg = _optional_float(current.get("wind_direction_10m"))
    return WeatherSnapshot(
        temp_f=_optional_float(current.get("temperature_2m")),
        wind_mph=_optional_float(current.get("wind_speed_10m")),
        wind_deg=wind_deg,
        wind_cardinal=degrees_to_cardinal(wind_deg) if wind_deg is not None else None,
        weather_label=WEATHER_LABELS.get(
            int(current.get("weather_code", -1)), "conditions unclear"
        ),
        fetched_at_utc=datetime.now(timezone.utc),
        location_name=location_name,
        surf_relation=wind_surf_relation(wind_deg, beach_faces_deg)
        if wind_deg is not None and beach_faces_deg is not None
        else None,
    )


def degrees_to_cardinal(degrees: float) -> str:
    return CARDINALS[round(degrees / 45) % 8]


def wind_surf_relation(wind_from_deg: float, beach_faces_deg: float) -> str:
    delta = abs((wind_from_deg - beach_faces_deg + 180) % 360 - 180)
    if delta < 45:
        return "onshore"
    if delta > 135:
        return "offshore"
    return "cross-shore"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
