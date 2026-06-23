"""Shared typed models for the Wave Vision app."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Line:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Calibration:
    video_id: str
    frame_width: int
    frame_height: int
    horizon_line: Line
    shore_boundary: Line
    surf_zone_roi: Rect
    reference_height_ft: float | None
    reference_segment: Line | None
    pixels_per_foot: float | None
    detection_sensitivity: float = 0.55
    beach_faces_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class WaveDetection:
    frame_index: int
    time_sec: float
    track_id: int
    bbox: tuple[int, int, int, int]
    height_ft: float | None
    confidence: float


@dataclass(frozen=True)
class WaveEvent:
    track_id: int
    start_sec: float
    end_sec: float
    max_height_ft: float | None
    avg_confidence: float
    velocity_ft_s: float | None


@dataclass(frozen=True)
class WeatherSnapshot:
    temp_f: float | None
    wind_mph: float | None
    wind_deg: float | None
    wind_cardinal: str | None
    weather_label: str | None
    fetched_at_utc: datetime
    location_name: str | None = None
    surf_relation: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    video_id: str
    wave_count: int
    avg_height_ft: float | None
    max_height_ft: float | None
    avg_velocity_ft_s: float | None
    avg_confidence: float
    set_intervals_sec: list[float]
    detections: list[WaveDetection]
    events: list[WaveEvent]
    debug_frames: list[str]
    weather: WeatherSnapshot | None
    calibration_valid: bool
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
