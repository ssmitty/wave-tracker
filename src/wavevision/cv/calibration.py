"""Calibration helpers for fixed-camera wave analysis."""

from __future__ import annotations

from math import hypot

from wavevision.models import Calibration, Line, Rect


class CalibrationError(ValueError):
    """Raised when calibration geometry is invalid."""


def line_mid_y(line: Line) -> int:
    return int((line.y1 + line.y2) / 2)


def line_length(line: Line) -> float:
    return hypot(line.x2 - line.x1, line.y2 - line.y1)


def calculate_pixels_per_foot(
    reference_segment: Line | None, reference_height_ft: float | None
) -> float | None:
    if reference_segment is None or reference_height_ft is None:
        return None
    if reference_height_ft <= 0:
        return None
    segment_px = line_length(reference_segment)
    if segment_px < 10:
        return None
    return segment_px / reference_height_ft


def pixels_per_foot_at_y(y: int, calibration: Calibration) -> float | None:
    if calibration.pixels_per_foot is None or calibration.reference_segment is None:
        return None
    horizon_y = line_mid_y(calibration.horizon_line)
    shore_y = line_mid_y(calibration.shore_boundary)
    if y <= horizon_y or y >= shore_y:
        return None
    ref_y = line_mid_y(calibration.reference_segment)
    return calibration.pixels_per_foot * (ref_y / max(y, ref_y))


def validate_calibration(calibration: Calibration) -> list[str]:
    errors: list[str] = []
    horizon_y = line_mid_y(calibration.horizon_line)
    shore_y = line_mid_y(calibration.shore_boundary)
    roi = calibration.surf_zone_roi

    if horizon_y >= shore_y:
        errors.append("Horizon must be above the shore line.")
    if roi.y < horizon_y or roi.y + roi.h > shore_y:
        errors.append("Surf zone must sit between horizon and shore.")
    if calibration.reference_segment is None:
        errors.append("Draw a reference segment to enable height estimates.")
    elif not _line_inside_rect(calibration.reference_segment, roi):
        errors.append("Place the reference segment inside the surf zone.")
    if calibration.reference_height_ft is None or calibration.reference_height_ft <= 0:
        errors.append("Enter a positive reference height in feet.")
    if calibration.pixels_per_foot is None:
        errors.append("Reference segment must be at least 10 pixels long.")

    return errors


def default_calibration(video_id: str, width: int, height: int) -> Calibration:
    horizon_y = int(height * 0.28)
    shore_y = int(height * 0.78)
    roi = Rect(
        x=int(width * 0.1),
        y=horizon_y + 10,
        w=int(width * 0.8),
        h=max(20, shore_y - horizon_y - 20),
    )
    reference = Line(
        x1=int(width * 0.5),
        y1=int(height * 0.62),
        x2=int(width * 0.5),
        y2=int(height * 0.54),
    )
    ref_height = 4.0
    return Calibration(
        video_id=video_id,
        frame_width=width,
        frame_height=height,
        horizon_line=Line(0, horizon_y, width, horizon_y),
        shore_boundary=Line(0, shore_y, width, shore_y),
        surf_zone_roi=roi,
        reference_height_ft=ref_height,
        reference_segment=reference,
        pixels_per_foot=calculate_pixels_per_foot(reference, ref_height),
    )


def _line_inside_rect(line: Line, rect: Rect) -> bool:
    return all(
        [
            rect.x <= line.x1 <= rect.x + rect.w,
            rect.y <= line.y1 <= rect.y + rect.h,
            rect.x <= line.x2 <= rect.x + rect.w,
            rect.y <= line.y2 <= rect.y + rect.h,
        ]
    )
