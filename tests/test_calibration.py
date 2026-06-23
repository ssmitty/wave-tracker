from wavevision.cv.calibration import (
    calculate_pixels_per_foot,
    default_calibration,
    pixels_per_foot_at_y,
    validate_calibration,
)
from wavevision.models import Line


def test_default_calibration_is_valid() -> None:
    calibration = default_calibration("video", 1280, 720)

    assert validate_calibration(calibration) == []
    assert calibration.pixels_per_foot is not None


def test_pixels_per_foot_rejects_tiny_reference() -> None:
    value = calculate_pixels_per_foot(Line(0, 0, 1, 1), 4.0)

    assert value is None


def test_pixels_per_foot_at_y_rejects_outside_band() -> None:
    calibration = default_calibration("video", 1280, 720)

    assert pixels_per_foot_at_y(1, calibration) is None
