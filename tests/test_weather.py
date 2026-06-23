from wavevision.weather.open_meteo import degrees_to_cardinal, wind_surf_relation


def test_degrees_to_cardinal() -> None:
    assert degrees_to_cardinal(0) == "N"
    assert degrees_to_cardinal(90) == "E"
    assert degrees_to_cardinal(225) == "SW"


def test_wind_surf_relation() -> None:
    assert wind_surf_relation(270, 270) == "onshore"
    assert wind_surf_relation(90, 270) == "offshore"
    assert wind_surf_relation(180, 270) == "cross-shore"
