"""Template and optional OpenAI surf report generation."""

from __future__ import annotations

import os

from wavevision.models import AnalysisResult


def generate_report(result: AnalysisResult, use_openai: bool = False) -> str:
    if use_openai and os.getenv("OPENAI_API_KEY"):
        try:
            return _generate_openai_report(result)
        except Exception:
            return generate_template_report(result)
    return generate_template_report(result)


def generate_template_report(result: AnalysisResult) -> str:
    location = (
        result.weather.location_name
        if result.weather and result.weather.location_name
        else "Beach footage"
    )
    duration = _format_duration(result.duration_sec)
    confidence_label = _confidence_label(result.avg_confidence)

    lines = [f"## {location} surf analysis from {duration} of beach footage", ""]
    lines.extend(["**Conditions snapshot**"])
    lines.append(f"- Breaking events: {result.wave_count}")
    if result.calibration_valid and result.avg_height_ft is not None:
        lines.append(f"- Average height: about {result.avg_height_ft:.1f} ft")
    else:
        lines.append("- Wave heights were not estimated for this clip.")
    if result.calibration_valid and result.max_height_ft is not None:
        lines.append(f"- Max height: about {result.max_height_ft:.1f} ft")
    if result.avg_velocity_ft_s is not None:
        lines.append(f"- Average tracked velocity: {result.avg_velocity_ft_s:.1f} ft/s")
    if result.weather:
        weather = result.weather
        if weather.wind_mph is not None and weather.wind_cardinal:
            lines.append(f"- Wind: {weather.wind_mph:.1f} mph {weather.wind_cardinal}")
        if weather.temp_f is not None:
            lines.append(f"- Air temperature: {weather.temp_f:.0f} F")
        if weather.weather_label:
            lines.append(f"- Weather: {weather.weather_label}")

    lines.extend(["", "**Wave activity**"])
    lines.append(f"{result.wave_count} breaking events detected in the clip.")
    if result.set_intervals_sec:
        avg_interval = sum(result.set_intervals_sec) / len(result.set_intervals_sec)
        lines.append(f"Average interval between tracked events was {avg_interval:.1f} seconds.")
    if not result.calibration_valid:
        lines.append("Height estimates unavailable; calibration needs a valid reference.")

    lines.extend(["", "**Confidence and limitations**"])
    lines.append(
        f"{confidence_label.capitalize()} confidence ({result.avg_confidence:.2f}). "
        "Estimates depend on the fixed camera angle and the visible break zone."
    )
    if result.avg_confidence < 0.5:
        lines.append("Low contrast, glare, or incomplete calibration may have reduced detection.")

    if result.weather and result.weather.surf_relation:
        lines.extend(["", "**Wind context**"])
        lines.append(_wind_context(result.weather.surf_relation))

    return "\n".join(lines)


def _generate_openai_report(result: AnalysisResult) -> str:
    from openai import OpenAI

    client = OpenAI()
    payload = result.to_dict()
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Write a neutral surf analysis from structured metrics only. "
                    "Do not invent swell, tide, hazards, crowd, or spot details. "
                    "If a metric is missing, omit it or say it is not available."
                ),
            },
            {"role": "user", "content": str(payload)},
        ],
    )
    return response.output_text


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} sec"
    return f"{seconds / 60:.1f} min"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.5:
        return "moderate"
    return "low"


def _wind_context(relation: str) -> str:
    if relation == "offshore":
        return "Offshore winds may groom wave faces during the recording window."
    if relation == "onshore":
        return "Onshore winds may add chop during the recording window."
    return "Cross-shore winds; texture may vary across the visible surf zone."
