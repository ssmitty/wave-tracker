"""Streamlit entrypoint for Wave Vision."""

from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import streamlit.elements.image as st_image
    from streamlit.elements.lib import image_utils

    if not hasattr(st_image, "image_to_url"):
        st_image.image_to_url = image_utils.image_to_url
except ImportError:
    pass

from wavevision.cv.calibration import (  # noqa: E402
    calculate_pixels_per_foot,
    default_calibration,
    validate_calibration,
)
from wavevision.cv.pipeline import WaveCVPipeline  # noqa: E402
from wavevision.cv.video import (  # noqa: E402
    extract_middle_frame,
    read_video_metadata,
    save_upload,
)
from wavevision.models import Calibration, Line, Rect  # noqa: E402
from wavevision.reporting.surf_report import generate_report  # noqa: E402
from wavevision.weather.open_meteo import (  # noqa: E402
    Location,
    fetch_current_weather,
    search_locations,
)

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # pragma: no cover
    st_canvas = None


st.set_page_config(page_title="Wave Vision", page_icon=None, layout="wide")


def main() -> None:
    st.title("Wave Vision")
    st.caption("Fixed-camera surf video analysis with OpenCV calibration.")
    _init_state()

    with st.sidebar:
        step = st.radio(
            "Workflow",
            ["Upload", "Calibrate", "Analyze", "Report"],
            index=["Upload", "Calibrate", "Analyze", "Report"].index(
                st.session_state.step
            ),
        )
        st.session_state.step = step

    if step == "Upload":
        _upload_step()
    elif step == "Calibrate":
        _calibration_step()
    elif step == "Analyze":
        _analysis_step()
    else:
        _report_step()


def _init_state() -> None:
    defaults = {
        "step": "Upload",
        "video_id": None,
        "video_path": None,
        "metadata": None,
        "calibration_frame": None,
        "calibration": None,
        "location": None,
        "weather": None,
        "analysis_result": None,
        "report_md": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _upload_step() -> None:
    st.subheader("Upload beach footage")
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi"])
    location_query = st.text_input(
        "Location",
        placeholder="Example: Malibu, CA",
        help="Used only for weather and wind context.",
    )

    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        if st.session_state.video_path is None:
            video_id, path = save_upload(uploaded, suffix)
            metadata = read_video_metadata(path)
            middle_frame = extract_middle_frame(path, metadata)
            st.session_state.video_id = video_id
            st.session_state.video_path = path
            st.session_state.metadata = metadata
            st.session_state.calibration_frame = middle_frame
            st.session_state.calibration = default_calibration(
                video_id, metadata.width, metadata.height
            )
            st.session_state.analysis_result = None
            st.session_state.report_md = None

        metadata = st.session_state.metadata
        st.success("Video loaded.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{metadata.duration_sec:.1f} sec")
        col2.metric("FPS", f"{metadata.fps:.1f}")
        col3.metric("Resolution", f"{metadata.width} x {metadata.height}")
        st.image(_frame_to_image(st.session_state.calibration_frame), caption="Calibration frame")

    if location_query and st.button("Find weather location"):
        try:
            matches = search_locations(location_query)
        except Exception:
            matches = []
            st.warning("Location lookup failed. Analysis can still run without weather.")
        if matches:
            st.session_state.location_matches = matches
            st.session_state.location = matches[0]
            st.success(f"Using {matches[0].label}")
        else:
            st.warning("No location match found. Try a city, beach, or coordinates later.")

    matches = st.session_state.get("location_matches", [])
    if matches:
        labels = [match.label for match in matches]
        selected = st.selectbox("Weather match", labels)
        st.session_state.location = matches[labels.index(selected)]

    if st.session_state.video_path and st.button("Continue to calibration", type="primary"):
        st.session_state.step = "Calibrate"
        st.rerun()


def _calibration_step() -> None:
    if not st.session_state.video_path:
        st.info("Upload a video first.")
        return
    if st_canvas is None:
        st.error(
            "Draggable overlays require streamlit-drawable-canvas. "
            "Install requirements, then restart the app."
        )
        return

    st.subheader("Calibrate the scene")
    st.caption("Drag the horizon, shoreline, surf zone, and reference line.")

    frame = st.session_state.calibration_frame
    image = _frame_to_image(frame)
    display_image, scale = _resize_for_canvas(image)
    initial = _canvas_initial_drawing(st.session_state.calibration, scale)

    col_canvas, col_settings = st.columns([2, 1])
    with col_canvas:
        canvas_result = st_canvas(
            fill_color="rgba(0, 180, 0, 0.12)",
            stroke_width=3,
            background_image=display_image,
            update_streamlit=True,
            height=display_image.height,
            width=display_image.width,
            drawing_mode="transform",
            initial_drawing=initial,
            key=f"calibration_canvas_{st.session_state.video_id}",
        )

    with col_settings:
        reference_height = st.number_input(
            "Reference height (ft)",
            min_value=0.1,
            max_value=50.0,
            value=float(st.session_state.calibration.reference_height_ft or 4.0),
            step=0.5,
        )
        sensitivity = st.slider(
            "Detection sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=float(st.session_state.calibration.detection_sensitivity),
            step=0.05,
        )
        beach_faces_deg = st.number_input(
            "Beach faces degrees, optional",
            min_value=0.0,
            max_value=359.0,
            value=270.0,
            step=5.0,
            help="Ocean-facing compass direction. West-facing beach is 270.",
        )

    if canvas_result.json_data:
        calibration = _calibration_from_canvas(
            canvas_result.json_data,
            st.session_state.calibration,
            scale,
            reference_height,
            sensitivity,
            beach_faces_deg,
        )
        errors = validate_calibration(calibration)
        if errors:
            for error in errors:
                st.warning(error)
        else:
            st.success(f"Calibration valid. Scale: {calibration.pixels_per_foot:.1f} px/ft")

        if st.button("Save calibration", type="primary", disabled=bool(errors)):
            st.session_state.calibration = calibration
            st.session_state.analysis_result = None
            st.session_state.report_md = None
            st.success("Calibration saved.")

    if st.button("Run analysis"):
        st.session_state.step = "Analyze"
        st.rerun()


def _analysis_step() -> None:
    if not st.session_state.video_path or not st.session_state.calibration:
        st.info("Upload and calibrate a video first.")
        return

    st.subheader("Analyze waves")
    location = st.session_state.location
    use_openai = st.toggle("Use OpenAI report if API key is available", value=False)

    if st.button("Run CV analysis", type="primary"):
        weather = None
        if isinstance(location, Location):
            with st.spinner("Fetching wind and weather..."):
                weather = fetch_current_weather(
                    location.latitude,
                    location.longitude,
                    location.label,
                    st.session_state.calibration.beach_faces_deg,
                )
        st.session_state.weather = weather

        with st.spinner("Processing video frames..."):
            pipeline = WaveCVPipeline()
            result = pipeline.analyze(
                st.session_state.video_path,
                st.session_state.metadata,
                st.session_state.calibration,
                weather,
            )
            st.session_state.analysis_result = result
            st.session_state.report_md = generate_report(result, use_openai=use_openai)

    result = st.session_state.analysis_result
    if result:
        _render_dashboard(result)
        if st.button("Continue to report"):
            st.session_state.step = "Report"
            st.rerun()


def _report_step() -> None:
    result = st.session_state.analysis_result
    if not result:
        st.info("Run analysis first.")
        return

    st.subheader("Surf report")
    if st.session_state.report_md is None:
        st.session_state.report_md = generate_report(result)
    st.markdown(st.session_state.report_md)

    metrics_json = json.dumps(result.to_dict(), indent=2, default=str)
    st.download_button("Download report markdown", st.session_state.report_md, "surf_report.md")
    st.download_button("Download metrics JSON", metrics_json, "wave_metrics.json")
    st.download_button(
        "Download calibration JSON",
        json.dumps(st.session_state.calibration.to_dict(), indent=2),
        "calibration.json",
    )

    with st.expander("Raw metrics"):
        st.json(result.to_dict())


def _render_dashboard(result: object) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Breaking events", result.wave_count)
    col2.metric("Average height", _metric_ft(result.avg_height_ft))
    col3.metric("Max height", _metric_ft(result.max_height_ft))
    col4.metric("Confidence", f"{result.avg_confidence:.2f}")

    if result.avg_velocity_ft_s is not None:
        st.metric("Average tracked velocity", f"{result.avg_velocity_ft_s:.1f} ft/s")

    if result.debug_frames:
        st.subheader("Annotated frames")
        columns = st.columns(2)
        for index, encoded in enumerate(result.debug_frames):
            image = Image.open(BytesIO(base64.b64decode(encoded)))
            columns[index % 2].image(image, caption=f"Debug frame {index + 1}")

    if result.events:
        st.subheader("Wave event timeline")
        st.bar_chart(
            {
                "start_sec": [event.start_sec for event in result.events],
                "confidence": [event.avg_confidence for event in result.events],
            },
            x="start_sec",
            y="confidence",
        )


def _frame_to_image(frame: object) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _resize_for_canvas(image: Image.Image, max_width: int = 980) -> tuple[Image.Image, float]:
    if image.width <= max_width:
        return image, 1.0
    scale = max_width / image.width
    height = int(image.height * scale)
    return image.resize((max_width, height)), scale


def _canvas_initial_drawing(calibration: Calibration, scale: float) -> dict[str, list[dict]]:
    roi = calibration.surf_zone_roi
    ref = calibration.reference_segment
    objects = [
        _line_object("horizon", calibration.horizon_line, scale, "#1f77b4"),
        _line_object("shore", calibration.shore_boundary, scale, "#8b5a2b"),
        {
            "type": "rect",
            "name": "roi",
            "left": roi.x * scale,
            "top": roi.y * scale,
            "width": roi.w * scale,
            "height": roi.h * scale,
            "fill": "rgba(0, 180, 0, 0.12)",
            "stroke": "#1f8f3a",
            "strokeWidth": 3,
        },
    ]
    if ref:
        objects.append(_line_object("reference", ref, scale, "#d6b300"))
    return {
        "version": "4.4.0",
        "objects": objects,
    }


def _line_object(name: str, line: Line, scale: float, color: str) -> dict:
    left = min(line.x1, line.x2) * scale
    top = min(line.y1, line.y2) * scale
    return {
        "type": "line",
        "name": name,
        "x1": line.x1 * scale - left,
        "y1": line.y1 * scale - top,
        "x2": line.x2 * scale - left,
        "y2": line.y2 * scale - top,
        "left": left,
        "top": top,
        "stroke": color,
        "strokeWidth": 3,
    }


def _calibration_from_canvas(
    json_data: dict,
    current: Calibration,
    scale: float,
    reference_height_ft: float,
    sensitivity: float,
    beach_faces_deg: float,
) -> Calibration:
    objects = [obj for obj in json_data.get("objects", []) if obj]
    by_name = {obj.get("name"): obj for obj in objects}
    horizon = _line_from_object(by_name.get("horizon"), current.horizon_line, scale)
    shore = _line_from_object(by_name.get("shore"), current.shore_boundary, scale)
    roi = _rect_from_object(by_name.get("roi"), current.surf_zone_roi, scale)
    reference = _line_from_object(by_name.get("reference"), current.reference_segment, scale)
    pixels_per_foot = calculate_pixels_per_foot(reference, reference_height_ft)
    return Calibration(
        video_id=current.video_id,
        frame_width=current.frame_width,
        frame_height=current.frame_height,
        horizon_line=horizon,
        shore_boundary=shore,
        surf_zone_roi=roi,
        reference_height_ft=reference_height_ft,
        reference_segment=reference,
        pixels_per_foot=pixels_per_foot,
        detection_sensitivity=sensitivity,
        beach_faces_deg=beach_faces_deg,
    )


def _line_from_object(obj: dict | None, fallback: Line | None, scale: float) -> Line:
    if obj is None:
        if fallback is None:
            return Line(0, 0, 0, 0)
        return fallback
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))
    x1 = left + float(obj.get("x1", 0)) * scale_x
    y1 = top + float(obj.get("y1", 0)) * scale_y
    x2 = left + float(obj.get("x2", 0)) * scale_x
    y2 = top + float(obj.get("y2", 0)) * scale_y
    return Line(
        x1=int(x1 / scale),
        y1=int(y1 / scale),
        x2=int(x2 / scale),
        y2=int(y2 / scale),
    )


def _rect_from_object(obj: dict | None, fallback: Rect, scale: float) -> Rect:
    if obj is None:
        return fallback
    left = float(obj.get("left", fallback.x * scale))
    top = float(obj.get("top", fallback.y * scale))
    width = float(obj.get("width", fallback.w * scale)) * float(obj.get("scaleX", 1))
    height = float(obj.get("height", fallback.h * scale)) * float(obj.get("scaleY", 1))
    return Rect(
        x=max(0, int(left / scale)),
        y=max(0, int(top / scale)),
        w=max(1, int(width / scale)),
        h=max(1, int(height / scale)),
    )


def _metric_ft(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f} ft"


if __name__ == "__main__":
    main()
