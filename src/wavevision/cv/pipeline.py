"""OpenCV wave detection pipeline for fixed-camera beach footage."""

from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from wavevision.cv.calibration import pixels_per_foot_at_y, validate_calibration
from wavevision.models import (
    AnalysisResult,
    Calibration,
    VideoMetadata,
    WaveDetection,
    WaveEvent,
    WeatherSnapshot,
)


@dataclass(frozen=True)
class PipelineConfig:
    sample_fps: float = 3.0
    min_area_px: int = 140
    max_debug_frames: int = 8
    min_confidence: float = 0.3
    max_track_distance_px: int = 90
    max_wave_height_ft: float = 25.0


class WaveCVPipeline:
    """Detect whitewater-like wave events using classical computer vision."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def analyze(
        self,
        video_path: str,
        metadata: VideoMetadata,
        calibration: Calibration,
        weather: WeatherSnapshot | None = None,
    ) -> AnalysisResult:
        validation_errors = validate_calibration(calibration)
        calibration_valid = not validation_errors

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Video could not be opened for analysis.")

        frame_step = max(1, int(metadata.fps / self._config.sample_fps))
        tracker = _CentroidTracker(self._config.max_track_distance_px)
        detections: list[WaveDetection] = []
        debug_frames: list[str] = []
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % frame_step != 0:
                    frame_index += 1
                    continue

                frame_detections = self._detect_frame(
                    frame=frame,
                    frame_index=frame_index,
                    fps=metadata.fps,
                    calibration=calibration,
                    tracker=tracker,
                    calibration_valid=calibration_valid,
                )
                detections.extend(frame_detections)

                if (
                    frame_detections
                    and len(debug_frames) < self._config.max_debug_frames
                ):
                    debug_frames.append(
                        _encode_debug_frame(frame.copy(), calibration, frame_detections)
                    )
                frame_index += 1
        finally:
            cap.release()

        events = _group_events(detections)
        heights = [
            event.max_height_ft
            for event in events
            if event.max_height_ft is not None and event.avg_confidence >= 0.5
        ]
        velocities = [
            event.velocity_ft_s for event in events if event.velocity_ft_s is not None
        ]
        confidences = [d.confidence for d in detections]

        return AnalysisResult(
            video_id=calibration.video_id,
            wave_count=len(events),
            avg_height_ft=_mean(heights),
            max_height_ft=max(heights) if heights else None,
            avg_velocity_ft_s=_mean(velocities),
            avg_confidence=_mean(confidences) or 0.0,
            set_intervals_sec=_set_intervals(events),
            detections=detections,
            events=events,
            debug_frames=debug_frames,
            weather=weather,
            calibration_valid=calibration_valid,
            duration_sec=metadata.duration_sec,
        )

    def _detect_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        calibration: Calibration,
        tracker: "_CentroidTracker",
        calibration_valid: bool,
    ) -> list[WaveDetection]:
        roi = calibration.surf_zone_roi
        roi_frame = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        mask = _whitewater_mask(roi_frame, calibration.detection_sensitivity)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes: list[tuple[int, int, int, int, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._config.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 6:
                continue
            full_box = (x + roi.x, y + roi.y, w, h)
            if not _inside_detection_band(full_box, calibration):
                continue
            confidence = _score_detection(area, full_box, roi, calibration_valid)
            if confidence < self._config.min_confidence:
                continue
            raw_boxes.append((*full_box, confidence))

        assignments = tracker.update([(x, y, w, h) for x, y, w, h, _ in raw_boxes])
        detections: list[WaveDetection] = []
        for raw_box in raw_boxes:
            x, y, w, h, confidence = raw_box
            track_id = assignments.get((x, y, w, h), -1)
            scale = pixels_per_foot_at_y(y + h // 2, calibration)
            height_ft = None
            if scale:
                height_ft = min(h / scale, self._config.max_wave_height_ft)
            detections.append(
                WaveDetection(
                    frame_index=frame_index,
                    time_sec=frame_index / fps,
                    track_id=track_id,
                    bbox=(x, y, w, h),
                    height_ft=height_ft,
                    confidence=round(confidence, 3),
                )
            )
        return detections


class _CentroidTracker:
    def __init__(self, max_distance_px: int) -> None:
        self._max_distance_px = max_distance_px
        self._next_id = 1
        self._centroids: dict[int, tuple[int, int]] = {}

    def update(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> dict[tuple[int, int, int, int], int]:
        assignments: dict[tuple[int, int, int, int], int] = {}
        unused_tracks = set(self._centroids)

        for box in boxes:
            centroid = _centroid(box)
            best_id: int | None = None
            best_distance = float("inf")
            for track_id in list(unused_tracks):
                distance = _distance(centroid, self._centroids[track_id])
                if distance < best_distance:
                    best_id = track_id
                    best_distance = distance
            if best_id is not None and best_distance <= self._max_distance_px:
                assignments[box] = best_id
                self._centroids[best_id] = centroid
                unused_tracks.remove(best_id)
            else:
                assignments[box] = self._next_id
                self._centroids[self._next_id] = centroid
                self._next_id += 1

        for track_id in unused_tracks:
            self._centroids.pop(track_id, None)

        return assignments


def _whitewater_mask(frame: np.ndarray, sensitivity: float) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    value_threshold = int(205 - (sensitivity * 45))
    sat_threshold = int(85 + (1 - sensitivity) * 30)
    hsv_mask = cv2.inRange(hsv, (0, 0, value_threshold), (180, sat_threshold, 255))
    light_mask = cv2.inRange(lab[:, :, 0], value_threshold, 255)
    mask = cv2.bitwise_and(hsv_mask, light_mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _inside_detection_band(
    box: tuple[int, int, int, int], calibration: Calibration
) -> bool:
    _, y, _, h = box
    horizon_y = int((calibration.horizon_line.y1 + calibration.horizon_line.y2) / 2)
    shore_y = int((calibration.shore_boundary.y1 + calibration.shore_boundary.y2) / 2)
    return y > horizon_y and y + h < shore_y


def _score_detection(
    area: float, box: tuple[int, int, int, int], roi: object, calibration_valid: bool
) -> float:
    _, _, w, h = box
    shape_score = min(1.0, max(0.1, (w / max(h, 1)) / 5))
    area_score = min(1.0, area / 1800)
    calibration_score = 1.0 if calibration_valid else 0.55
    return (0.45 * area_score) + (0.25 * shape_score) + (0.3 * calibration_score)


def _group_events(detections: list[WaveDetection]) -> list[WaveEvent]:
    by_track: dict[int, list[WaveDetection]] = defaultdict(list)
    for detection in detections:
        by_track[detection.track_id].append(detection)

    events: list[WaveEvent] = []
    for track_id, items in by_track.items():
        if track_id < 0 or len(items) < 2:
            continue
        items = sorted(items, key=lambda item: item.time_sec)
        heights = [item.height_ft for item in items if item.height_ft is not None]
        confidence = _mean([item.confidence for item in items]) or 0.0
        velocity = _track_velocity(items)
        events.append(
            WaveEvent(
                track_id=track_id,
                start_sec=items[0].time_sec,
                end_sec=items[-1].time_sec,
                max_height_ft=max(heights) if heights else None,
                avg_confidence=round(confidence, 3),
                velocity_ft_s=velocity,
            )
        )
    return sorted(events, key=lambda event: event.start_sec)


def _track_velocity(items: list[WaveDetection]) -> float | None:
    first = items[0]
    last = items[-1]
    elapsed = last.time_sec - first.time_sec
    if elapsed <= 0 or first.height_ft is None:
        return None
    pixel_distance = abs(_centroid(last.bbox)[0] - _centroid(first.bbox)[0])
    _, _, _, h = first.bbox
    pixels_per_foot = h / first.height_ft if first.height_ft > 0 else None
    if not pixels_per_foot:
        return None
    return round((pixel_distance / pixels_per_foot) / elapsed, 2)


def _set_intervals(events: list[WaveEvent]) -> list[float]:
    if len(events) < 2:
        return []
    return [
        round(events[index].start_sec - events[index - 1].start_sec, 1)
        for index in range(1, len(events))
    ]


def _encode_debug_frame(
    frame: np.ndarray, calibration: Calibration, detections: list[WaveDetection]
) -> str:
    cv2.line(
        frame,
        (calibration.horizon_line.x1, calibration.horizon_line.y1),
        (calibration.horizon_line.x2, calibration.horizon_line.y2),
        (255, 150, 0),
        2,
    )
    cv2.line(
        frame,
        (calibration.shore_boundary.x1, calibration.shore_boundary.y1),
        (calibration.shore_boundary.x2, calibration.shore_boundary.y2),
        (35, 120, 200),
        2,
    )
    roi = calibration.surf_zone_roi
    cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 180, 0), 2)
    for detection in detections:
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 255), 2)
        label = f"#{detection.track_id} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
        )
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _centroid(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 2)
