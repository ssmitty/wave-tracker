"""Video loading and frame utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import cv2

from wavevision.models import VideoMetadata


class VideoLoadError(RuntimeError):
    """Raised when an uploaded video cannot be read."""


def save_upload(uploaded_file: object, suffix: str) -> tuple[str, str]:
    video_id = uuid4().hex[:12]
    temp_dir = Path(tempfile.gettempdir()) / "wavevision_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{video_id}{suffix}"
    with path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return video_id, str(path)


def read_video_metadata(path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoLoadError("Video could not be opened. Try another file.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoLoadError("Video metadata is incomplete. Try re-exporting it.")

    return VideoMetadata(
        path=path,
        duration_sec=frame_count / fps,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def extract_frame(path: str, frame_index: int) -> tuple[bool, object | None]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return ok, frame


def extract_middle_frame(path: str, metadata: VideoMetadata) -> object:
    ok, frame = extract_frame(path, metadata.frame_count // 2)
    if not ok or frame is None:
        raise VideoLoadError("Could not extract a calibration frame.")
    return frame
