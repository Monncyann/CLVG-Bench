from __future__ import annotations

import base64
import math
import mimetypes
from pathlib import Path


def _jpeg_data_url(frame, *, scale: float = 1.0, quality: int = 85) -> str:
    import cv2

    if scale < 1.0:
        height, width = frame.shape[:2]
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def image_data_url(path: str | Path, max_encoded_bytes: int | None = None) -> str:
    """Encode a reference image for an OpenAI-compatible multimodal request."""
    image_path = Path(path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported reference image type: {image_path}")
    original = f"data:{mime_type};base64," + base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")
    if max_encoded_bytes is None or len(original.encode("utf-8")) <= max_encoded_bytes:
        return original

    import cv2

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot decode reference image: {image_path}")
    scale = min(1.0, math.sqrt(max_encoded_bytes / len(original.encode("utf-8"))) * 0.9)
    for quality in (85, 75, 60, 45, 30, 20):
        value = _jpeg_data_url(frame, scale=scale, quality=quality)
        if len(value.encode("utf-8")) <= max_encoded_bytes:
            return value
        scale *= 0.8
    raise ValueError(
        f"Reference image cannot fit within its request budget: {image_path}"
    )


def _evenly_spaced(values: list[dict], count: int) -> list[dict]:
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def sample_video(
    path: str | Path,
    fps: float = 2.0,
    max_frames: int = 32,
    max_encoded_bytes: int | None = None,
) -> list[dict]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        capture.release()
        raise ValueError(f"Video reports no frames: {path}")
    duration = total_frames / source_fps
    effective_fps = fps
    if duration >= 10:
        effective_fps = min(4.0, effective_fps)
    if duration >= 15:
        effective_fps = min(2.0, effective_fps)
    sample_count = min(
        max_frames,
        total_frames,
        max(1, int(duration * effective_fps)),
    )
    frame_indices = [
        int(index * total_frames / sample_count) for index in range(sample_count)
    ]
    timestamp_interval = duration / sample_count
    target_positions = {
        frame_index: position for position, frame_index in enumerate(frame_indices)
    }
    sampled: list[tuple[float, object]] = []
    frames: list[dict] = []
    index = 0
    try:
        while len(sampled) + len(frames) < sample_count:
            ok, frame = capture.read()
            if not ok:
                break
            if index in target_positions:
                timestamp = target_positions[index] * timestamp_interval
                if max_encoded_bytes is None:
                    frames.append(
                        {
                            "timestamp": timestamp,
                            "data_url": _jpeg_data_url(frame),
                        }
                    )
                else:
                    sampled.append((timestamp, frame))
            index += 1
    finally:
        capture.release()
    if max_encoded_bytes is None:
        if not frames:
            raise ValueError(f"No frames decoded from video: {path}")
        return frames
    if not sampled:
        raise ValueError(f"No frames decoded from video: {path}")

    def encode(scale: float, quality: int) -> list[dict]:
        return [
            {
                "timestamp": timestamp,
                "data_url": _jpeg_data_url(frame, scale=scale, quality=quality),
            }
            for timestamp, frame in sampled
        ]

    frames = encode(1.0, 85)
    if max_encoded_bytes < 1:
        raise ValueError("max_encoded_bytes must be positive")

    def size(values: list[dict]) -> int:
        return sum(len(frame["data_url"].encode("utf-8")) for frame in values)

    scale = 1.0
    quality = 85
    for _ in range(8):
        current_size = size(frames)
        if current_size <= max_encoded_bytes:
            return frames
        scale *= max(0.35, min(0.9, math.sqrt(max_encoded_bytes / current_size) * 0.92))
        frames = encode(scale, quality)

    for quality in (75, 60, 45, 30, 20):
        frames = encode(scale, quality)
        if size(frames) <= max_encoded_bytes:
            return frames

    for count in range(len(frames) - 1, 0, -1):
        reduced = _evenly_spaced(frames, count)
        if size(reduced) <= max_encoded_bytes:
            return reduced
    raise ValueError(f"Video frames cannot fit within their request budget: {path}")
