import cv2
import numpy as np

from ave.video import image_data_url, sample_video


def test_reference_image_is_compressed_to_request_budget(tmp_path) -> None:
    image_path = tmp_path / "reference.png"
    image = np.random.default_rng(7).integers(0, 256, (900, 900, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    encoded = image_data_url(image_path, max_encoded_bytes=100_000)

    assert encoded.startswith("data:image/jpeg;base64,")
    assert len(encoded.encode("utf-8")) <= 100_000


def test_video_frames_are_compressed_to_request_budget(tmp_path) -> None:
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 240),
    )
    assert writer.isOpened()
    generator = np.random.default_rng(11)
    for _ in range(12):
        writer.write(generator.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    writer.release()

    frames = sample_video(
        video_path,
        fps=10.0,
        max_frames=12,
        max_encoded_bytes=80_000,
    )

    assert frames
    assert sum(len(frame["data_url"].encode("utf-8")) for frame in frames) <= 80_000


def test_frame_cap_samples_uniformly_across_full_video(tmp_path) -> None:
    video_path = tmp_path / "uniform.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 32),
    )
    assert writer.isOpened()
    for value in range(30):
        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    writer.release()

    frames = sample_video(video_path, fps=4.0, max_frames=3)

    assert [round(frame["timestamp"], 1) for frame in frames] == [0.0, 1.0, 2.0]
