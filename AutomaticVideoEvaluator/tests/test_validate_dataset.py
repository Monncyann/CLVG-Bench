import json
import subprocess
import sys
from pathlib import Path


def row(identifier: str, video: str) -> dict:
    return {
        "id": identifier,
        "prompt": "",
        "reference_images": [],
        "video": video,
        "label": 0,
        "feedback": {"abnormality": "", "prompt_following": "", "consistency": ""},
    }


def test_validator_rejects_generated_video_leakage(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "shared.mp4").write_bytes(b"shared")
    (videos / "test.mp4").write_bytes(b"test")
    (tmp_path / "train.json").write_text(
        json.dumps([row("train", "videos/shared.mp4")]), encoding="utf-8"
    )
    (tmp_path / "val.json").write_text(
        json.dumps([row("val", "videos/shared.mp4")]), encoding="utf-8"
    )
    (tmp_path / "test.json").write_text(
        json.dumps([row("test", "videos/test.mp4")]), encoding="utf-8"
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Generated-video leakage" in result.stderr
