from scripts.create_splits import balanced_split, materialize_media


def make_rows(label: int, count: int) -> list[dict]:
    return [
        {"id": f"{label}_{index}", "label": label, "video": f"videos/{label}_{index}.mp4"}
        for index in range(count)
    ]


def test_abnormality_balancing_counts_and_determinism() -> None:
    source = make_rows(0, 382) + make_rows(1, 128)
    arguments = (source, (1.0, 1.0, 1.0), 42, 450, 0.5)
    first = balanced_split(*arguments)
    second = balanced_split(*arguments)

    assert first == second
    assert [len(first[name]) for name in ("train", "val", "test")] == [150, 150, 150]
    assert [
        sum(row["label"] != 0 for row in first[name])
        for name in ("train", "val", "test")
    ] == [75, 75, 75]
    assert sum(
        len(split) - len({row["video"] for row in split}) for split in first.values()
    ) == 97

    video_splits: dict[str, set[str]] = {
        name: {row["video"] for row in split} for name, split in first.items()
    }
    assert video_splits["train"].isdisjoint(video_splits["val"])
    assert video_splits["train"].isdisjoint(video_splits["test"])
    assert video_splits["val"].isdisjoint(video_splits["test"])


def test_materialize_media_hardlinks_a_runnable_split(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    video = source / "videos" / "example.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    row = {"id": "1", "label": 0, "video": "videos/example.mp4"}
    splits = {"train": [row], "val": [], "test": []}

    assert materialize_media(source, output, splits, "hardlink") == 1
    copied = output / "videos" / "example.mp4"
    assert copied.read_bytes() == b"video"
    assert video.stat().st_ino == copied.stat().st_ino
