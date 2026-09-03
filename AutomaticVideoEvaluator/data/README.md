---
pretty_name: AVE Data
language:
  - zh
  - en
size_categories:
  - n<1K
tags:
  - video-evaluation
  - multimodal-evaluation
  - prompt-optimization
---

# AVE Data

AVE Data contains generated videos and human-written weakness annotations for
training and evaluating the Adaptive Video Evaluator (AVE).

## Tasks and composition

| Task | Train / validation / test rows | Unique generated videos | Conditioning media |
|---|---:|---:|---:|
| Abnormality | 154 / 148 / 148 | 353 | - |
| Perception | 60 / 60 / 63 | 183 | - |
| Prompt following | 39 / 39 / 39 | 117 | 165 images + 10 videos |

Across the three tasks, the canonical `all.json` files contain 653 unique
generated-video records. Abnormality has 450 split rows because 97 positive
metadata records are repeated within their original split for class balancing;
the video files themselves are stored once.

## Download

Install the dependency and download the complete dataset:

```bash
python -m pip install "huggingface_hub>=1.0,<2"
python download.py
```

To download one task:

```bash
python download.py --task perception
```

For an immutable download, specify a tag or commit hash:

```bash
python download.py --revision COMMIT_HASH
```

The equivalent Hugging Face CLI command is:

```bash
hf download JianhuiWei/AVE_data \
  --repo-type dataset \
  --local-dir .
```

## Integrity verification

The downloader verifies split JSON files, `all.json`, split plans, and every
media file against the bundled SHA-256 metadata. Verify an existing download
without network access using:

```bash
python download.py --verify-only
```

The AVE code repository also provides deeper schema checks:

```bash
python scripts/validate_dataset.py --data-dir data/abnormality --checksums
python scripts/validate_dataset.py --data-dir data/perception --checksums
python scripts/validate_dataset.py --data-dir data/prompt_following --checksums
```

## Layout

```text
AVE_data/
├── README.md
├── download.py
├── abnormality/
├── perception/
└── prompt_following/
```

Each task contains:

- `all.json`: one canonical record per generated video, without repeated IDs or
  generated-video paths;
- `train.json`, `val.json`, `test.json`: evaluator-training splits;
- `SPLIT_INFO.json`: split provenance and parameters;
- `SPLIT_PLAN.json`: exact mapping from canonical records to released split rows;
- `MANIFEST.json`: split and media sizes plus SHA-256 checksums;
- `videos/`, and conditioning-media directories where applicable.

## Record schema

```json
{
  "id": "unique record identifier",
  "prompt": "video-generation instruction",
  "reference_images": ["reference_images/example.png"],
  "reference_videos": ["reference_videos/example.mp4"],
  "video": "videos/example.mp4",
  "label": 1,
  "feedback": {
    "abnormality": "human-written weakness, or an empty string",
    "prompt_following": "",
    "consistency": ""
  }
}
```

`reference_videos` is optional for records without video conditioning. `label=0`
denotes an example with no annotated weakness; non-zero labels denote examples
with a weakness. The human-written text in `feedback.abnormality` is the
authoritative open-ended supervision.

## Split reproducibility

The accompanying AVE code supports:

- exact reconstruction of released splits from `all.json` and
  `SPLIT_PLAN.json`;
- deterministic label-stratified creation of new splits;
- deterministic creation and class balancing of new splits.

Use `scripts/create_splits.py` in the AVE code release. The checked-in split JSON
files are the canonical benchmark splits.

## Intended use and limitations

The dataset supports evaluator prompt optimization, video-evaluator benchmarking,
and false-positive/false-negative analysis. Human annotations may not enumerate
every plausible defect. Results also depend on temporal sampling and the
capabilities of the selected multimodal model. Reference videos are evaluated
through sampled frames in the released implementation.
