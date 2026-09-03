# Automatic Video Evaluator (AVE)

[简体中文](README_CN.md) | English

Self-contained implementation of the **Adaptive Video Evaluator (AVE)**. It
includes evaluator training and evaluation code, three task configurations,
initial prompts, reproducible dataset splitting, and a public dataset downloader.
It does not import anything from the parent `VideoFactory` repository.

## Supported experiments

| `algorithm` | Prompt selection | Loss |
|---|---|---|
| `textgrad` | best validation prompt | binary |
| `textgrad_semantic_loss` | best validation prompt | semantic matching |
| `gepa` | validation Pareto sampling | binary |
| `gepa_semantic_loss` | validation Pareto sampling | semantic matching |

A cold-start evaluator can also be run directly without prompt optimization.

TextGrad always continues from the prompt with the best validation score. GEPA
samples from the validation Pareto front using per-instance scores, then normally
admits a proposal only if it does not regress on the training mini-batch. Every
admitted prompt is evaluated on the complete validation split before entering
the prompt table. As in the original experiment runner, test is evaluated before
training, after every three complete validations, and once more on the final best
prompt.

## Installation

Python 3.10 or newer is required.

```bash
cd AutomaticVideoEvaluator
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

AVE uses a vision-capable OpenAI-compatible Chat Completions endpoint:

```bash
export AVE_API_KEY="your-key"
```

`AVE_BASE_URL` defaults to the Volcengine Ark endpoint used by the bundled Seed
configurations. Set it explicitly only when using another OpenAI-compatible
endpoint.

The bundled configurations use Seed 2.0 Lite as the video judge and Seed 2.0
Pro as the prompt optimizer and semantic matcher. Model names and token prices
can be changed in `configs/*.json`.

## Download the dataset

The public dataset is hosted at `JianhuiWei/AVE_data`.

```bash
python data/download.py
```

Download one task or verify an existing download:

```bash
python data/download.py --task perception
python data/download.py --verify-only
```

The downloader places data under this project's `data/` directory and verifies
the published SHA-256 metadata.

## Dataset overview

| Task | Train / val / test | Rows | Unique videos |
|---|---:|---:|---:|
| Abnormality | 154 / 148 / 148 | 450 | 353 |
| Perception | 60 / 60 / 63 | 183 | 183 |
| Prompt following | 39 / 39 / 39 | 117 | 117 |

Prompt following additionally contains 165 reference images and 10 reference
videos. You can also use `scripts/create_splits.py` to create a new split with
the desired ratios and random seed.

Each task contains:

- `all.json`: one canonical record per generated video, with no duplicate IDs
  or video paths;
- `train.json`, `val.json`, `test.json`: runnable evaluator splits;
- `SPLIT_PLAN.json`: exact mapping from `all.json` to the released splits;
- `MANIFEST.json`: media and split checksums.

`label=0` means no annotated weakness; non-zero means a weakness is present.
The human text in `feedback.abnormality` is the authoritative supervision. See
[`data/README.md`](data/README.md) for the full schema and dataset description.

Validate the downloaded data with:

```bash
python scripts/validate_dataset.py --data-dir data/abnormality
python scripts/validate_dataset.py --data-dir data/perception
python scripts/validate_dataset.py --data-dir data/prompt_following
```

Add `--checksums` for full media hashing.

## Run AVE

Train the default GEPA + Semantic Loss evaluator:

```bash
python scripts/train.py --config configs/abnormality.json
python scripts/train.py --config configs/perception.json
python scripts/train.py --config configs/prompt_following.json
```

Run any of the four baselines without editing a configuration:

```bash
python scripts/train.py --config configs/abnormality.json --algorithm textgrad
python scripts/train.py --config configs/abnormality.json --algorithm textgrad_semantic_loss
python scripts/train.py --config configs/abnormality.json --algorithm gepa
python scripts/train.py --config configs/abnormality.json --algorithm gepa_semantic_loss
```

When `--algorithm` is supplied, its output defaults to
`outputs/<task>_<algorithm>`. Use `--output-dir` to choose another directory.
An output directory is bound to one complete configuration, preventing results
from different baselines from being mixed.

Training follows the historical protocol by default:

1. evaluate the cold-start prompt on test and validation;
2. sample mini-batches with training seed 47;
3. skip prompt revision when a mini-batch is already perfect;
4. compare before/after on the same mini-batch for GEPA;
5. after ten consecutive regressions, force the proposal through full validation;
6. evaluate test after every three full validations;
7. select the best validation prompt and run the final test.

Dataset split generation remains independently seeded with 42. Evaluations use
up to 64 concurrent items, while each item's five votes remain sequential so an
irreversible majority can stop early. Video frames are sampled uniformly across
the full duration at the configured rate, with a default cap of 32 frames. Every
API request has a 240-second timeout, and optimizer visual input is capped at
45 MiB.

Completed items are saved incrementally under `checkpoints/`. If training is
interrupted, rerun the identical command to reuse completed evaluations and
prompt proposals. A changed configuration must use a new `output_dir`.

Evaluate a cold-start prompt without training:

```bash
python scripts/evaluate.py \
  --config configs/abnormality.json \
  --prompt prompts/abnormality.json \
  --split test \
  --output outputs/cold_start_test.json
```

Evaluate a learned prompt:

```bash
python scripts/evaluate.py \
  --config configs/abnormality.json \
  --algorithm gepa_semantic_loss \
  --prompt outputs/abnormality_gepa_semantic_loss/best_prompt.json \
  --split test \
  --output outputs/abnormality_gepa_semantic_loss/test_repeat.json
```

Semantic Loss asks a model to align predicted and human weakness sets, returns
TP/TN/FP/FN counts, and normalizes each instance before aggregation. Empty-set
cases are computed directly. The judge uses five-vote majority by default and
stops early when a majority is irreversible.

## Reproduce or create splits

Exactly reconstruct a released split from `all.json` and `SPLIT_PLAN.json`:

```bash
python scripts/create_splits.py \
  --data-dir data/abnormality \
  --output-dir /tmp/ave-abnormality-rebuilt \
  --strategy released
```

Create a new deterministic 1:1:1 split:

```bash
python scripts/create_splits.py \
  --data-dir data/prompt_following \
  --output-dir /tmp/ave-prompt-following-new \
  --strategy stratified \
  --seed 42 \
  --ratios 1 1 1
```

Use `--strategy balanced --total-size N --positive-ratio 0.5` to create a
balanced split. Media is hard-linked by default, so the new directory is
runnable without duplicating file contents. Use `--media-mode copy` across
filesystems or `--media-mode none` for JSON-only auditing.

## Outputs

Training writes the following files under the configured `output_dir`:

```text
prompt_000.json              initial prompt
prompt_NNN_proposal.json     every proposed prompt
prompt_NNN.json              prompts admitted after validation
candidate_table.json         validation scores, lineage, and Pareto front
best_prompt.json             best validation prompt
history.json                 optimization traces
initial_test_results.json    cold-start test predictions
test_step_NNN.json           periodic test predictions
test_results.json            final frozen-prompt test predictions
run_summary.json             metrics, rollouts, tokens, and estimated cost
run_metadata.json            configuration identity used to prevent mixed runs
training_state.json          latest replay/resume status
checkpoints/                 incremental per-stage item results
```

The default stopping limits are 70 prompt revisions, 4,500 dataset-item
rollouts, and an estimated USD 30 API budget. Initial and periodic test calls
count toward the historical budget and rollout totals. To reproduce the old
stopping rule, all tokens are charged to that budget at the configured judge
rate. `run_summary.json` additionally reports a role-specific estimate for the
judge, optimizer, and matcher, which is the better approximation of the actual
bill. A complete iteration is atomic, so the final iteration may slightly
exceed a configured limit.

## Project structure

```text
AutomaticVideoEvaluator/
├── ave/                 core evaluator and optimizer implementation
├── configs/             three runnable task configurations
├── prompts/             cold-start evaluator prompts
├── scripts/             train, evaluate, split, and validate entry points
├── data/                downloader and downloaded datasets
├── tests/               offline regression tests
├── pyproject.toml
└── requirements.txt
```

Run the tests with:

```bash
python -m pip install -e '.[dev]'
pytest -q
```
