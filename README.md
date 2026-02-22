# Extracting Chinese Hard Subs

Tiny project for extracting Chinese hard subtitles from a video and turning them into a `.vtt` subtitle file.

## What the files do

- `main.py`: extracts hard subtitles from video frames and saves structured OCR results as JSON.
- `vtt_from_results.py`: reads the JSON results and creates a WebVTT (`.vtt`) file.

## Install

Requires Python `3.13+`.

Using `uv` (recommended):

```bash
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

## Use

### 1) Extract subtitles to JSON

Edit the example values at the bottom of `main.py` (`video_path_str`, `start_sec`, `end_sec`, `frequency`), then run:

```bash
uv run python main.py
```

This saves `results.json` and `config.json` in `results_runs/...`.

### 2) Convert results to VTT

```bash
uv run python vtt_from_results.py \
  --results results_runs/<video>/<run_id>/results.json \
  --config results_runs/<video>/<run_id>/config.json
```

This creates `results.vtt` next to the `results.json` file (unless `--out` is provided).
