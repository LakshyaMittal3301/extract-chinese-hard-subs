import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from paddleocr import PaddleOCR

DEFAULT_BOTTOM_RATIO = 0.15
DEFAULT_CENTER_WIDTH_RATIO = 0.5


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    frame_count: int
    duration_sec: float


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    debug_frames_dir: Path | None
    debug_processed_dir: Path | None
    results_run_dir: Path | None


@lru_cache(maxsize=1)
def get_ocr_model() -> PaddleOCR:
    return PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _read_video_meta(video_path: str) -> VideoMeta:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Could not read valid FPS from video: {video_path}")
    if frame_count <= 0:
        raise RuntimeError(f"Video has no decodable frames: {video_path}")

    duration_sec = frame_count / fps
    return VideoMeta(fps=fps, frame_count=frame_count, duration_sec=duration_sec)


def _normalize_range(meta: VideoMeta, start_sec: float, end_sec: float) -> tuple[float, float]:
    if start_sec < 0:
        raise ValueError("start_sec must be >= 0")
    if end_sec < start_sec:
        raise ValueError("end_sec must be >= start_sec")

    max_timestamp_sec = (meta.frame_count - 1) / meta.fps
    effective_end_sec = min(end_sec, max_timestamp_sec)
    return start_sec, effective_end_sec


def _effective_frequency(requested_frequency: float, fps: float) -> float:
    if requested_frequency <= 0:
        raise ValueError("frequency must be > 0")

    effective = min(requested_frequency, fps)
    if effective < requested_frequency:
        print(f"Sampling frequency clamped from {requested_frequency} to {effective} (video fps)")
    return effective


def _sample_timestamps(start_sec: float, end_sec: float, frequency: float) -> list[float]:
    if end_sec < start_sec:
        return []

    step = 1.0 / frequency
    timestamps: list[float] = []
    t = start_sec
    epsilon = step / 1000.0
    while t <= end_sec + epsilon:
        timestamps.append(round(t, 6))
        t += step
    return timestamps


def _sample_plan(
    timestamps: list[float], fps: float, frame_count: int
) -> list[tuple[float, int]]:
    max_index = frame_count - 1
    seen_indices: set[int] = set()
    plan: list[tuple[float, int]] = []

    for timestamp in timestamps:
        frame_idx = int(round(timestamp * fps))
        frame_idx = max(0, min(frame_idx, max_index))
        if frame_idx in seen_indices:
            continue
        seen_indices.add(frame_idx)
        plan.append((timestamp, frame_idx))
    return plan


def _format_float_for_id(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _open_run_paths(
    video_path: str,
    start_sec: float,
    end_sec: float,
    frequency: float,
    debug: bool,
    save_results_json: bool,
    debug_root: str | None,
    results_root: str | None,
) -> RunPaths:
    video_path_obj = Path(video_path)
    video_stem = video_path_obj.stem
    run_id = (
        f"{datetime.now():%Y%m%d_%H%M%S_%f}"
        f"_start_{_format_float_for_id(start_sec)}"
        f"_end_{_format_float_for_id(end_sec)}"
        f"_freq_{_format_float_for_id(frequency)}"
    )

    debug_frames_dir: Path | None = None
    debug_processed_dir: Path | None = None
    if debug:
        base_debug = (
            Path(debug_root)
            if debug_root
            else video_path_obj.parent / "processed_frames" / "debug_runs"
        )
        run_dir = base_debug / video_stem / run_id
        debug_frames_dir = run_dir / "frames"
        debug_processed_dir = run_dir / "processed"
        debug_frames_dir.mkdir(parents=True, exist_ok=True)
        debug_processed_dir.mkdir(parents=True, exist_ok=True)

    results_run_dir: Path | None = None
    if save_results_json:
        base_results = Path(results_root) if results_root else Path("results_runs")
        results_run_dir = base_results / video_stem / run_id
        results_run_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        debug_frames_dir=debug_frames_dir,
        debug_processed_dir=debug_processed_dir,
        results_run_dir=results_run_dir,
    )


def _build_config(
    video_path: str,
    meta: VideoMeta,
    start_sec: float,
    end_sec: float,
    effective_end_sec: float,
    requested_frequency: float,
    effective_frequency: float,
    score_threshold: float,
    debug: bool,
    save_results_json: bool,
    sampled_timestamps_count: int,
    sampled_unique_frames_count: int,
    returned_records_count: int,
    run_id: str,
) -> dict[str, float | int | bool | str]:
    return {
        "video_path": str(video_path),
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "duration_sec": round(meta.duration_sec, 6),
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "effective_end_sec": round(effective_end_sec, 6),
        "requested_frequency": requested_frequency,
        "frequency": effective_frequency,
        "score_threshold": score_threshold,
        "debug": debug,
        "save_results_json": save_results_json,
        "sampled_timestamps": sampled_timestamps_count,
        "sampled_unique_frames": sampled_unique_frames_count,
        "returned_records": returned_records_count,
        "run_id": run_id,
    }


def _write_results_and_config(
    results: list[dict[str, float | str]],
    config: dict[str, float | int | bool | str],
    results_dir: Path,
) -> None:
    (results_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved results: {results_dir / 'results.json'}")


def _crop_subtitle_roi(
    image: np.ndarray, bottom_ratio: float, center_width_ratio: float
) -> np.ndarray:
    h, w = image.shape[:2]
    y1 = int(h * (1 - bottom_ratio))
    y2 = h
    crop_w = int(w * center_width_ratio)
    x1 = (w - crop_w) // 2
    x2 = x1 + crop_w
    return image[y1:y2, x1:x2]


def _iter_sampled_frames(
    cap: cv2.VideoCapture,
    sample_plan: list[tuple[float, int]],
    debug_frames_dir: Path | None = None,
) -> Iterator[tuple[float, int, np.ndarray, str | None]]:
    if not sample_plan:
        return

    first_frame_idx = sample_plan[0][1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
    cursor_idx = first_frame_idx

    for timestamp_sec, frame_idx in sample_plan:
        while cursor_idx < frame_idx:
            if not cap.grab():
                raise RuntimeError(
                    f"Failed while seeking to sampled frame {frame_idx} at {timestamp_sec:.6f}s"
                )
            cursor_idx += 1

        if not cap.grab():
            raise RuntimeError(
                f"Failed to read sampled frame {frame_idx} at {timestamp_sec:.6f}s"
            )
        cursor_idx += 1

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to retrieve sampled frame {frame_idx} at {timestamp_sec:.6f}s"
            )

        frame_save_path: str | None = None
        if debug_frames_dir is not None:
            artifact_name = (
                f"{int(round(timestamp_sec * 1000)):010d}_{frame_idx:08d}.png"
            )
            frame_save_path = str(debug_frames_dir / artifact_name)
            cv2.imwrite(frame_save_path, frame)

        yield timestamp_sec, frame_idx, frame, frame_save_path


def _ocr_record_from_frame(
    frame: np.ndarray,
    score_threshold: float,
    bottom_ratio: float,
    center_width_ratio: float,
    processed_save_path: str | None = None,
) -> dict[str, float | str]:
    processed = _crop_subtitle_roi(
        frame, bottom_ratio=bottom_ratio, center_width_ratio=center_width_ratio
    )
    if processed_save_path:
        cv2.imwrite(processed_save_path, processed)

    if processed.ndim == 2:
        ocr_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    elif processed.ndim == 3 and processed.shape[2] == 1:
        ocr_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        ocr_input = processed

    results = get_ocr_model().predict(input=ocr_input)
    kept_texts: list[str] = []
    kept_scores: list[float] = []

    for res in results:
        payload = res if isinstance(res, dict) else {}
        rec_texts = payload.get("rec_texts", [])
        rec_scores = payload.get("rec_scores", [])

        if not rec_scores:
            kept_texts.extend(rec_texts)
            kept_scores.extend([1.0] * len(rec_texts))
            continue

        for text, score in zip(rec_texts, rec_scores):
            if score is not None and float(score) >= score_threshold:
                kept_texts.append(text)
                kept_scores.append(float(score))

    del processed
    merged_text = "".join(kept_texts)
    confidence = sum(kept_scores) / len(kept_scores) if kept_scores else 0.0
    return {"text": merged_text, "confidence": float(confidence)}


def run_subtitle_ocr_for_range(
    video_path_str: str,
    start_sec: float,
    end_sec: float,
    frequency: float,
    score_threshold: float = 0.5,
    debug: bool = False,
    save_results_json: bool = False,
    debug_root: str | None = None,
    results_root: str | None = None,
    bottom_ratio: float = DEFAULT_BOTTOM_RATIO,
    center_width_ratio: float = DEFAULT_CENTER_WIDTH_RATIO,
) -> list[dict[str, float | str]]:
    meta = _read_video_meta(video_path_str)
    start_sec, effective_end_sec = _normalize_range(meta, start_sec, end_sec)
    effective_frequency = _effective_frequency(frequency, meta.fps)

    sample_timestamps = _sample_timestamps(
        start_sec=start_sec,
        end_sec=effective_end_sec,
        frequency=effective_frequency,
    )
    sample_plan = _sample_plan(sample_timestamps, meta.fps, meta.frame_count)

    paths = _open_run_paths(
        video_path=video_path_str,
        start_sec=start_sec,
        end_sec=effective_end_sec,
        frequency=effective_frequency,
        debug=debug,
        save_results_json=save_results_json,
        debug_root=debug_root,
        results_root=results_root,
    )

    if effective_end_sec < start_sec or not sample_plan:
        empty_results: list[dict[str, float | str]] = []
        config = _build_config(
            video_path=video_path_str,
            meta=meta,
            start_sec=start_sec,
            end_sec=end_sec,
            effective_end_sec=effective_end_sec,
            requested_frequency=frequency,
            effective_frequency=effective_frequency,
            score_threshold=score_threshold,
            debug=debug,
            save_results_json=save_results_json,
            sampled_timestamps_count=len(sample_timestamps),
            sampled_unique_frames_count=0,
            returned_records_count=0,
            run_id=paths.run_id,
        )
        if paths.results_run_dir is not None:
            _write_results_and_config(empty_results, config, paths.results_run_dir)
        return empty_results

    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path_str}")

    records: list[dict[str, float | str]] = []
    try:
        for timestamp_sec, frame_idx, frame, frame_save_path in _iter_sampled_frames(
            cap=cap,
            sample_plan=sample_plan,
            debug_frames_dir=paths.debug_frames_dir,
        ):
            processed_save_path: str | None = None
            if paths.debug_processed_dir is not None and frame_save_path is not None:
                processed_save_path = str(
                    paths.debug_processed_dir / Path(frame_save_path).name
                )

            ocr_record = _ocr_record_from_frame(
                frame=frame,
                score_threshold=score_threshold,
                bottom_ratio=bottom_ratio,
                center_width_ratio=center_width_ratio,
                processed_save_path=processed_save_path,
            )
            records.append(
                {
                    "timestamp_sec": float(timestamp_sec),
                    "text": str(ocr_record["text"]),
                    "confidence": float(ocr_record["confidence"]),
                }
            )
            del frame
    finally:
        cap.release()

    config = _build_config(
        video_path=video_path_str,
        meta=meta,
        start_sec=start_sec,
        end_sec=end_sec,
        effective_end_sec=effective_end_sec,
        requested_frequency=frequency,
        effective_frequency=effective_frequency,
        score_threshold=score_threshold,
        debug=debug,
        save_results_json=save_results_json,
        sampled_timestamps_count=len(sample_timestamps),
        sampled_unique_frames_count=len(sample_plan),
        returned_records_count=len(records),
        run_id=paths.run_id,
    )
    if paths.results_run_dir is not None:
        _write_results_and_config(records, config, paths.results_run_dir)

    return records


def run_subtitle_ocr_for_timestamp(
    video_path_str: str,
    timestamp_sec: float,
    score_threshold: float = 0.5,
) -> dict[str, float | str]:
    rows = run_subtitle_ocr_for_range(
        video_path_str=video_path_str,
        start_sec=timestamp_sec,
        end_sec=timestamp_sec,
        frequency=1.0,
        score_threshold=score_threshold,
        debug=False,
        save_results_json=False,
    )
    if rows:
        return rows[0]
    return {"timestamp_sec": float(timestamp_sec), "text": "", "confidence": 0.0}


if __name__ == "__main__":
    sample_results = run_subtitle_ocr_for_range(
        video_path_str="./data/01/vid.mp4",
        start_sec=90,
        end_sec=200,
        frequency=5,
        score_threshold=0.5,
        debug=False,
        save_results_json=True,
    )
    print(sample_results)
