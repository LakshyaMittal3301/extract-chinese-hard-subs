import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

DEFAULT_BOTTOM_RATIO = 0.15
DEFAULT_CENTER_WIDTH_RATIO = 0.5


@lru_cache(maxsize=1)
def get_ocr_model() -> PaddleOCR:
    return PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _open_video_metadata(video_path_str: str) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path_str}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Could not read valid FPS from video: {video_path_str}")
    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    return fps, frame_count, duration_sec


def _generate_sample_timestamps(
    start_sec: float, end_sec: float, frequency: float
) -> list[float]:
    step = 1.0 / frequency
    timestamps: list[float] = []
    t = start_sec
    epsilon = step / 1000.0
    while t <= end_sec + epsilon:
        timestamps.append(round(t, 6))
        t += step

    if not timestamps:
        timestamps.append(round(start_sec, 6))
    return timestamps


def _build_sample_plan(
    timestamps: list[float], fps: float, frame_count: int
) -> list[tuple[float, int]]:
    plan: list[tuple[float, int]] = []
    seen_indices: set[int] = set()
    max_index = max(frame_count - 1, 0)
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


def _build_run_paths(
    video_path_str: str,
    start_sec: float,
    end_sec: float,
    frequency: float,
    debug: bool,
    save_results_json: bool,
    debug_root: str | None,
    results_root: str | None,
) -> dict[str, Path | str | None]:
    video_path = Path(video_path_str)
    video_stem = video_path.stem
    run_id = (
        f"{datetime.now():%Y%m%d_%H%M%S_%f}"
        f"_start_{_format_float_for_id(start_sec)}"
        f"_end_{_format_float_for_id(end_sec)}"
        f"_freq_{_format_float_for_id(frequency)}"
    )

    debug_run_dir: Path | None = None
    debug_frames_dir: Path | None = None
    debug_processed_dir: Path | None = None
    if debug:
        base = (
            Path(debug_root)
            if debug_root
            else video_path.parent / "processed_frames" / "debug_runs"
        )
        debug_run_dir = base / video_stem / run_id
        debug_frames_dir = debug_run_dir / "frames"
        debug_processed_dir = debug_run_dir / "processed"
        debug_frames_dir.mkdir(parents=True, exist_ok=True)
        debug_processed_dir.mkdir(parents=True, exist_ok=True)

    results_run_dir: Path | None = None
    if save_results_json:
        base = Path(results_root) if results_root else Path("results_runs")
        results_run_dir = base / video_stem / run_id
        results_run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_id": run_id,
        "debug_run_dir": debug_run_dir,
        "debug_frames_dir": debug_frames_dir,
        "debug_processed_dir": debug_processed_dir,
        "results_run_dir": results_run_dir,
    }


def _write_json_outputs(
    results: list[dict[str, float | str]],
    config: dict[str, float | int | bool | str],
    results_run_dir: Path,
) -> None:
    results_path = results_run_dir / "results.json"
    config_path = results_run_dir / "config.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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


def preprocess_for_subtitle_ocr(
    image: np.ndarray,
    bottom_ratio: float = DEFAULT_BOTTOM_RATIO,
    center_width_ratio: float = DEFAULT_CENTER_WIDTH_RATIO,
    save_path: str | None = None,
) -> np.ndarray:
    processed = _crop_subtitle_roi(
        image=image, bottom_ratio=bottom_ratio, center_width_ratio=center_width_ratio
    )
    if save_path:
        cv2.imwrite(save_path, processed)
    return processed


def extract_chinese_text(
    processed_image: np.ndarray, score_threshold: float = 0.5
) -> tuple[str, float]:
    if processed_image.ndim == 2:
        ocr_input = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    elif processed_image.ndim == 3 and processed_image.shape[2] == 1:
        ocr_input = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    else:
        ocr_input = processed_image

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

    merged_text = "".join(kept_texts)
    confidence = sum(kept_scores) / len(kept_scores) if kept_scores else 0.0
    return merged_text, confidence


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
) -> list[dict[str, float | str]]:
    if frequency <= 0:
        raise ValueError("frequency must be > 0")
    if start_sec < 0:
        raise ValueError("start_sec must be >= 0")
    if end_sec < start_sec:
        raise ValueError("end_sec must be >= start_sec")

    fps, frame_count, duration_sec = _open_video_metadata(video_path_str)
    if frame_count <= 0:
        raise RuntimeError(f"Video has no decodable frames: {video_path_str}")

    max_timestamp_sec = (frame_count - 1) / fps if frame_count > 0 else 0.0
    effective_end_sec = min(end_sec, max_timestamp_sec)

    if effective_end_sec < start_sec:
        sample_timestamps: list[float] = []
    else:
        sample_timestamps = _generate_sample_timestamps(
            start_sec=start_sec, end_sec=effective_end_sec, frequency=frequency
        )
    sample_plan = _build_sample_plan(
        timestamps=sample_timestamps, fps=fps, frame_count=frame_count
    )

    paths = _build_run_paths(
        video_path_str=video_path_str,
        start_sec=start_sec,
        end_sec=effective_end_sec,
        frequency=frequency,
        debug=debug,
        save_results_json=save_results_json,
        debug_root=debug_root,
        results_root=results_root,
    )
    debug_frames_dir = paths["debug_frames_dir"]
    debug_processed_dir = paths["debug_processed_dir"]
    results_run_dir = paths["results_run_dir"]

    results: list[dict[str, float | str]] = []
    if not sample_plan:
        config = {
            "video_path": str(video_path_str),
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 6),
            "start_sec": round(start_sec, 6),
            "end_sec": round(end_sec, 6),
            "effective_end_sec": round(effective_end_sec, 6),
            "frequency": frequency,
            "score_threshold": score_threshold,
            "debug": debug,
            "save_results_json": save_results_json,
            "sampled_timestamps": len(sample_timestamps),
            "sampled_unique_frames": 0,
            "returned_records": 0,
            "run_id": str(paths["run_id"]),
        }
        if isinstance(results_run_dir, Path):
            _write_json_outputs(
                results=results, config=config, results_run_dir=results_run_dir
            )
        return results

    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path_str}")

    first_frame_idx = sample_plan[0][1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
    cursor_idx = first_frame_idx
    stream_exhausted = False

    for timestamp_sec, frame_idx in sample_plan:
        if stream_exhausted:
            results.append(
                {
                    "timestamp_sec": float(timestamp_sec),
                    "text": "",
                    "confidence": 0.0,
                }
            )
            continue

        while cursor_idx < frame_idx:
            if not cap.grab():
                stream_exhausted = True
                break
            cursor_idx += 1

        if stream_exhausted:
            results.append(
                {
                    "timestamp_sec": float(timestamp_sec),
                    "text": "",
                    "confidence": 0.0,
                }
            )
            continue

        if not cap.grab():
            stream_exhausted = True
            results.append(
                {
                    "timestamp_sec": float(timestamp_sec),
                    "text": "",
                    "confidence": 0.0,
                }
            )
            continue
        cursor_idx += 1

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            results.append(
                {
                    "timestamp_sec": float(timestamp_sec),
                    "text": "",
                    "confidence": 0.0,
                }
            )
            continue

        artifact_name = f"{int(round(timestamp_sec * 1000)):010d}_{frame_idx:08d}.png"
        frame_save_path = (
            str(debug_frames_dir / artifact_name)
            if isinstance(debug_frames_dir, Path)
            else None
        )
        if frame_save_path:
            cv2.imwrite(frame_save_path, frame)

        processed_save_path = (
            str(debug_processed_dir / artifact_name)
            if isinstance(debug_processed_dir, Path)
            else None
        )
        processed_image = preprocess_for_subtitle_ocr(
            frame,
            bottom_ratio=DEFAULT_BOTTOM_RATIO,
            center_width_ratio=DEFAULT_CENTER_WIDTH_RATIO,
            save_path=processed_save_path,
        )
        text, confidence = extract_chinese_text(
            processed_image=processed_image, score_threshold=score_threshold
        )
        results.append(
            {
                "timestamp_sec": float(timestamp_sec),
                "text": text,
                "confidence": float(confidence),
            }
        )
        del frame
        del processed_image

    cap.release()

    config = {
        "video_path": str(video_path_str),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": round(duration_sec, 6),
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "effective_end_sec": round(effective_end_sec, 6),
        "frequency": frequency,
        "score_threshold": score_threshold,
        "debug": debug,
        "save_results_json": save_results_json,
        "sampled_timestamps": len(sample_timestamps),
        "sampled_unique_frames": len(sample_plan),
        "returned_records": len(results),
        "run_id": str(paths["run_id"]),
    }
    if isinstance(results_run_dir, Path):
        _write_json_outputs(
            results=results, config=config, results_run_dir=results_run_dir
        )

    return results


def run_subtitle_ocr_for_timestamp(
    video_path_str: str,
    timestamp_sec: float,
    score_threshold: float = 0.5,
) -> dict[str, float | str]:
    results = run_subtitle_ocr_for_range(
        video_path_str=video_path_str,
        start_sec=timestamp_sec,
        end_sec=timestamp_sec,
        frequency=1.0,
        score_threshold=score_threshold,
        debug=False,
        save_results_json=False,
    )
    if results:
        return results[0]
    return {"timestamp_sec": float(timestamp_sec), "text": "", "confidence": 0.0}


if __name__ == "__main__":
    sample_results = run_subtitle_ocr_for_range(
        video_path_str="./data/01/vid.mp4",
        start_sec=613,
        end_sec=620,
        frequency=5,
        score_threshold=0.5,
        debug=False,
        save_results_json=True,
    )
    print(sample_results)
