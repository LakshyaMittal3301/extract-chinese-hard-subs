from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


@lru_cache(maxsize=1)
def get_ocr_model() -> PaddleOCR:
    return PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def extract_frame(video_path_str: str, timestamp_sec: float) -> tuple[np.ndarray, str]:
    timestamp_msec = int(round(timestamp_sec * 1000))
    video_path = Path(video_path_str)
    frames_dir = video_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"{timestamp_msec}.png"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_msec)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(
            f"Could not read frame at {timestamp_sec}s from {video_path}"
        )

    cv2.imwrite(str(frame_path), frame)
    print(f"Saved frame: {frame_path}")
    return frame, str(frame_path)


def _get_video_dir_from_frame_path(source_frame_path: str) -> Path:
    image_path = Path(source_frame_path)
    if image_path.parent.name == "frames":
        return image_path.parent.parent
    return image_path.parent


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
    source_frame_path: str,
    bottom_ratio: float = 0.15,
    center_width_ratio: float = 0.5,
) -> tuple[np.ndarray, str]:
    image_path = Path(source_frame_path)
    processed = _crop_subtitle_roi(
        image=image, bottom_ratio=bottom_ratio, center_width_ratio=center_width_ratio
    )

    processed_dir = _get_video_dir_from_frame_path(source_frame_path) / "processed_frames"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / image_path.name

    cv2.imwrite(str(processed_path), processed)
    print(f"Saved processed frame: {processed_path}")
    return processed, str(processed_path)


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


def run_subtitle_ocr_for_timestamp(
    video_path_str: str,
    timestamp_sec: float,
    score_threshold: float = 0.5,
) -> dict[str, float | str]:
    frame, frame_path = extract_frame(video_path_str, timestamp_sec)
    processed_image, _ = preprocess_for_subtitle_ocr(frame, frame_path)
    merged_text, confidence = extract_chinese_text(
        processed_image, score_threshold=score_threshold
    )

    result: dict[str, float | str] = {
        "timestamp_sec": float(timestamp_sec),
        "text": merged_text,
        "confidence": confidence,
    }
    print(f"OCR result @ {timestamp_sec}s: {result}")

    del frame
    del processed_image
    return result


if __name__ == "__main__":
    run_subtitle_ocr_for_timestamp("./data/01/vid.mp4", 111, score_threshold=0.5)
