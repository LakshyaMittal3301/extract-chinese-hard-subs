from pathlib import Path
from functools import lru_cache

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


def preprocess_for_subtitle_ocr(
    image: np.ndarray,
    source_frame_path: str,
    bottom_ratio: float = 0.30,  # keep bottom 30% of image
    center_width_ratio: float = 0.90,  # keep center 90% width
    white_v_min: int = 170,  # minimum brightness for white-ish text
    white_s_max: int = 60,  # maximum saturation for white-ish text
    min_component_area: int = 18,  # remove tiny noise blobs
    min_component_height: int = 10,  # remove very short blobs
    upscale: float = 2.0,
) -> tuple[np.ndarray, str]:
    image_path = Path(source_frame_path)
    h, w = image.shape[:2]

    # ---- 1) Crop bottom-center region ----
    y1 = int(h * (1 - bottom_ratio))
    y2 = h

    crop_w = int(w * center_width_ratio)
    x1 = (w - crop_w) // 2
    x2 = x1 + crop_w

    roi = image[y1:y2, x1:x2]

    # ---- 2) Keep only white-ish pixels in HSV ----
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array((0, 0, white_v_min), dtype=np.uint8)
    upper_white = np.array((179, white_s_max, 255), dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # ---- 3) Cleanup mask ----
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, open_kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, close_kernel)

    # ---- 4) Keep text-like connected components ----
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )
    filtered_mask = white_mask.copy()
    filtered_mask[:] = 0
    for label in range(1, num_labels):  # 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_component_area and comp_h >= min_component_height:
            filtered_mask[labels == label] = 255

    # ---- 5) Build OCR image from filtered mask ----
    isolated = cv2.bitwise_and(roi, roi, mask=filtered_mask)
    gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
    _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- 6) Upscale for better OCR on thin strokes ----
    if upscale > 1.0:
        processed = cv2.resize(
            processed,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_CUBIC,
        )

    if image_path.parent.name == "frames":
        video_dir = image_path.parent.parent
    else:
        video_dir = image_path.parent
    processed_dir = video_dir / "processed_frames"
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
    video_path_str: str, timestamp_sec: float, score_threshold: float = 0.5
) -> dict[str, float | str]:
    frame, frame_path = extract_frame(video_path_str, timestamp_sec)
    processed_image, _ = preprocess_for_subtitle_ocr(
        frame, frame_path, bottom_ratio=0.15, center_width_ratio=0.5
    )
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
    run_subtitle_ocr_for_timestamp("./data/01/vid.mp4", 112, score_threshold=0.5)
