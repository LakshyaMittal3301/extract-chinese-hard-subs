import json
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


def write_image(video_path_str: str, timestamp_sec: float) -> str:
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
    return str(frame_path)


def preprocess_for_subtitle_ocr(
    image_path_str: str,
    bottom_ratio: float = 0.30,  # keep bottom 30% of image
    center_width_ratio: float = 0.90,  # keep center 90% width
    contrast_alpha: float = 1.8,  # contrast multiplier (1.0 = no change)
    brightness_beta: int = 10,  # brightness shift
    use_threshold: bool = True,
) -> str:
    image_path = Path(image_path_str)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    # ---- 1) Crop bottom-center region ----
    y1 = int(h * (1 - bottom_ratio))
    y2 = h

    crop_w = int(w * center_width_ratio)
    x1 = (w - crop_w) // 2
    x2 = x1 + crop_w

    roi = image[y1:y2, x1:x2]

    # ---- 2) Convert to grayscale ----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ---- 3) Boost contrast / brightness ----
    enhanced = cv2.convertScaleAbs(gray, alpha=contrast_alpha, beta=brightness_beta)

    # ---- 4) Light denoise (optional but often helpful) ----
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # ---- 5) Threshold (helps subtitles stand out) ----
    if use_threshold:
        _, processed = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        processed = denoised

    if image_path.parent.name == "frames":
        video_dir = image_path.parent.parent
    else:
        video_dir = image_path.parent
    processed_dir = video_dir / "processed_frames"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / image_path.name
    cv2.imwrite(str(processed_path), processed)
    print(f"Saved processed frame: {processed_path}")
    return str(processed_path)


def extract_chinese_text(image_path_str: str) -> list[str]:
    ocr = PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    results = ocr.predict(input=image_path_str)
    texts: list[str] = []
    for res in results:
        data = res.json if hasattr(res, "json") else res
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict):
            texts.extend(data.get("rec_texts", []))
    return texts


def run_subtitle_ocr_for_timestamp(
    video_path_str: str, timestamp_sec: float
) -> list[str]:
    frame_path = write_image(video_path_str, timestamp_sec)
    processed_path = preprocess_for_subtitle_ocr(
        frame_path, bottom_ratio=0.15, center_width_ratio=0.5
    )
    texts = extract_chinese_text(processed_path)
    print(f"OCR text @ {timestamp_sec}s: {texts}")
    return texts


if __name__ == "__main__":
    run_subtitle_ocr_for_timestamp("./data/01/vid.mp4", 217.4)
