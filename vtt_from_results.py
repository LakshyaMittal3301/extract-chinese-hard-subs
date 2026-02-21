import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

FALLBACK_SAMPLE_STEP_SEC = 0.2


def _load_results(results_json_path: str) -> list[dict[str, Any]]:
    path = Path(results_json_path)
    if not path.exists():
        raise FileNotFoundError(f"results.json not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("results.json must contain a JSON array")

    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp_sec")
        text = item.get("text")
        confidence = item.get("confidence", 0.0)
        try:
            timestamp_f = float(timestamp)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "timestamp_sec": timestamp_f,
                "text": "" if text is None else str(text),
                "confidence": float(confidence),
            }
        )
    return rows


def _detect_sample_step_sec(
    results: list[dict[str, Any]], config_json_path: str | None
) -> float:
    if config_json_path:
        config_path = Path(config_json_path)
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                freq = float(config.get("frequency", 0))
                if freq > 0:
                    return 1.0 / freq
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

    timestamps = sorted(float(row["timestamp_sec"]) for row in results)
    diffs: list[float] = []
    for i in range(1, len(timestamps)):
        diff = timestamps[i] - timestamps[i - 1]
        if diff > 0:
            diffs.append(diff)
    if diffs:
        return float(median(diffs))
    return FALLBACK_SAMPLE_STEP_SEC


def _chunk_records_exact(
    results: list[dict[str, Any]],
    sample_step_sec: float,
    end_exclusive_ms: int,
) -> list[dict[str, Any]]:
    if sample_step_sec <= 0:
        sample_step_sec = FALLBACK_SAMPLE_STEP_SEC

    rows = sorted(results, key=lambda r: float(r["timestamp_sec"]))
    cues: list[dict[str, Any]] = []
    active_text: str | None = None
    active_start: float | None = None
    active_last_timestamp: float | None = None
    eps = end_exclusive_ms / 1000.0

    def close_active() -> None:
        nonlocal active_text, active_start, active_last_timestamp
        if active_text is None or active_start is None or active_last_timestamp is None:
            return
        raw_end = active_last_timestamp + sample_step_sec
        end = raw_end - eps
        if end <= active_start:
            end = active_start + 0.001
        cues.append(
            {
                "start_sec": active_start,
                "end_sec": end,
                "text": active_text,
            }
        )
        active_text = None
        active_start = None
        active_last_timestamp = None

    for row in rows:
        timestamp = float(row["timestamp_sec"])
        text = str(row.get("text", "")).strip()

        if text == "":
            close_active()
            continue

        if active_text is None:
            active_text = text
            active_start = timestamp
            active_last_timestamp = timestamp
            continue

        if text == active_text:
            active_last_timestamp = timestamp
            continue

        close_active()
        active_text = text
        active_start = timestamp
        active_last_timestamp = timestamp

    close_active()
    return cues


def _format_vtt_timestamp(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    total_ms = int(round(sec * 1000))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def _write_vtt(
    cues: list[dict[str, Any]],
    output_path: str,
    kind: str,
    language: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "WEBVTT",
        f"Kind: {kind}",
        f"Language: {language}",
        "",
    ]
    for cue in cues:
        start = _format_vtt_timestamp(float(cue["start_sec"]))
        end = _format_vtt_timestamp(float(cue["end_sec"]))
        text = str(cue["text"])
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_vtt_from_results(
    results_json_path: str,
    config_json_path: str | None = None,
    output_vtt_path: str | None = None,
    language: str = "zh-CN",
    kind: str = "captions",
    end_exclusive_ms: int = 1,
) -> str:
    results_path = Path(results_json_path)
    output_path = (
        Path(output_vtt_path) if output_vtt_path else results_path.with_name("results.vtt")
    )

    rows = _load_results(str(results_path))
    sample_step_sec = _detect_sample_step_sec(rows, config_json_path)
    cues = _chunk_records_exact(
        results=rows,
        sample_step_sec=sample_step_sec,
        end_exclusive_ms=end_exclusive_ms,
    )
    _write_vtt(cues=cues, output_path=str(output_path), kind=kind, language=language)
    return str(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a WebVTT file from OCR results.json"
    )
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument(
        "--config", default=None, help="Optional path to config.json for frequency"
    )
    parser.add_argument("--out", default=None, help="Optional output .vtt path")
    parser.add_argument("--language", default="zh-CN", help="VTT language header")
    parser.add_argument("--kind", default="captions", help="VTT kind header")
    parser.add_argument(
        "--end-exclusive-ms",
        type=int,
        default=1,
        help="Milliseconds to subtract from cue end time to keep end exclusive",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output = generate_vtt_from_results(
        results_json_path=args.results,
        config_json_path=args.config,
        output_vtt_path=args.out,
        language=args.language,
        kind=args.kind,
        end_exclusive_ms=args.end_exclusive_ms,
    )
    print(f"Saved VTT: {output}")


if __name__ == "__main__":
    main()
