from __future__ import annotations

import csv
import math
from pathlib import Path

CSV_HEADER = "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target"
CSV_COLUMNS = CSV_HEADER.split(",")


def metadata_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".metadata.json")


def _parse_float(value: str, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{column} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{column} is not finite")
    return parsed


def _parse_int(value: str, column: str) -> int:
    parsed = _parse_float(value, column)
    if parsed < 0:
        raise ValueError(f"{column} must be non-negative")
    if not parsed.is_integer():
        raise ValueError(f"{column} must be an integer")
    return int(parsed)


def validate_csv_row(row: list[str]) -> list[str]:
    if len(row) != len(CSV_COLUMNS):
        raise ValueError("wrong column count")

    timestamp_ms = _parse_float(row[0], "timestamp_ms")
    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")

    _parse_int(row[1], "frame_id")
    _parse_int(row[2], "detection_id")
    _parse_float(row[3], "bbox_x")
    _parse_float(row[4], "bbox_y")
    bbox_w = _parse_float(row[5], "bbox_w")
    bbox_h = _parse_float(row[6], "bbox_h")
    confidence = _parse_float(row[7], "confidence")
    _parse_int(row[8], "target")

    if bbox_w < 0 or bbox_h < 0:
        raise ValueError("bbox_w and bbox_h must be non-negative")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    return row


def parse_serial_csv_line(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    return next(csv.reader([stripped]))
