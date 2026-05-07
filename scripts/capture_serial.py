from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import serial


CSV_HEADER = "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target"
CSV_COLUMNS = CSV_HEADER.split(",")


def _metadata_path_for(output_path: Path) -> Path:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Grove Vision bbox CSV rows from USB serial.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    metadata_path = _metadata_path_for(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    rows_written = 0
    skipped_row_count = 0
    comment_row_count = 0

    with serial.Serial(args.port, args.baud, timeout=1) as device, output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        saw_header = False
        writer = csv.writer(output, lineterminator="\n")
        print(f"Capturing bbox CSV from {args.port} to {output_path}. Press Ctrl+C to stop.")

        try:
            while True:
                raw_line = device.readline()
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line.startswith("#"):
                    comment_row_count += 1
                    print(line)
                    continue

                try:
                    row = parse_serial_csv_line(line)
                except csv.Error:
                    skipped_row_count += 1
                    continue

                if row is None:
                    continue

                if row == CSV_COLUMNS:
                    if not saw_header:
                        writer.writerow(CSV_COLUMNS)
                        output.flush()
                        saw_header = True
                    continue

                if saw_header:
                    try:
                        writer.writerow(validate_csv_row(row))
                    except ValueError:
                        skipped_row_count += 1
                        continue
                    output.flush()
                    rows_written += 1
                    print(",".join(row))
                else:
                    skipped_row_count += 1
        except KeyboardInterrupt:
            print("\nStopped capture.")

    end_time = datetime.now(timezone.utc)
    metadata = {
        "start_utc": start_time.isoformat(),
        "end_utc": end_time.isoformat(),
        "port": args.port,
        "baud": int(args.baud),
        "output_path": str(output_path),
        "rows_written": rows_written,
        "skipped_row_count": skipped_row_count,
        "comment_row_count": comment_row_count,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    print(f"Capture metadata written to {metadata_path}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
