from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import serial

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pedflow.serial_protocol import (
    CSV_COLUMNS,
    UNKNOWN_FIRMWARE_VERSION,
    firmware_version_from_status_line,
    parse_serial_csv_line,
    validate_csv_row,
)
from pedflow.serial_protocol import metadata_path_for as _metadata_path_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Grove Vision bbox CSV rows from USB serial.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional calibration JSON path to record in capture metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    metadata_path = _metadata_path_for(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(UTC)
    rows_written = 0
    skipped_row_count = 0
    comment_row_count = 0
    firmware_version = UNKNOWN_FIRMWARE_VERSION

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
                    detected_version = firmware_version_from_status_line(line)
                    if detected_version is not None:
                        firmware_version = detected_version
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

    end_time = datetime.now(UTC)
    metadata = {
        "schema_version": 1,
        "session_type": "detection_capture",
        "start_utc": start_time.isoformat(),
        "end_utc": end_time.isoformat(),
        "firmware_version": firmware_version,
        "calibration_json_path": str(args.calibration) if args.calibration else None,
        "settings": {
            "transport": "usb_serial",
            "port": args.port,
            "baud": int(args.baud),
        },
        "output_path": str(output_path),
        "output_files": {
            "detections_csv": str(output_path),
            "metadata_json": str(metadata_path),
        },
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
