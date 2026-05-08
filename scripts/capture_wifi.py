from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from pedflow.serial_protocol import CSV_COLUMNS, metadata_path_for as _metadata_path_for
from pedflow.serial_protocol import parse_serial_csv_line, validate_csv_row


DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_UDP_PORT = 4210
DEFAULT_BUFFER_SIZE = 4096


def parse_udp_payload_lines(payload: bytes) -> list[str]:
    text = payload.decode("utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Grove Vision bbox CSV rows from the PedFlow sensor Wi-Fi UDP feed."
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--host", default=DEFAULT_BIND_HOST, help="Local bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_UDP_PORT, help="UDP port")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="UDP receive buffer")
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
    source_counts: dict[str, int] = {}

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver, output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver.bind((args.host, args.port))

        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        output.flush()

        print(
            f"Capturing bbox CSV from UDP {args.host}:{args.port} to {output_path}. "
            "Press Ctrl+C to stop."
        )

        try:
            while True:
                payload, address = receiver.recvfrom(args.buffer_size)
                source = f"{address[0]}:{address[1]}"
                source_counts[source] = source_counts.get(source, 0) + 1

                for line in parse_udp_payload_lines(payload):
                    if line.startswith("#"):
                        comment_row_count += 1
                        print(f"{source} {line}")
                        continue

                    try:
                        row = parse_serial_csv_line(line)
                    except csv.Error:
                        skipped_row_count += 1
                        continue

                    if row is None or row == CSV_COLUMNS:
                        continue

                    try:
                        writer.writerow(validate_csv_row(row))
                    except ValueError:
                        skipped_row_count += 1
                        continue

                    output.flush()
                    rows_written += 1
                    print(",".join(row))
        except KeyboardInterrupt:
            print("\nStopped capture.")

    end_time = datetime.now(timezone.utc)
    metadata = {
        "start_utc": start_time.isoformat(),
        "end_utc": end_time.isoformat(),
        "transport": "udp_wifi_ap",
        "bind_host": args.host,
        "port": int(args.port),
        "output_path": str(output_path),
        "rows_written": rows_written,
        "skipped_row_count": skipped_row_count,
        "comment_row_count": comment_row_count,
        "source_counts": source_counts,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    print(f"Capture metadata written to {metadata_path}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
