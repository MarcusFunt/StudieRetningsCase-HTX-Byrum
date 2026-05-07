from __future__ import annotations

import argparse
import sys
from pathlib import Path

import serial


CSV_HEADER = "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Grove Vision bbox CSV rows from USB serial.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with serial.Serial(args.port, args.baud, timeout=1) as device, output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        saw_header = False
        print(f"Capturing bbox CSV from {args.port} to {output_path}. Press Ctrl+C to stop.")

        try:
            while True:
                raw_line = device.readline()
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line == CSV_HEADER:
                    if not saw_header:
                        output.write(line + "\n")
                        output.flush()
                        saw_header = True
                    continue

                if saw_header:
                    output.write(line + "\n")
                    output.flush()
                    print(line)
        except KeyboardInterrupt:
            print("\nStopped capture.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
