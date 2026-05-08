from __future__ import annotations

import csv
import json
import socket
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .serial_protocol import CSV_COLUMNS, metadata_path_for, parse_serial_csv_line, validate_csv_row

DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_UDP_PORT = 4210
DEFAULT_BUFFER_SIZE = 4096


@dataclass(frozen=True)
class CaptureSnapshot:
    running: bool
    output_path: Path
    metadata_path: Path
    rows_written: int
    skipped_row_count: int
    comment_row_count: int
    source_counts: dict[str, int]
    log_lines: list[str]
    error: str | None
    started_utc: str | None
    ended_utc: str | None


def parse_udp_payload_lines(payload: bytes) -> list[str]:
    text = payload.decode("utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


class _CsvCaptureWorker:
    def __init__(self, output_path: str | Path, max_log_lines: int = 300) -> None:
        self.output_path = Path(output_path)
        self.metadata_path = metadata_path_for(self.output_path)
        self._max_log_lines = int(max_log_lines)
        self._log_lines: deque[str] = deque(maxlen=self._max_log_lines)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._rows_written = 0
        self._skipped_row_count = 0
        self._comment_row_count = 0
        self._source_counts: dict[str, int] = {}
        self._error: str | None = None
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._running = True
            self._rows_written = 0
            self._skipped_row_count = 0
            self._comment_row_count = 0
            self._source_counts = {}
            self._error = None
            self._started_at = datetime.now(UTC)
            self._ended_at = None
            self._log_lines.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_guarded,
            name=type(self).__name__,
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

    def snapshot(self) -> CaptureSnapshot:
        with self._lock:
            return CaptureSnapshot(
                running=self._running,
                output_path=self.output_path,
                metadata_path=self.metadata_path,
                rows_written=self._rows_written,
                skipped_row_count=self._skipped_row_count,
                comment_row_count=self._comment_row_count,
                source_counts=dict(self._source_counts),
                log_lines=list(self._log_lines),
                error=self._error,
                started_utc=self._started_at.isoformat() if self._started_at else None,
                ended_utc=self._ended_at.isoformat() if self._ended_at else None,
            )

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            self._log(f"error: {exc}")
        finally:
            with self._lock:
                if self._ended_at is None:
                    self._ended_at = datetime.now(UTC)
                self._running = False
            try:
                self._write_metadata()
                self._log(f"metadata written: {self.metadata_path}")
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                self._log(f"metadata error: {exc}")

    def _run(self) -> None:
        raise NotImplementedError

    def _metadata(self) -> dict:
        snapshot = self.snapshot()
        return {
            "start_utc": snapshot.started_utc,
            "end_utc": snapshot.ended_utc,
            "output_path": str(snapshot.output_path),
            "rows_written": snapshot.rows_written,
            "skipped_row_count": snapshot.skipped_row_count,
            "comment_row_count": snapshot.comment_row_count,
        }

    def _write_metadata(self) -> None:
        metadata = self._metadata()
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
            metadata_file.write("\n")

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._log_lines.append(f"[{timestamp}] {message}")

    def _increment_rows(self) -> None:
        with self._lock:
            self._rows_written += 1

    def _increment_skipped(self) -> None:
        with self._lock:
            self._skipped_row_count += 1

    def _increment_comment(self) -> None:
        with self._lock:
            self._comment_row_count += 1

    def _increment_source(self, source: str) -> None:
        with self._lock:
            self._source_counts[source] = self._source_counts.get(source, 0) + 1


class SerialCsvCaptureWorker(_CsvCaptureWorker):
    def __init__(
        self,
        port: str,
        output_path: str | Path,
        baud: int = 115200,
        max_log_lines: int = 300,
    ) -> None:
        super().__init__(output_path, max_log_lines=max_log_lines)
        self.port = str(port).strip()
        self.baud = int(baud)

    def _run(self) -> None:
        import serial

        self._log(f"opening USB serial {self.port} at {self.baud} baud")
        with serial.Serial(self.port, self.baud, timeout=0.5) as device, self.output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output:
            saw_header = False
            writer = csv.writer(output, lineterminator="\n")
            self._log(f"capturing USB CSV to {self.output_path}")
            while not self._stop_event.is_set():
                raw_line = device.readline()
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("#"):
                    self._increment_comment()
                    self._log(line)
                    continue

                try:
                    row = parse_serial_csv_line(line)
                except csv.Error:
                    self._increment_skipped()
                    continue

                if row is None:
                    continue
                if row == CSV_COLUMNS:
                    if not saw_header:
                        writer.writerow(CSV_COLUMNS)
                        output.flush()
                        saw_header = True
                        self._log("CSV header received")
                    continue

                if not saw_header:
                    self._increment_skipped()
                    continue

                try:
                    writer.writerow(validate_csv_row(row))
                except ValueError as exc:
                    self._increment_skipped()
                    self._log(f"skipped row: {exc}")
                    continue

                output.flush()
                self._increment_rows()
                if self.snapshot().rows_written <= 5:
                    self._log(",".join(row))
        self._log("USB serial capture stopped")

    def _metadata(self) -> dict:
        metadata = super()._metadata()
        metadata.update(
            {
                "transport": "usb_serial",
                "port": self.port,
                "baud": self.baud,
            }
        )
        return metadata


class UdpCsvCaptureWorker(_CsvCaptureWorker):
    def __init__(
        self,
        output_path: str | Path,
        host: str = DEFAULT_BIND_HOST,
        port: int = DEFAULT_UDP_PORT,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        max_log_lines: int = 300,
    ) -> None:
        super().__init__(output_path, max_log_lines=max_log_lines)
        self.host = str(host).strip()
        self.port = int(port)
        self.buffer_size = int(buffer_size)

    def _run(self) -> None:
        self._log(f"binding UDP {self.host}:{self.port}")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver, self.output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output:
            receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            receiver.bind((self.host, self.port))
            receiver.settimeout(0.5)

            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            output.flush()
            self._log(f"capturing UDP CSV to {self.output_path}")

            while not self._stop_event.is_set():
                try:
                    payload, address = receiver.recvfrom(self.buffer_size)
                except TimeoutError:
                    continue

                source = f"{address[0]}:{address[1]}"
                self._increment_source(source)
                for line in parse_udp_payload_lines(payload):
                    if line.startswith("#"):
                        self._increment_comment()
                        self._log(f"{source} {line}")
                        continue

                    try:
                        row = parse_serial_csv_line(line)
                    except csv.Error:
                        self._increment_skipped()
                        continue

                    if row is None or row == CSV_COLUMNS:
                        continue

                    try:
                        writer.writerow(validate_csv_row(row))
                    except ValueError as exc:
                        self._increment_skipped()
                        self._log(f"skipped row from {source}: {exc}")
                        continue

                    output.flush()
                    self._increment_rows()
                    if self.snapshot().rows_written <= 5:
                        self._log(",".join(row))
        self._log("UDP capture stopped")

    def _metadata(self) -> dict:
        metadata = super()._metadata()
        metadata.update(
            {
                "transport": "udp_wifi_ap",
                "bind_host": self.host,
                "port": self.port,
                "source_counts": self.snapshot().source_counts,
            }
        )
        return metadata
