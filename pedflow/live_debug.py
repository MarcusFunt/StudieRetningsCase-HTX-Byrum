from __future__ import annotations

import csv
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import serial

from .analysis import FlowAnalysisSettings, run_flow_analysis
from .geometry import bbox_foot_points, validate_detection_input
from .metrics import summarize_flow
from .serial_protocol import CSV_COLUMNS, parse_serial_csv_line, validate_csv_row

DEFAULT_DEBUG_SETTINGS = FlowAnalysisSettings(
    confidence_threshold=0.5,
    max_matching_speed_m_s=4.5,
    close_after_s=1.5,
    min_track_duration_s=0.0,
    min_detections=2,
    smoothing_alpha=0.4,
    speed_window_s=0.75,
    stop_speed_threshold_m_s=0.2,
    stop_duration_threshold_s=1.0,
    grid_size_m=0.5,
)


@dataclass(frozen=True)
class ParsedDebugLine:
    kind: str
    row: list[str] | None = None
    text: str = ""
    error: str | None = None


@dataclass(frozen=True)
class SerialDebugSnapshot:
    rows: list[list[str]]
    comments: list[str]
    rows_read: int
    skipped_rows: int
    running: bool
    error: str | None


@dataclass(frozen=True)
class LiveDebugResult:
    detections: pd.DataFrame
    bbox_overlay: pd.DataFrame
    ground: pd.DataFrame
    tracks: pd.DataFrame
    summary: pd.DataFrame
    track_summaries: pd.DataFrame
    grid: pd.DataFrame
    calibration_loaded: bool
    error: str | None = None


def parse_debug_serial_line(line: str) -> ParsedDebugLine:
    stripped = line.strip()
    if not stripped:
        return ParsedDebugLine(kind="empty")
    if stripped.startswith("#"):
        return ParsedDebugLine(kind="comment", text=stripped)

    try:
        row = parse_serial_csv_line(stripped)
    except csv.Error as exc:
        return ParsedDebugLine(kind="invalid", text=stripped, error=str(exc))

    if row is None:
        return ParsedDebugLine(kind="empty")
    if row == CSV_COLUMNS:
        return ParsedDebugLine(kind="header", text=stripped)

    try:
        return ParsedDebugLine(kind="row", row=validate_csv_row(row), text=stripped)
    except ValueError as exc:
        return ParsedDebugLine(kind="invalid", text=stripped, error=str(exc))


def rows_to_detections(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)

    validated_rows = [validate_csv_row(row) for row in rows]
    detections = pd.DataFrame(validated_rows, columns=CSV_COLUMNS)
    return validate_detection_input(detections)


def build_bbox_overlay(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_ms",
                "frame_id",
                "detection_id",
                "bbox_x0",
                "bbox_y0",
                "bbox_x1",
                "bbox_y1",
                "foot_x",
                "foot_y",
                "confidence",
                "target",
            ]
        )

    validated = validate_detection_input(detections)
    foot_points = bbox_foot_points(validated)
    output = pd.DataFrame(
        {
            "timestamp_ms": validated["timestamp_ms"],
            "frame_id": validated["frame_id"],
            "detection_id": validated["detection_id"],
            "bbox_x0": validated["bbox_x"],
            "bbox_y0": validated["bbox_y"],
            "bbox_x1": validated["bbox_x"] + validated["bbox_w"],
            "bbox_y1": validated["bbox_y"] + validated["bbox_h"],
            "foot_x": foot_points[:, 0],
            "foot_y": foot_points[:, 1],
            "confidence": validated["confidence"],
            "target": validated["target"],
        }
    )
    return output.reset_index(drop=True)


def _empty_flow_result(
    detections: pd.DataFrame, calibration_loaded: bool, error: str | None = None
):
    return LiveDebugResult(
        detections=detections,
        bbox_overlay=build_bbox_overlay(detections),
        ground=pd.DataFrame(),
        tracks=pd.DataFrame(),
        summary=summarize_flow(pd.DataFrame()),
        track_summaries=pd.DataFrame(),
        grid=pd.DataFrame(),
        calibration_loaded=calibration_loaded,
        error=error,
    )


def analyze_live_debug_rows(
    rows: list[list[str]],
    calibration: dict[str, Any] | None,
    settings: FlowAnalysisSettings = DEFAULT_DEBUG_SETTINGS,
) -> LiveDebugResult:
    detections = rows_to_detections(rows)
    if detections.empty:
        return _empty_flow_result(detections, calibration_loaded=calibration is not None)
    if calibration is None:
        return _empty_flow_result(detections, calibration_loaded=False)

    try:
        analysis = run_flow_analysis(detections, calibration, settings)
    except Exception as exc:
        return _empty_flow_result(detections, calibration_loaded=True, error=str(exc))

    return LiveDebugResult(
        detections=detections,
        bbox_overlay=build_bbox_overlay(detections),
        ground=analysis.detections_ground,
        tracks=analysis.tracks,
        summary=analysis.summary,
        track_summaries=analysis.track_summaries,
        grid=analysis.grid,
        calibration_loaded=True,
    )


@dataclass
class SerialDebugReader:
    """Read firmware debug CSV rows from USB serial on a background thread.

    Call ``start`` to open the port, ``snapshot`` from the UI thread to copy buffered state, and
    ``stop`` before discarding the reader. A repeated CSV header is treated as a firmware stream reset.
    """

    port: str
    baud: int = 115200
    max_rows: int = 2000
    _rows: deque[list[str]] = field(init=False, repr=False)
    _comments: deque[str] = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)
    _stop_event: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _rows_read: int = field(default=0, init=False, repr=False)
    _skipped_rows: int = field(default=0, init=False, repr=False)
    _error: str | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rows = deque(maxlen=int(self.max_rows))
        self._comments = deque(maxlen=50)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        with self._lock:
            self._error = None
            self._running = True
            self._rows.clear()
            self._comments.clear()
            self._rows_read = 0
            self._skipped_rows = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="SerialDebugReader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            self._running = False

    def snapshot(self) -> SerialDebugSnapshot:
        with self._lock:
            return SerialDebugSnapshot(
                rows=list(self._rows),
                comments=list(self._comments),
                rows_read=self._rows_read,
                skipped_rows=self._skipped_rows,
                running=self._running,
                error=self._error,
            )

    def _run(self) -> None:
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as device:
                device.dtr = True
                device.rts = True
                while not self._stop_event.is_set():
                    raw_line = device.readline()
                    if not raw_line:
                        continue
                    self._handle_line(raw_line.decode("utf-8", errors="replace"))
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._running = False

    def _handle_line(self, line: str) -> None:
        parsed = parse_debug_serial_line(line)
        with self._lock:
            if parsed.kind == "row" and parsed.row is not None:
                self._rows.append(parsed.row)
                self._rows_read += 1
            elif parsed.kind == "comment":
                self._comments.append(parsed.text)
            elif parsed.kind == "invalid":
                self._skipped_rows += 1
                self._error = parsed.error
            elif parsed.kind == "header":
                self._rows.clear()
                self._rows_read = 0
                self._skipped_rows = 0
                self._error = None
                self._comments.append("#status,csv_header_reset")
