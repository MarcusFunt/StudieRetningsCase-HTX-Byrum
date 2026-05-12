from __future__ import annotations

import html
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import pandas as pd
import panel as pn

from scripts.generate_secrets import (
    DEFAULT_PASSWORD_LENGTH,
    DEFAULT_SSID,
    WifiSecrets,
    generated_password,
    read_existing,
    validate_password,
    validate_ssid,
    write_all,
)

from .analysis import FlowAnalysisSettings, run_flow_analysis, write_analysis_outputs
from .calibration import (
    DEFAULT_CHARUCO_DPI,
    DEFAULT_CHARUCO_MARGIN_MM,
    DEFAULT_CHARUCO_MARKER_LENGTH_MM,
    DEFAULT_CHARUCO_SQUARE_LENGTH_MM,
    DEFAULT_CHARUCO_SQUARES_X,
    DEFAULT_CHARUCO_SQUARES_Y,
    generate_charuco_board,
)
from .capture import (
    DEFAULT_BIND_HOST,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_UDP_PORT,
    SerialCsvCaptureWorker,
    UdpCsvCaptureWorker,
)
from .geometry import load_calibration
from .ui_helpers import (
    directory_options,
    display_path,
    file_options,
    keep_or_first,
    resolve_path,
    serial_port_options,
)
from .usb_calibration_capture import capture_usb_calibration_photos

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTRINSICS_IMAGE_DIR = "data/calibration_images/charuco"
GROUND_IMAGE_DIR = "data/calibration_images/ground"


def _status_html(title: str, message: str, kind: str = "info") -> str:
    safe_kind = kind if kind in {"info", "success", "danger"} else "info"
    return f"""
    <div class="pedflow-status pedflow-status-{safe_kind}">
      <div class="pedflow-status-title">{html.escape(title)}</div>
      <div class="pedflow-status-body">{html.escape(message)}</div>
    </div>
    """


def _section_title(title: str, eyebrow: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"""
        <div class="pedflow-section-title">
          <strong>{html.escape(title)}</strong>
          <span>{html.escape(eyebrow)}</span>
        </div>
        """,
        margin=(0, 0, 2, 0),
    )


def _capture_metrics_html(rows: int, skipped: int, comments: int, sources: int) -> str:
    metrics = {
        "Rows": f"{rows:,}",
        "Skipped": f"{skipped:,}",
        "Comments": f"{comments:,}",
        "Sources": f"{sources:,}",
    }
    items = [
        f"""
        <div class="pedflow-metric">
          <div class="pedflow-metric-label">{html.escape(label)}</div>
          <div class="pedflow-metric-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in metrics.items()
    ]
    return f'<div class="pedflow-metrics">{"".join(items)}</div>'


@dataclass(frozen=True)
class OperationTaskSnapshot:
    running: bool
    title: str
    log_lines: list[str]
    error: str | None
    success_message: str | None


class OperationTask:
    def __init__(self, max_log_lines: int = 400) -> None:
        self._lock = threading.Lock()
        self._log_lines: deque[str] = deque(maxlen=max_log_lines)
        self._thread: threading.Thread | None = None
        self._running = False
        self._title = "Idle"
        self._error: str | None = None
        self._success_message: str | None = None

    def start(self, title: str, task: Callable[[Callable[[str], None]], str | None]) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        with self._lock:
            self._running = True
            self._title = title
            self._error = None
            self._success_message = None
            self._log_lines.clear()
        self.log(f"{title} started")
        self._thread = threading.Thread(target=self._run, args=(task,), name=title, daemon=True)
        self._thread.start()
        return True

    def snapshot(self) -> OperationTaskSnapshot:
        with self._lock:
            return OperationTaskSnapshot(
                running=self._running,
                title=self._title,
                log_lines=list(self._log_lines),
                error=self._error,
                success_message=self._success_message,
            )

    def log(self, message: str) -> None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._log_lines.append(f"[{timestamp}] {message}")

    def _run(self, task: Callable[[Callable[[str], None]], str | None]) -> None:
        try:
            success_message = task(self.log)
            with self._lock:
                self._success_message = success_message or "Operation complete."
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            self.log(f"error: {exc}")
        finally:
            with self._lock:
                self._running = False


class OperationsPanel:
    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root
        self.capture_worker: SerialCsvCaptureWorker | UdpCsvCaptureWorker | None = None
        self.task = OperationTask()

        port_options = serial_port_options()
        detection_options = self._detection_options()
        calibration_options = self._calibration_options()
        output_options = self._output_options()
        image_dir_options = self._calibration_image_dir_options()
        board_output_options = self._board_output_options()

        self.capture_transport = pn.widgets.RadioButtonGroup(
            name="Capture transport",
            options=["Wi-Fi UDP", "USB serial"],
            value="Wi-Fi UDP",
            button_type="default",
        )
        self.capture_output = pn.widgets.TextInput(
            name="Output CSV",
            value="data/detections/gui_capture.csv",
        )
        self.capture_host = pn.widgets.TextInput(name="UDP bind host", value=DEFAULT_BIND_HOST)
        self.capture_udp_port = pn.widgets.IntInput(
            name="UDP port",
            value=DEFAULT_UDP_PORT,
            start=1,
        )
        self.capture_buffer_size = pn.widgets.IntInput(
            name="UDP buffer size",
            value=DEFAULT_BUFFER_SIZE,
            start=512,
        )
        self.capture_serial_port = pn.widgets.Select(
            name="USB serial port",
            options=port_options,
            value=keep_or_first("COM5", port_options),
        )
        self.capture_baud = pn.widgets.IntInput(name="Baud", value=115200, start=9600)
        self.refresh_capture_button = pn.widgets.Button(name="Refresh ports", height=38)
        self.start_capture_button = pn.widgets.Button(
            name="Start capture",
            button_type="primary",
            height=42,
        )
        self.stop_capture_button = pn.widgets.Button(name="Stop capture", height=42)
        self.capture_status = pn.pane.HTML(
            _status_html("Ready", "Start a Wi-Fi UDP or USB serial CSV capture.")
        )
        self.capture_metrics = pn.pane.HTML(
            _capture_metrics_html(0, 0, 0, 0),
            sizing_mode="stretch_width",
        )
        self.capture_log = pn.widgets.TextAreaInput(
            name="Capture log",
            value="",
            disabled=True,
            height=260,
        )

        self.image_output_dir = pn.widgets.Select(
            name="Calibration image folder",
            options=image_dir_options,
            value=keep_or_first(INTRINSICS_IMAGE_DIR, image_dir_options),
        )
        self.image_port = pn.widgets.Select(
            name="USB serial port",
            options=port_options,
            value=keep_or_first("COM5", port_options),
        )
        self.image_baud = pn.widgets.IntInput(name="Baud", value=115200, start=9600)
        self.image_basename = pn.widgets.TextInput(name="Photo basename", value="charuco_usb")
        self.image_count = pn.widgets.IntInput(name="Photos", value=1, start=1)
        self.image_interval_s = pn.widgets.FloatInput(name="Interval (s)", value=1.0, start=0.0)
        self.image_settle_delay_s = pn.widgets.FloatInput(
            name="USB settle delay (s)",
            value=2.0,
            start=0.0,
        )
        self.image_timeout_s = pn.widgets.FloatInput(name="Timeout (s)", value=60.0, start=1.0)
        self.capture_image_button = pn.widgets.Button(
            name="Capture USB image",
            button_type="primary",
            height=42,
        )

        self.secret_ssid = pn.widgets.TextInput(name="SSID", value=DEFAULT_SSID)
        self.secret_password = pn.widgets.PasswordInput(
            name="Password",
            placeholder="Leave empty to reuse or generate",
        )
        self.secret_password_length = pn.widgets.IntInput(
            name="Generated password length",
            value=DEFAULT_PASSWORD_LENGTH,
            start=8,
            end=63,
        )
        self.secret_rotate = pn.widgets.Checkbox(name="Rotate password", value=False)
        self.generate_secrets_button = pn.widgets.Button(
            name="Generate Wi-Fi secrets",
            button_type="primary",
            height=42,
        )

        self.board_output_dir = pn.widgets.Select(
            name="Board output folder",
            options=board_output_options,
            value=keep_or_first("outputs/charuco_board", board_output_options),
        )
        self.board_basename = pn.widgets.TextInput(name="Board basename", value="charuco_board")
        self.board_dictionary = pn.widgets.Select(
            name="ArUco dictionary",
            options=["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_7X7_1000"],
            value="DICT_5X5_100",
        )
        self.board_squares_x = pn.widgets.IntInput(
            name="Squares x",
            value=DEFAULT_CHARUCO_SQUARES_X,
            start=3,
        )
        self.board_squares_y = pn.widgets.IntInput(
            name="Squares y",
            value=DEFAULT_CHARUCO_SQUARES_Y,
            start=3,
        )
        self.board_square_length_mm = pn.widgets.FloatInput(
            name="Square length (mm)",
            value=DEFAULT_CHARUCO_SQUARE_LENGTH_MM,
        )
        self.board_marker_length_mm = pn.widgets.FloatInput(
            name="Marker length (mm)",
            value=DEFAULT_CHARUCO_MARKER_LENGTH_MM,
        )
        self.board_dpi = pn.widgets.IntInput(name="DPI", value=DEFAULT_CHARUCO_DPI, start=72)
        self.board_margin_mm = pn.widgets.FloatInput(
            name="Margin (mm)",
            value=DEFAULT_CHARUCO_MARGIN_MM,
        )
        self.generate_board_button = pn.widgets.Button(
            name="Generate ChArUco board",
            button_type="primary",
            height=42,
        )

        self.analysis_detections_path = pn.widgets.Select(
            name="Detections CSV",
            options=detection_options,
            value=keep_or_first("data/detections/session.csv", detection_options),
        )
        self.analysis_calibration_path = pn.widgets.Select(
            name="Calibration JSON",
            options=calibration_options,
            value=keep_or_first("outputs/calibration.json", calibration_options),
        )
        self.analysis_output_dir = pn.widgets.Select(
            name="Output folder",
            options=output_options,
            value=keep_or_first("outputs/analysis", output_options),
        )
        self.analysis_save_outputs = pn.widgets.Checkbox(
            name="Write outputs and QA files", value=True
        )
        self.run_logged_analysis_button = pn.widgets.Button(
            name="Run analysis with logs",
            button_type="primary",
            height=42,
        )
        self.refresh_files_button = pn.widgets.Button(name="Refresh files", height=38)

        self.operation_status = pn.pane.HTML(
            _status_html("Ready", "Run setup, OpenCV, USB image, or PedPy operations.")
        )
        self.operation_log = pn.widgets.TextAreaInput(
            name="Operation log",
            value="",
            disabled=True,
            height=360,
        )

        self.refresh_capture_button.on_click(self._on_refresh_files)
        self.refresh_files_button.on_click(self._on_refresh_files)
        self.start_capture_button.on_click(self._on_start_capture)
        self.stop_capture_button.on_click(self._on_stop_capture)
        self.capture_image_button.on_click(self._on_capture_usb_image)
        self.generate_secrets_button.on_click(self._on_generate_secrets)
        self.generate_board_button.on_click(self._on_generate_board)
        self.run_logged_analysis_button.on_click(self._on_run_logged_analysis)
        self._periodic = pn.state.add_periodic_callback(self._refresh, period=500, start=False)

    def panel(self) -> pn.Column:
        return pn.Column(
            pn.Tabs(
                ("Data Capture", self._capture_panel()),
                ("USB Images", self._usb_image_panel()),
                ("Setup Scripts", self._setup_panel()),
                ("Analysis Logs", self._analysis_log_panel()),
                dynamic=True,
                css_classes=["pedflow-tabs"],
            ),
            sizing_mode="stretch_width",
        )

    def _capture_panel(self) -> pn.Column:
        controls = pn.Column(
            _section_title("CSV capture", "scripts/capture_wifi.py and scripts/capture_serial.py"),
            self.capture_transport,
            self.capture_output,
            self.capture_host,
            self.capture_udp_port,
            self.capture_buffer_size,
            self.capture_serial_port,
            self.capture_baud,
            self.refresh_capture_button,
            pn.Row(
                self.start_capture_button,
                self.stop_capture_button,
                css_classes=["pedflow-button-row"],
            ),
            css_classes=["pedflow-controls"],
            max_width=420,
        )
        return pn.Column(
            pn.Row(
                controls,
                pn.Column(
                    self.capture_status,
                    self.capture_metrics,
                    self.capture_log,
                    css_classes=["pedflow-results"],
                ),
                css_classes=["pedflow-layout", "pedflow-workspace"],
            ),
            sizing_mode="stretch_width",
        )

    def _usb_image_panel(self) -> pn.Column:
        controls = pn.Column(
            _section_title("Image capture", "USB-only CALIB_CAPTURE"),
            self.image_output_dir,
            self.image_port,
            self.image_baud,
            self.image_basename,
            self.image_count,
            self.image_interval_s,
            self.image_settle_delay_s,
            self.image_timeout_s,
            self.capture_image_button,
            css_classes=["pedflow-controls"],
            max_width=420,
        )
        return pn.Column(
            pn.Row(
                controls,
                pn.Column(
                    self.operation_status,
                    self.operation_log,
                    css_classes=["pedflow-results"],
                ),
                css_classes=["pedflow-layout", "pedflow-workspace"],
            ),
            sizing_mode="stretch_width",
        )

    def _setup_panel(self) -> pn.Column:
        secrets_controls = pn.Column(
            _section_title("Wi-Fi secrets", "scripts/generate_secrets.py"),
            self.secret_ssid,
            self.secret_password,
            self.secret_password_length,
            self.secret_rotate,
            self.generate_secrets_button,
            css_classes=["pedflow-controls"],
            max_width=420,
        )
        board_controls = pn.Column(
            _section_title("ChArUco board", "scripts/generate_charuco_board.py"),
            self.board_output_dir,
            self.board_basename,
            self.board_dictionary,
            self.board_squares_x,
            self.board_squares_y,
            self.board_square_length_mm,
            self.board_marker_length_mm,
            self.board_dpi,
            self.board_margin_mm,
            self.generate_board_button,
            css_classes=["pedflow-controls"],
            max_width=420,
        )
        return pn.Column(
            pn.Row(
                pn.Column(secrets_controls, board_controls),
                pn.Column(
                    self.operation_status,
                    self.operation_log,
                    css_classes=["pedflow-results"],
                ),
                css_classes=["pedflow-layout", "pedflow-workspace"],
            ),
            sizing_mode="stretch_width",
        )

    def _analysis_log_panel(self) -> pn.Column:
        controls = pn.Column(
            _section_title("PedPy analysis", "logged run_flow_analysis"),
            self.analysis_detections_path,
            self.analysis_calibration_path,
            self.analysis_save_outputs,
            self.analysis_output_dir,
            self.refresh_files_button,
            self.run_logged_analysis_button,
            css_classes=["pedflow-controls"],
            max_width=420,
        )
        return pn.Column(
            pn.Row(
                controls,
                pn.Column(
                    self.operation_status,
                    self.operation_log,
                    css_classes=["pedflow-results"],
                ),
                css_classes=["pedflow-layout", "pedflow-workspace"],
            ),
            sizing_mode="stretch_width",
        )

    def _detection_options(self) -> list[str]:
        return file_options(
            self.project_root,
            ("data/detections/*.csv", "data/detections/**/*.csv"),
            ("data/detections/session.csv",),
        )

    def _calibration_options(self) -> list[str]:
        return file_options(
            self.project_root,
            ("outputs/calibration*.json", "outputs/**/*.json"),
            ("outputs/calibration.json",),
        )

    def _output_options(self) -> list[str]:
        return directory_options(self.project_root, ("outputs",), ("outputs/analysis",))

    def _calibration_image_dir_options(self) -> list[str]:
        return directory_options(
            self.project_root,
            ("data/calibration_images",),
            (INTRINSICS_IMAGE_DIR, GROUND_IMAGE_DIR),
        )

    def _board_output_options(self) -> list[str]:
        return directory_options(self.project_root, ("outputs",), ("outputs/charuco_board",))

    def _on_refresh_files(self, _event: object) -> None:
        port_options = serial_port_options()
        detection_options = self._detection_options()
        calibration_options = self._calibration_options()
        output_options = self._output_options()
        image_dir_options = self._calibration_image_dir_options()
        board_output_options = self._board_output_options()

        self.capture_serial_port.options = port_options
        self.capture_serial_port.value = keep_or_first(
            str(self.capture_serial_port.value),
            port_options,
        )
        self.image_port.options = port_options
        self.image_port.value = keep_or_first(str(self.image_port.value), port_options)
        self.analysis_detections_path.options = detection_options
        self.analysis_detections_path.value = keep_or_first(
            str(self.analysis_detections_path.value),
            detection_options,
        )
        self.analysis_calibration_path.options = calibration_options
        self.analysis_calibration_path.value = keep_or_first(
            str(self.analysis_calibration_path.value),
            calibration_options,
        )
        self.analysis_output_dir.options = output_options
        self.analysis_output_dir.value = keep_or_first(
            str(self.analysis_output_dir.value),
            output_options,
        )
        self.image_output_dir.options = image_dir_options
        self.image_output_dir.value = keep_or_first(
            str(self.image_output_dir.value),
            image_dir_options,
        )
        self.board_output_dir.options = board_output_options
        self.board_output_dir.value = keep_or_first(
            str(self.board_output_dir.value),
            board_output_options,
        )

    def _on_start_capture(self, _event: object) -> None:
        if self.capture_worker and self.capture_worker.snapshot().running:
            self.capture_status.object = _status_html(
                "Capture already running",
                "Stop the active capture before starting another one.",
                kind="danger",
            )
            return

        output_path = resolve_path(self.project_root, str(self.capture_output.value))
        if self.capture_transport.value == "USB serial":
            self.capture_worker = SerialCsvCaptureWorker(
                port=str(self.capture_serial_port.value),
                baud=int(self.capture_baud.value),
                output_path=output_path,
            )
        else:
            self.capture_worker = UdpCsvCaptureWorker(
                output_path=output_path,
                host=str(self.capture_host.value),
                port=int(self.capture_udp_port.value),
                buffer_size=int(self.capture_buffer_size.value),
            )
        self.capture_worker.start()
        self.capture_status.object = _status_html(
            "Capture running",
            f"Writing CSV rows to {display_path(self.project_root, output_path)}.",
            kind="success",
        )
        self._periodic.start()

    def _on_stop_capture(self, _event: object) -> None:
        if self.capture_worker is not None:
            self.capture_worker.stop()
            snapshot = self.capture_worker.snapshot()
            self.capture_status.object = _status_html(
                "Capture stopped",
                f"Wrote {snapshot.rows_written:,} rows to "
                f"{display_path(self.project_root, snapshot.output_path)}.",
            )
            self._refresh()

    def _on_capture_usb_image(self, _event: object) -> None:
        def task(log: Callable[[str], None]) -> str:
            output_dir = resolve_path(self.project_root, str(self.image_output_dir.value))
            log(f"USB image capture on {self.image_port.value} at {self.image_baud.value} baud")
            log(
                "USB calibration/reference photos will be saved in "
                f"{display_path(self.project_root, output_dir)}"
            )
            captures = capture_usb_calibration_photos(
                port=str(self.image_port.value),
                baud=int(self.image_baud.value),
                output_dir=output_dir,
                basename=str(self.image_basename.value),
                count=int(self.image_count.value),
                interval_s=float(self.image_interval_s.value),
                settle_delay_s=float(self.image_settle_delay_s.value),
                timeout_s=float(self.image_timeout_s.value),
                settings={"workflow": "operations_usb_image_capture"},
            )
            for capture in captures:
                log(
                    "saved "
                    f"{display_path(self.project_root, capture.image_path)} "
                    f"({capture.jpeg_byte_count:,} bytes)"
                )
            return f"Saved {len(captures)} USB calibration image(s)."

        self._start_task("USB image capture", task)

    def _on_generate_secrets(self, _event: object) -> None:
        def task(log: Callable[[str], None]) -> str:
            ssid = validate_ssid(str(self.secret_ssid.value))
            existing = read_existing(self.project_root / "secrets" / "pedflow_wifi.json")
            if self.secret_password.value:
                password = validate_password(str(self.secret_password.value))
                log("using explicit WPA2 password from the GUI")
            elif bool(self.secret_rotate.value) or existing is None:
                password = generated_password(int(self.secret_password_length.value))
                log("generated a new local WPA2 password")
            else:
                password = existing.password
                log("reused the existing local WPA2 password")
            paths = write_all(self.project_root, WifiSecrets(ssid=ssid, password=password))
            log(f"firmware header: {display_path(self.project_root, paths.firmware_header)}")
            log(f"laptop credentials: {display_path(self.project_root, paths.laptop_text)}")
            log(
                "Windows Wi-Fi profile: "
                f"{display_path(self.project_root, paths.windows_wifi_profile)}"
            )
            return f"Generated local Wi-Fi secrets for SSID {ssid!r}."

        self._start_task("Generate Wi-Fi secrets", task)

    def _on_generate_board(self, _event: object) -> None:
        def task(log: Callable[[str], None]) -> str:
            output_dir = resolve_path(self.project_root, str(self.board_output_dir.value))
            log(f"OpenCV {cv2.__version__}: generating ChArUco board")
            metadata = generate_charuco_board(
                output_dir=output_dir,
                squares_x=int(self.board_squares_x.value),
                squares_y=int(self.board_squares_y.value),
                square_length_mm=float(self.board_square_length_mm.value),
                marker_length_mm=float(self.board_marker_length_mm.value),
                dictionary_name=str(self.board_dictionary.value),
                dpi=int(self.board_dpi.value),
                margin_mm=float(self.board_margin_mm.value),
                basename=str(self.board_basename.value),
            )
            log(f"PNG: {display_path(self.project_root, Path(str(metadata['png_path'])))}")
            log(f"PDF: {display_path(self.project_root, Path(str(metadata['pdf_path'])))}")
            log(
                f"metadata: {display_path(self.project_root, Path(str(metadata['metadata_path'])))}"
            )
            log(
                f"board size: {metadata['board_width_m']:.3f} m x "
                f"{metadata['board_height_m']:.3f} m"
            )
            return "Generated ChArUco board assets."

        self._start_task("Generate ChArUco board", task)

    def _on_run_logged_analysis(self, _event: object) -> None:
        def task(log: Callable[[str], None]) -> str:
            detections_path = resolve_path(
                self.project_root,
                str(self.analysis_detections_path.value),
            )
            calibration_path = resolve_path(
                self.project_root,
                str(self.analysis_calibration_path.value),
            )
            output_dir = resolve_path(self.project_root, str(self.analysis_output_dir.value))
            if not detections_path.exists():
                raise FileNotFoundError(
                    f"Detections CSV not found: {display_path(self.project_root, detections_path)}"
                )
            if not calibration_path.exists():
                raise FileNotFoundError(
                    "Calibration JSON not found: "
                    f"{display_path(self.project_root, calibration_path)}"
                )

            log(f"reading detections: {display_path(self.project_root, detections_path)}")
            detections = pd.read_csv(detections_path)
            log(f"loaded {len(detections):,} raw detection row(s)")
            log(f"reading calibration: {display_path(self.project_root, calibration_path)}")
            calibration = load_calibration(calibration_path)
            started_utc = datetime.now(UTC).isoformat()
            result = run_flow_analysis(
                detections,
                calibration,
                FlowAnalysisSettings(),
                logger=log,
            )
            if bool(self.analysis_save_outputs.value):
                log(
                    "writing analysis CSV, QA, and manifest outputs to "
                    f"{display_path(self.project_root, output_dir)}"
                )
                write_analysis_outputs(
                    result,
                    output_dir,
                    settings=FlowAnalysisSettings(),
                    detections_path=detections_path,
                    calibration_path=calibration_path,
                    started_utc=started_utc,
                    ended_utc=datetime.now(UTC).isoformat(),
                )
            track_count = result.tracks["track_id"].nunique() if not result.tracks.empty else 0
            log(f"PedPy output grid cells: {len(result.grid):,}")
            return (
                f"Analysis complete: {len(result.detections_ground):,} ground detections, "
                f"{track_count:,} tracks."
            )

        self._start_task("Logged PedPy analysis", task)

    def _start_task(self, title: str, task: Callable[[Callable[[str], None]], str | None]) -> None:
        if not self.task.start(title, task):
            self.operation_status.object = _status_html(
                "Operation already running",
                "Wait for the active operation to finish.",
                kind="danger",
            )
            return
        self.operation_status.object = _status_html("Running", title)
        self._periodic.start()

    def _refresh(self) -> None:
        capture_running = False
        if self.capture_worker is not None:
            snapshot = self.capture_worker.snapshot()
            capture_running = snapshot.running
            self.capture_log.value = "\n".join(snapshot.log_lines)
            self.capture_metrics.object = _capture_metrics_html(
                snapshot.rows_written,
                snapshot.skipped_row_count,
                snapshot.comment_row_count,
                len(snapshot.source_counts),
            )
            if snapshot.error:
                self.capture_status.object = _status_html(
                    "Capture failed",
                    snapshot.error,
                    kind="danger",
                )
            elif snapshot.running:
                self.capture_status.object = _status_html(
                    "Capture running",
                    f"Wrote {snapshot.rows_written:,} row(s) to "
                    f"{display_path(self.project_root, snapshot.output_path)}.",
                    kind="success",
                )

        task_snapshot = self.task.snapshot()
        self.operation_log.value = "\n".join(task_snapshot.log_lines)
        if task_snapshot.error:
            self.operation_status.object = _status_html(
                f"{task_snapshot.title} failed",
                task_snapshot.error,
                kind="danger",
            )
        elif task_snapshot.running:
            self.operation_status.object = _status_html("Running", task_snapshot.title)
        elif task_snapshot.success_message:
            self.operation_status.object = _status_html(
                "Complete",
                task_snapshot.success_message,
                kind="success",
            )

        if not capture_running and not task_snapshot.running:
            self._periodic.stop()
