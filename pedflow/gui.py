from __future__ import annotations

import html
import io
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn

try:
    from .analysis import (
        FlowAnalysisResult,
        FlowAnalysisSettings,
        run_flow_analysis,
        write_analysis_outputs,
    )
    from .geometry import load_calibration
except ImportError:
    from pedflow.analysis import (
        FlowAnalysisResult,
        FlowAnalysisSettings,
        run_flow_analysis,
        write_analysis_outputs,
    )
    from pedflow.geometry import load_calibration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_OPTIONS = {
    "Flow analysis": "notebooks/03_flow_analysis.ipynb",
    "Ground homography": "notebooks/02_ground_homography.ipynb",
    "Camera calibration": "notebooks/01_camera_calibration.ipynb",
}

pn.extension("tabulator")
pn.config.sizing_mode = "stretch_width"
hv.extension("bokeh")


_UI_CSS = """
:root {
  --pedflow-bg: #f4f7f6;
  --pedflow-panel: #ffffff;
  --pedflow-ink: #152029;
  --pedflow-muted: #62717d;
  --pedflow-line: #d8e1e5;
  --pedflow-accent: #1f7a83;
  --pedflow-accent-dark: #155b62;
  --pedflow-soft: #e8f6f7;
  --pedflow-warn: #f4a261;
}
body {
  background: var(--pedflow-bg);
  color: var(--pedflow-ink);
  font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
.pedflow-section-title h2,
.pedflow-section-title h3 {
  color: var(--pedflow-ink);
  font-weight: 700;
  letter-spacing: 0;
  margin: 0;
}
.pedflow-empty {
  align-items: center;
  background: linear-gradient(180deg, #f8fbfb 0%, #eef5f5 100%);
  border: 1px solid var(--pedflow-line);
  border-radius: 8px;
  color: var(--pedflow-muted);
  display: flex;
  flex-direction: column;
  font-family: Inter, "Segoe UI", system-ui, sans-serif;
  gap: 8px;
  justify-content: center;
  min-height: 360px;
  text-align: center;
}
.pedflow-empty strong {
  color: var(--pedflow-ink);
  font-size: 18px;
}
.pedflow-empty span {
  font-size: 14px;
}
.pedflow-status {
  align-items: flex-start;
  background: #ffffff;
  border: 1px solid var(--pedflow-line);
  border-left: 4px solid var(--pedflow-accent);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-height: 74px;
  padding: 16px 18px;
}
.pedflow-status-title {
  color: var(--pedflow-ink);
  font-size: 14px;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.pedflow-status-body {
  color: var(--pedflow-muted);
  font-size: 14px;
  font-weight: 500;
}
.pedflow-status-success {
  border-left-color: #2f855a;
}
.pedflow-status-danger {
  border-left-color: #c2410c;
}
.pedflow-metrics {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  width: 100%;
}
.pedflow-metric {
  background: #ffffff;
  border: 1px solid var(--pedflow-line);
  border-radius: 8px;
  min-height: 118px;
  padding: 18px 18px 16px;
}
.pedflow-metric-label {
  color: var(--pedflow-muted);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.pedflow-metric-value {
  color: var(--pedflow-ink);
  font-size: 42px;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.05;
  margin-top: 16px;
  white-space: nowrap;
}
.pedflow-metric-unit {
  color: var(--pedflow-muted);
  font-size: 18px;
  font-weight: 600;
  margin-left: 4px;
}
.pedflow-main-tabs .bk-tab {
  font-weight: 650;
}
.pedflow-sidebar-title h2 {
  color: var(--pedflow-ink);
  font-size: 18px;
  margin: 14px 0 8px;
}
.pedflow-sidebar-title h3 {
  color: var(--pedflow-muted);
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.06em;
  margin: 16px 0 8px;
  text-transform: uppercase;
}
.pedflow-sidebar-divider {
  background: var(--pedflow-line);
  height: 1px;
  margin: 14px 0 6px;
}
@media (max-width: 1100px) {
  .pedflow-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 720px) {
  .pedflow-metrics {
    grid-template-columns: minmax(0, 1fr);
  }
  .pedflow-metric-value {
    font-size: 34px;
  }
}
"""
pn.config.raw_css.append(_UI_CSS)


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _rounded_frame(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.round(digits)


def _empty_plot(message: str) -> pn.pane.Markdown:
    return pn.pane.HTML(
        f"""
        <div class="pedflow-empty">
          <strong>{html.escape(message)}</strong>
          <span>Waiting for analysis results.</span>
        </div>
        """,
        styles={
            "min-height": "360px",
        },
    )


def _paths_plot(tracks: pd.DataFrame) -> hv.core.Dimensioned | pn.pane.Markdown:
    if tracks.empty:
        return _empty_plot("No path data yet")

    plot = tracks.hvplot.line(
        x="smooth_ground_x_m",
        y="smooth_ground_y_m",
        by="track_id",
        hover_cols=["track_id", "timestamp_ms", "speed_m_s"],
        height=460,
        legend=False,
        line_width=2,
        alpha=0.75,
        title="Pedestrian Paths / Desire Lines",
        xlabel="Ground x (m)",
        ylabel="Ground y (m)",
    )
    return plot.opts(responsive=True, aspect="equal", show_grid=True)


def _count_plot(tracks: pd.DataFrame, bin_seconds: int = 60) -> hv.core.Dimensioned | pn.pane.Markdown:
    if tracks.empty:
        return _empty_plot("No count data yet")

    first_seen = tracks.groupby("track_id", as_index=False)["timestamp_ms"].min()
    elapsed_s = (first_seen["timestamp_ms"] - first_seen["timestamp_ms"].min()) / 1000.0
    bins = np.floor(elapsed_s / bin_seconds).astype(int)
    counts = (
        bins.value_counts()
        .sort_index()
        .rename_axis("minute_bin")
        .reset_index(name="new_tracks")
    )
    counts["minute"] = counts["minute_bin"] * bin_seconds / 60.0

    return counts.hvplot.bar(
        x="minute",
        y="new_tracks",
        height=360,
        color="#256f78",
        title="Pedestrian Count Over Time",
        xlabel="Minutes since start",
        ylabel="New tracks",
    ).opts(responsive=True, show_grid=True)


def _heatmap_plot(
    grid: pd.DataFrame,
    value_col: str,
    title: str,
    color_label: str,
    cmap: str,
) -> hv.core.Dimensioned | pn.pane.Markdown:
    if grid.empty or value_col not in grid.columns:
        return _empty_plot(f"No {title.lower()} data yet")

    plot = grid.hvplot.heatmap(
        x="grid_x_m",
        y="grid_y_m",
        C=value_col,
        cmap=cmap,
        colorbar=True,
        clabel=color_label,
        height=460,
        tools=["hover"],
        title=title,
        xlabel="Ground x (m)",
        ylabel="Ground y (m)",
    )
    return plot.opts(responsive=True, aspect="equal", show_grid=True)


def _bottleneck_grid(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty or not {"detection_count", "median_speed_m_s"}.issubset(grid.columns):
        return grid.copy()
    output = grid.copy()
    output["bottleneck_index"] = output["detection_count"] / (
        output["median_speed_m_s"].fillna(0) + 0.2
    )
    return output


class JupyterEmbed:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.process: subprocess.Popen | None = None
        self.config_path: Path | None = None
        self.token = ""

        self.notebook = pn.widgets.Select(
            name="Notebook",
            options=NOTEBOOK_OPTIONS,
            value=NOTEBOOK_OPTIONS["Flow analysis"],
        )
        self.port = pn.widgets.IntInput(name="Jupyter port", value=8888, start=1024, end=65535)
        self.manual_url = pn.widgets.TextInput(
            name="Jupyter URL",
            placeholder="http://127.0.0.1:8888/notebooks/notebooks/03_flow_analysis.ipynb?token=...",
        )
        self.start_button = pn.widgets.Button(name="Start embedded Jupyter", button_type="primary")
        self.stop_button = pn.widgets.Button(name="Stop", button_type="light")
        self.load_button = pn.widgets.Button(name="Load URL", button_type="light")
        self.status = pn.pane.Alert(
            "Start Jupyter here, or paste an existing local Jupyter URL.",
            alert_type="info",
        )
        self.frame = pn.pane.HTML(self._frame_html(""), height=760, sanitize_html=False)

        self.start_button.on_click(self._on_start)
        self.stop_button.on_click(self._on_stop)
        self.load_button.on_click(self._on_load)

    def panel(self) -> pn.Column:
        controls = pn.Row(
            self.notebook,
            self.port,
            pn.Column(pn.Spacer(height=20), self.start_button, width=190),
            pn.Column(pn.Spacer(height=20), self.stop_button, width=80),
        )
        return pn.Column(
            pn.pane.Markdown(
                "Use this tab for the original notebooks without leaving the dashboard. "
                "The embedded server is local-only and protected with a generated token."
            ),
            controls,
            pn.Row(self.manual_url, pn.Column(pn.Spacer(height=20), self.load_button, width=90)),
            self.status,
            self.frame,
        )

    def _on_start(self, _event: object) -> None:
        try:
            url = self.start()
        except Exception as exc:
            self.status.object = f"Could not start Jupyter: {exc}"
            self.status.alert_type = "danger"
            return

        self.manual_url.value = url
        self.frame.object = self._frame_html(url)
        self.status.object = f"Embedded Jupyter is running on port {self.port.value}."
        self.status.alert_type = "success"

    def _on_stop(self, _event: object) -> None:
        self.stop()
        self.frame.object = self._frame_html("")
        self.status.object = "Embedded Jupyter stopped."
        self.status.alert_type = "info"

    def _on_load(self, _event: object) -> None:
        self.frame.object = self._frame_html(self.manual_url.value.strip())
        self.status.object = "Loaded the supplied Jupyter URL in the embedded frame."
        self.status.alert_type = "info"

    def start(self) -> str:
        if self.process is not None and self.process.poll() is None:
            return self._notebook_url()

        self.stop()
        self.token = secrets.token_urlsafe(24)
        self.config_path = self._write_config()
        command = [sys.executable, "-m", "notebook", "--config", str(self.config_path)]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

        self.process = subprocess.Popen(
            command,
            cwd=self.project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        self._wait_until_ready()
        return self._notebook_url()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

        if self.config_path is not None:
            try:
                self.config_path.unlink(missing_ok=True)
            except OSError:
                pass
        self.config_path = None

    def _write_config(self) -> Path:
        panel_origin = os.environ.get("PEDFLOW_PANEL_ORIGIN", "http://localhost:5006")
        frame_ancestors = f"frame-ancestors {panel_origin} http://localhost:5006 http://127.0.0.1:5006 'self'"
        config = "\n".join(
            [
                f"c.ServerApp.root_dir = {str(self.project_root)!r}",
                "c.ServerApp.ip = '127.0.0.1'",
                f"c.ServerApp.port = {int(self.port.value)}",
                "c.ServerApp.port_retries = 0",
                "c.ServerApp.open_browser = False",
                f"c.ServerApp.token = {self.token!r}",
                "c.ServerApp.password = ''",
                "c.ServerApp.disable_check_xsrf = False",
                "c.ServerApp.tornado_settings = {",
                "    'headers': {",
                f"        'Content-Security-Policy': {frame_ancestors!r}",
                "    }",
                "}",
                "",
            ]
        )
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix="_pedflow_jupyter_config.py",
            delete=False,
        )
        with handle:
            handle.write(config)
        return Path(handle.name)

    def _wait_until_ready(self) -> None:
        status_url = f"http://127.0.0.1:{int(self.port.value)}/api/status?token={self.token}"
        deadline = time.monotonic() + 15
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Jupyter exited early with code {self.process.returncode}. "
                    "The port may already be in use."
                )
            try:
                with urllib.request.urlopen(status_url, timeout=1) as response:
                    if response.status < 500:
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(0.5)

        raise RuntimeError(f"Jupyter did not become ready within 15 seconds: {last_error}")

    def _notebook_url(self) -> str:
        selected = _resolve_path(self.project_root, self.notebook.value)
        if selected.exists():
            relative = selected.relative_to(self.project_root).as_posix()
            notebook_path = urllib.parse.quote(relative, safe="/")
            path = f"/notebooks/{notebook_path}"
        else:
            path = "/tree"
        return f"http://127.0.0.1:{int(self.port.value)}{path}?token={self.token}"

    @staticmethod
    def _frame_html(url: str) -> str:
        if not url:
            return """
            <div style="height:720px;border:1px solid #d5dbe3;border-radius:6px;
                        display:flex;align-items:center;justify-content:center;
                        color:#4b5563;background:#f8fafc;font-family:sans-serif;">
                Start embedded Jupyter to load a notebook here.
            </div>
            """

        escaped_url = html.escape(url, quote=True)
        return f"""
        <iframe
            src="{escaped_url}"
            style="width:100%;height:720px;border:1px solid #d5dbe3;border-radius:6px;background:white;"
            allow="clipboard-read; clipboard-write; fullscreen">
        </iframe>
        """


class PedFlowDashboard:
    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root
        self.result: FlowAnalysisResult | None = None

        self.detections_path = pn.widgets.TextInput(
            name="Detections CSV",
            value="data/detections/session.csv",
        )
        self.calibration_path = pn.widgets.TextInput(
            name="Calibration JSON",
            value="outputs/calibration.json",
        )
        self.output_dir = pn.widgets.TextInput(name="Output folder", value="outputs/analysis")
        self.detections_upload = pn.widgets.FileInput(name="Upload detections CSV", accept=".csv,text/csv")
        self.calibration_upload = pn.widgets.FileInput(
            name="Upload calibration JSON",
            accept=".json,application/json",
        )
        self.save_outputs = pn.widgets.Checkbox(name="Write CSVs and PNG plots", value=True)

        self.confidence_threshold = pn.widgets.FloatSlider(
            name="Confidence threshold",
            start=0.0,
            end=1.0,
            step=0.05,
            value=0.6,
        )
        self.max_matching_speed = pn.widgets.FloatInput(name="Max matching speed (m/s)", value=4.5)
        self.close_after = pn.widgets.FloatInput(name="Close track after (s)", value=1.0)
        self.min_track_duration = pn.widgets.FloatInput(name="Min track duration (s)", value=1.5)
        self.min_detections = pn.widgets.IntInput(name="Min detections", value=4, start=1)
        self.smoothing_alpha = pn.widgets.FloatSlider(
            name="Smoothing alpha",
            start=0.0,
            end=1.0,
            step=0.05,
            value=0.3,
        )
        self.speed_window = pn.widgets.FloatInput(name="Speed window (s)", value=0.75)
        self.stop_speed_threshold = pn.widgets.FloatInput(name="Stop speed threshold (m/s)", value=0.2)
        self.stop_duration_threshold = pn.widgets.FloatInput(name="Stop duration threshold (s)", value=2.0)
        self.grid_size = pn.widgets.FloatInput(name="Grid size (m)", value=0.5)
        self.run_button = pn.widgets.Button(
            name="Run analysis",
            button_type="primary",
            height=42,
            sizing_mode="stretch_width",
        )
        self.status = pn.pane.HTML(_status_html("Ready", "Select files and run the analysis."))

        self.metrics = pn.pane.HTML(
            _metrics_html(0, 0, 0, 0),
            sizing_mode="stretch_width",
        )

        self.summary_table = pn.widgets.Tabulator(pd.DataFrame(), pagination="remote", page_size=10)
        self.track_summary_table = pn.widgets.Tabulator(pd.DataFrame(), pagination="remote", page_size=12)
        self.grid_table = pn.widgets.Tabulator(pd.DataFrame(), pagination="remote", page_size=12)
        self.tracks_table = pn.widgets.Tabulator(pd.DataFrame(), pagination="remote", page_size=12)

        self.plots = pn.Tabs(
            ("Paths", _empty_plot("No path data yet")),
            ("Count", _empty_plot("No count data yet")),
            ("Position", _empty_plot("No position heatmap data yet")),
            ("Speed", _empty_plot("No speed heatmap data yet")),
            ("Dwell", _empty_plot("No dwell map data yet")),
            ("Bottleneck", _empty_plot("No bottleneck data yet")),
            dynamic=True,
        )
        self.jupyter = JupyterEmbed(project_root)

        self.run_button.on_click(self._on_run)

    def panel(self) -> pn.template.FastListTemplate:
        template = pn.template.FastListTemplate(
            title="Pedestrian Flow Logger",
            sidebar=[
                pn.pane.Markdown("## Inputs", css_classes=["pedflow-sidebar-title"]),
                self.detections_path,
                self.detections_upload,
                self.calibration_path,
                self.calibration_upload,
                self.output_dir,
                self.save_outputs,
                self.run_button,
                pn.pane.HTML('<div class="pedflow-sidebar-divider"></div>', height=16),
                pn.pane.Markdown("### Tracking", css_classes=["pedflow-sidebar-title"]),
                self.confidence_threshold,
                self.max_matching_speed,
                self.close_after,
                self.min_track_duration,
                self.min_detections,
                self.smoothing_alpha,
                pn.pane.Markdown("### Grid and Stops", css_classes=["pedflow-sidebar-title"]),
                self.speed_window,
                self.stop_speed_threshold,
                self.stop_duration_threshold,
                self.grid_size,
            ],
            main=[
                self.status,
                self.metrics,
                pn.Tabs(
                    ("Plots", self.plots),
                    (
                        "Tables",
                        pn.Tabs(
                            ("Summary", self.summary_table),
                            ("Track summaries", self.track_summary_table),
                            ("Grid metrics", self.grid_table),
                            ("Processed tracks", self.tracks_table),
                            dynamic=True,
                        ),
                    ),
                    ("Jupyter", self.jupyter.panel()),
                    dynamic=True,
                    css_classes=["pedflow-main-tabs"],
                ),
            ],
            accent_base_color="#256f78",
            header_background="#17212b",
        )
        return template

    def _settings(self) -> FlowAnalysisSettings:
        return FlowAnalysisSettings(
            confidence_threshold=float(self.confidence_threshold.value),
            max_matching_speed_m_s=float(self.max_matching_speed.value),
            close_after_s=float(self.close_after.value),
            min_track_duration_s=float(self.min_track_duration.value),
            min_detections=int(self.min_detections.value),
            smoothing_alpha=float(self.smoothing_alpha.value),
            speed_window_s=float(self.speed_window.value),
            stop_speed_threshold_m_s=float(self.stop_speed_threshold.value),
            stop_duration_threshold_s=float(self.stop_duration_threshold.value),
            grid_size_m=float(self.grid_size.value),
        )

    def _read_detections(self) -> pd.DataFrame:
        if self.detections_upload.value:
            return pd.read_csv(io.BytesIO(self.detections_upload.value))

        path = _resolve_path(self.project_root, self.detections_path.value)
        if not path.exists():
            raise FileNotFoundError(
                f"Detections CSV not found: {_display_path(self.project_root, path)}"
            )
        return pd.read_csv(path)

    def _read_calibration(self) -> dict:
        if self.calibration_upload.value:
            return json.loads(self.calibration_upload.value.decode("utf-8-sig"))

        path = _resolve_path(self.project_root, self.calibration_path.value)
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration JSON not found: {_display_path(self.project_root, path)}"
            )
        return load_calibration(path)

    def _on_run(self, _event: object) -> None:
        self.run_button.loading = True
        self._set_status("Running", "Analysis is processing with the current inputs.")

        try:
            detections = self._read_detections()
            calibration = self._read_calibration()
            settings = self._settings()
            self.result = run_flow_analysis(detections, calibration, settings)

            if self.save_outputs.value:
                output = _resolve_path(self.project_root, self.output_dir.value)
                write_analysis_outputs(self.result, output, grid_size_m=settings.grid_size_m)

            self._update_outputs(self.result)
            message = (
                f"Analysis complete: {len(self.result.detections_ground):,} ground detections, "
                f"{self.result.tracks['track_id'].nunique() if not self.result.tracks.empty else 0:,} tracks."
            )
            if self.save_outputs.value:
                message += f" Outputs written to {self.output_dir.value}."
            self._set_status("Complete", message, kind="success")
        except Exception as exc:
            self._set_status("Analysis failed", str(exc), kind="danger")
        finally:
            self.run_button.loading = False

    def _set_status(self, title: str, message: str, kind: str = "info") -> None:
        self.status.object = _status_html(title, message, kind)

    def _update_outputs(self, result: FlowAnalysisResult) -> None:
        summary = result.summary.iloc[0] if not result.summary.empty else pd.Series(dtype=float)
        count = _indicator_value(summary.get("pedestrian_count", 0))
        rate = _indicator_value(summary.get("people_per_hour", 0))
        speed = _indicator_value(summary.get("median_speed_m_s", 0))
        dwell = _indicator_value(summary.get("dwell_points", 0))
        self.metrics.object = _metrics_html(count, rate, speed, dwell)

        self.summary_table.value = _rounded_frame(result.summary)
        self.track_summary_table.value = _rounded_frame(result.track_summaries)
        self.grid_table.value = _rounded_frame(result.grid)
        self.tracks_table.value = _rounded_frame(result.tracks)

        bottleneck_grid = _bottleneck_grid(result.grid)
        self.plots[:] = [
            ("Paths", _paths_plot(result.tracks)),
            ("Count", _count_plot(result.tracks)),
            (
                "Position",
                _heatmap_plot(
                    result.grid,
                    "detection_count",
                    "Pedestrian Position Heatmap",
                    "Detections",
                    "magma",
                ),
            ),
            (
                "Speed",
                _heatmap_plot(
                    result.grid.dropna(subset=["median_speed_m_s"])
                    if "median_speed_m_s" in result.grid.columns
                    else result.grid,
                    "median_speed_m_s",
                    "Median Speed Heatmap",
                    "Median speed (m/s)",
                    "plasma",
                ),
            ),
            (
                "Dwell",
                _heatmap_plot(
                    result.grid,
                    "dwell_points",
                    "Stop / Dwell Map",
                    "Dwell points",
                    "cividis",
                ),
            ),
            (
                "Bottleneck",
                _heatmap_plot(
                    bottleneck_grid,
                    "bottleneck_index",
                    "Bottleneck Index",
                    "Density / speed",
                    "inferno",
                ),
            ),
        ]


def _indicator_value(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(number):
        return 0.0
    return number


def _metrics_html(count: float, rate: float, speed: float, dwell: float) -> str:
    metrics = [
        ("Pedestrians", f"{count:,.0f}", ""),
        ("People / hour", f"{rate:,.1f}", ""),
        ("Median speed", f"{speed:,.2f}", "m/s"),
        ("Dwell points", f"{dwell:,.0f}", ""),
    ]
    items = []
    for label, value, unit in metrics:
        unit_html = f'<span class="pedflow-metric-unit">{html.escape(unit)}</span>' if unit else ""
        items.append(
            f"""
            <div class="pedflow-metric">
              <div class="pedflow-metric-label">{html.escape(label)}</div>
              <div class="pedflow-metric-value">{html.escape(value)}{unit_html}</div>
            </div>
            """
        )
    return f'<div class="pedflow-metrics">{"".join(items)}</div>'


def _status_html(title: str, message: str, kind: str = "info") -> str:
    safe_kind = kind if kind in {"info", "success", "danger"} else "info"
    return f"""
    <div class="pedflow-status pedflow-status-{safe_kind}">
      <div class="pedflow-status-title">{html.escape(title)}</div>
      <div class="pedflow-status-body">{html.escape(message)}</div>
    </div>
    """


def build_app(project_root: Path = PROJECT_ROOT) -> pn.template.FastListTemplate:
    return PedFlowDashboard(project_root).panel()


app = build_app()
app.servable()
