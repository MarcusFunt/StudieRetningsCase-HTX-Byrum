from __future__ import annotations

import html
import io
import json
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
    from .calibration import (
        calibrate_camera_from_charuco,
        compute_ground_homography,
        generate_charuco_board,
        merge_and_save_calibration,
        read_marker_csv,
    )
    from .debug_panel import UsbDebugPanel
    from .geometry import load_calibration, save_calibration
    from .ui_helpers import (
        directory_options,
        display_path,
        file_options,
        keep_or_first,
        resolve_path,
        serial_port_options,
    )
    from .usb_calibration_capture import capture_usb_calibration_photos
except ImportError:
    from pedflow.analysis import (
        FlowAnalysisResult,
        FlowAnalysisSettings,
        run_flow_analysis,
        write_analysis_outputs,
    )
    from pedflow.calibration import (
        calibrate_camera_from_charuco,
        compute_ground_homography,
        generate_charuco_board,
        merge_and_save_calibration,
        read_marker_csv,
    )
    from pedflow.debug_panel import UsbDebugPanel
    from pedflow.geometry import load_calibration, save_calibration
    from pedflow.ui_helpers import (
        directory_options,
        display_path,
        file_options,
        keep_or_first,
        resolve_path,
        serial_port_options,
    )
    from pedflow.usb_calibration_capture import capture_usb_calibration_photos


PROJECT_ROOT = Path(__file__).resolve().parents[1]

pn.extension("tabulator")
pn.config.sizing_mode = "stretch_width"
hv.extension("bokeh")


_UI_CSS = """
:root {
  --pedflow-bg: #f3f6f8;
  --pedflow-bg-strong: #eaf1f3;
  --pedflow-panel: #ffffff;
  --pedflow-ink: #152029;
  --pedflow-muted: #5a6977;
  --pedflow-line: #d5e0e5;
  --pedflow-line-strong: #b9c9d0;
  --pedflow-accent: #24787f;
  --pedflow-accent-strong: #176168;
  --pedflow-focus: #90d0d3;
}
body {
  background: var(--pedflow-bg);
  color: var(--pedflow-ink);
  font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
.bk-FastListTemplate {
  background: var(--pedflow-bg);
}
.bk-FastListTemplate .bk-main {
  padding-top: 18px !important;
}
#header {
  box-shadow: 0 8px 24px rgba(15, 29, 38, 0.14);
}
#header .title {
  font-weight: 720;
  letter-spacing: 0;
}
.bk-btn-primary {
  background-color: var(--pedflow-accent) !important;
  border-color: var(--pedflow-accent) !important;
  border-radius: 7px !important;
  box-shadow: 0 8px 18px rgba(36, 120, 127, 0.16);
  font-weight: 700 !important;
}
.bk-btn-primary:hover,
.bk-btn-primary:focus {
  background-color: var(--pedflow-accent-strong) !important;
  border-color: var(--pedflow-accent-strong) !important;
}
.bk-btn-default {
  background: #f8fbfc !important;
  border: 1px solid var(--pedflow-line-strong) !important;
  border-radius: 7px !important;
  color: var(--pedflow-ink) !important;
  font-weight: 650 !important;
}
.bk-btn-default:hover,
.bk-btn-default:focus {
  background: #edf5f6 !important;
  border-color: var(--pedflow-accent) !important;
}
.pedflow-layout {
  align-items: flex-start;
  gap: 18px;
}
.pedflow-workspace {
  align-items: stretch;
}
.pedflow-workspace > .pedflow-controls {
  flex: 0 0 390px;
}
.pedflow-workspace > *:not(.pedflow-controls) {
  min-width: 0;
}
.pedflow-controls {
  background: var(--pedflow-panel);
  border: 1px solid var(--pedflow-line);
  border-radius: 8px;
  box-sizing: border-box;
  box-shadow: 0 10px 26px rgba(21, 32, 41, 0.05);
  min-width: 0;
  padding: 18px 18px 20px;
}
.pedflow-button-row {
  gap: 10px;
}
.pedflow-button-row .bk-btn {
  min-height: 42px;
}
.pedflow-section-title {
  border-bottom: 1px solid var(--pedflow-line);
  margin: -2px 0 12px;
  padding-bottom: 11px;
}
.pedflow-section-title strong {
  color: var(--pedflow-ink);
  display: block;
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0;
}
.pedflow-section-title span {
  color: var(--pedflow-muted);
  display: block;
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.04em;
  margin-top: 3px;
  text-transform: uppercase;
}
.pedflow-preview {
  background: #f8fbfc;
  border: 1px dashed var(--pedflow-line-strong);
  border-radius: 8px;
  min-height: 246px;
  overflow: hidden;
  padding: 14px;
}
.pedflow-preview-empty {
  align-items: center;
  color: var(--pedflow-muted);
  display: flex;
  font-size: 13px;
  font-weight: 650;
  justify-content: center;
  min-height: 214px;
  text-align: center;
}
.pedflow-board-image img {
  background: #ffffff;
  border: 1px solid var(--pedflow-line);
  border-radius: 6px;
  box-shadow: 0 8px 20px rgba(21, 32, 41, 0.08);
  object-fit: contain;
}
.pedflow-controls .bk-input,
.pedflow-controls select,
.pedflow-controls input[type="text"],
.pedflow-controls input[type="number"],
.pedflow-controls input[type="file"] {
  border-color: var(--pedflow-line-strong) !important;
  border-radius: 6px !important;
  min-height: 38px;
}
.pedflow-controls .bk-input:focus,
.pedflow-controls select:focus,
.pedflow-controls input:focus {
  border-color: var(--pedflow-accent) !important;
  box-shadow: 0 0 0 3px var(--pedflow-focus) !important;
}
.pedflow-accordion {
  border: 1px solid var(--pedflow-line);
  border-radius: 8px;
  overflow: hidden;
}
.pedflow-accordion button {
  background: #ffffff !important;
  border: 0 !important;
  border-bottom: 1px solid var(--pedflow-line) !important;
  color: var(--pedflow-ink) !important;
  min-height: 42px;
}
.pedflow-accordion h3 {
  font-size: 14px !important;
  font-weight: 750 !important;
}
.pedflow-inline-section {
  background: #f8fbfc;
  border: 1px solid var(--pedflow-line);
  border-radius: 8px;
  margin: -2px 0 6px;
  padding: 12px;
}
.pedflow-empty {
  align-items: center;
  background: linear-gradient(180deg, #fbfdfd 0%, var(--pedflow-bg-strong) 100%);
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
  box-shadow: 0 10px 26px rgba(21, 32, 41, 0.04);
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
  box-shadow: 0 10px 26px rgba(21, 32, 41, 0.04);
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
.pedflow-tabs .bk-tab {
  color: var(--pedflow-muted);
  font-weight: 700;
  padding-left: 16px;
  padding-right: 16px;
}
.pedflow-tabs .bk-tab.bk-active {
  color: var(--pedflow-accent-strong);
}
@media (max-width: 1100px) {
  .pedflow-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 720px) {
  #header .title {
    display: block;
    flex: 1 1 auto !important;
    font-size: 18px !important;
    line-height: 1.15 !important;
    max-width: 100%;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  #header #header-items {
    flex: 0 0 8px !important;
    width: 8px !important;
  }
  #header .pn-toggle-theme {
    flex: 0 0 85px !important;
  }
  #header .app-header {
    flex: 1 1 auto !important;
    max-width: calc(100vw - 118px);
    min-width: 0;
  }
  #header .pn-busy-container {
    display: none !important;
  }
  .bk-FastListTemplate .bk-main {
    padding-left: 10px !important;
    padding-right: 10px !important;
  }
  .pedflow-layout {
    display: flex !important;
    flex-direction: column !important;
    gap: 14px;
  }
  .pedflow-layout > *,
  .pedflow-controls {
    flex: 1 1 auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
    width: 100% !important;
  }
  .pedflow-workspace > .pedflow-controls {
    flex-basis: auto !important;
  }
  .pedflow-tabs .bk-tab {
    padding-left: 10px;
    padding-right: 10px;
  }
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
    return resolve_path(project_root, value)


def _display_path(project_root: Path, path: Path) -> str:
    return display_path(project_root, path)


def _rounded_frame(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.round(digits)


def _empty_plot(message: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"""
        <div class="pedflow-empty">
          <strong>{html.escape(message)}</strong>
          <span>Waiting for analysis results.</span>
        </div>
        """,
        styles={"min-height": "360px"},
    )


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


def _preview_empty(message: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<div class="pedflow-preview-empty">{html.escape(message)}</div>',
        sizing_mode="stretch_width",
    )


def _paths_plot(tracks: pd.DataFrame) -> hv.core.Dimensioned | pn.pane.HTML:
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


def _count_plot(tracks: pd.DataFrame, bin_seconds: int = 60) -> hv.core.Dimensioned | pn.pane.HTML:
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
) -> hv.core.Dimensioned | pn.pane.HTML:
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


class AnalysisPanel:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.result: FlowAnalysisResult | None = None

        detection_options = self._detection_options()
        calibration_options = self._calibration_options()
        output_options = self._output_options()
        self.detections_path = pn.widgets.Select(
            name="Detection session",
            options=detection_options,
            value=keep_or_first("data/detections/session.csv", detection_options),
        )
        self.calibration_path = pn.widgets.Select(
            name="Calibration",
            options=calibration_options,
            value=keep_or_first("outputs/calibration.json", calibration_options),
        )
        self.output_dir = pn.widgets.Select(
            name="Output folder",
            options=output_options,
            value=keep_or_first("outputs/analysis", output_options),
        )
        self.refresh_files_button = pn.widgets.Button(name="Refresh files", height=38)
        self.upload_toggle_button = pn.widgets.Button(name="Upload files", height=38)
        self.settings_toggle_button = pn.widgets.Button(name="Analysis settings", height=38)
        self.detections_upload = pn.widgets.FileInput(name="Upload detections CSV", accept=".csv,text/csv")
        self.calibration_upload = pn.widgets.FileInput(
            name="Upload calibration JSON",
            accept=".json,application/json",
        )
        self.save_outputs = pn.widgets.Checkbox(name="Write CSV outputs", value=True)

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
        self.metrics = pn.pane.HTML(_metrics_html(0, 0, 0, 0), sizing_mode="stretch_width")

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
            css_classes=["pedflow-tabs"],
        )
        self.upload_section = pn.Column(
            self.detections_upload,
            self.calibration_upload,
            visible=False,
            css_classes=["pedflow-inline-section"],
        )
        self.settings_section = pn.Column(
            self.confidence_threshold,
            self.max_matching_speed,
            self.close_after,
            self.min_track_duration,
            self.min_detections,
            self.smoothing_alpha,
            self.speed_window,
            self.stop_speed_threshold,
            self.stop_duration_threshold,
            self.grid_size,
            visible=False,
            css_classes=["pedflow-inline-section"],
        )

        self.refresh_files_button.on_click(self._on_refresh_files)
        self.upload_toggle_button.on_click(self._on_toggle_upload)
        self.settings_toggle_button.on_click(self._on_toggle_settings)
        self.run_button.on_click(self._on_run)

    def panel(self) -> pn.Column:
        workflow_controls = pn.Column(
            _section_title("Analyze session", "Workflow"),
            self.detections_path,
            self.calibration_path,
            self.refresh_files_button,
            self.save_outputs,
            self.output_dir,
            self.upload_toggle_button,
            self.upload_section,
            self.settings_toggle_button,
            self.settings_section,
            self.run_button,
            css_classes=["pedflow-controls"],
            max_width=420,
            sizing_mode="stretch_width",
        )
        tables = pn.Tabs(
            ("Summary", self.summary_table),
            ("Track summaries", self.track_summary_table),
            ("Grid metrics", self.grid_table),
            ("Processed tracks", self.tracks_table),
            dynamic=True,
            css_classes=["pedflow-tabs"],
        )
        results = pn.Column(
            self.status,
            self.metrics,
            pn.Tabs(
                ("Plots", self.plots),
                ("Tables", tables),
                dynamic=True,
                css_classes=["pedflow-tabs"],
            ),
            sizing_mode="stretch_width",
        )
        return pn.Column(
            pn.Row(workflow_controls, results, css_classes=["pedflow-layout", "pedflow-workspace"]),
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

    def _on_refresh_files(self, _event: object) -> None:
        detection_options = self._detection_options()
        calibration_options = self._calibration_options()
        output_options = self._output_options()
        self.detections_path.options = detection_options
        self.detections_path.value = keep_or_first(str(self.detections_path.value), detection_options)
        self.calibration_path.options = calibration_options
        self.calibration_path.value = keep_or_first(str(self.calibration_path.value), calibration_options)
        self.output_dir.options = output_options
        self.output_dir.value = keep_or_first(str(self.output_dir.value), output_options)

    def _on_toggle_upload(self, _event: object) -> None:
        self.upload_section.visible = not self.upload_section.visible
        self.upload_toggle_button.name = (
            "Hide upload files" if self.upload_section.visible else "Upload files"
        )

    def _on_toggle_settings(self, _event: object) -> None:
        self.settings_section.visible = not self.settings_section.visible
        self.settings_toggle_button.name = (
            "Hide analysis settings" if self.settings_section.visible else "Analysis settings"
        )

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
                write_analysis_outputs(self.result, output)

            self._update_outputs(self.result)
            message = (
                f"Analysis complete: {len(self.result.detections_ground):,} ground detections, "
                f"{self.result.tracks['track_id'].nunique() if not self.result.tracks.empty else 0:,} tracks."
            )
            if self.save_outputs.value:
                message += f" CSV outputs written to {self.output_dir.value}."
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


class CalibrationPanel:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

        board_output_options = self._board_output_options()
        board_metadata_options = self._board_metadata_options()
        image_dir_options = self._calibration_image_dir_options()
        intrinsics_options = self._intrinsics_options()
        marker_options = self._marker_options()
        calibration_output_options = self._calibration_output_options()
        port_options = serial_port_options()

        self.board_output_dir = pn.widgets.Select(
            name="Output folder",
            options=board_output_options,
            value=keep_or_first("outputs/charuco_board", board_output_options),
        )
        self.board_basename = pn.widgets.TextInput(name="File basename", value="charuco_board")
        self.squares_x = pn.widgets.IntInput(name="Squares x", value=7, start=3)
        self.squares_y = pn.widgets.IntInput(name="Squares y", value=5, start=3)
        self.square_length_mm = pn.widgets.FloatInput(name="Square length (mm)", value=35.0)
        self.marker_length_mm = pn.widgets.FloatInput(name="Marker length (mm)", value=25.0)
        self.dictionary = pn.widgets.Select(
            name="ArUco dictionary",
            options=["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_7X7_1000"],
            value="DICT_5X5_100",
        )
        self.dpi = pn.widgets.IntInput(name="DPI", value=300, start=72)
        self.margin_mm = pn.widgets.FloatInput(name="Margin (mm)", value=10.0)
        self.refresh_board_files_button = pn.widgets.Button(name="Refresh files", height=38)
        self.generate_board_button = pn.widgets.Button(
            name="Generate board",
            button_type="primary",
            height=42,
        )
        self.board_status = pn.pane.HTML(_status_html("Ready", "Generate a printable ChArUco board."))

        self.image_dir = pn.widgets.Select(
            name="ChArUco image folder",
            options=image_dir_options,
            value=keep_or_first("data/calibration_images/charuco", image_dir_options),
        )
        self.board_metadata_path = pn.widgets.Select(
            name="Board metadata JSON",
            options=board_metadata_options,
            value=keep_or_first("outputs/charuco_board/charuco_board.json", board_metadata_options),
        )
        self.intrinsics_output_path = pn.widgets.Select(
            name="Intrinsics output JSON",
            options=intrinsics_options,
            value=keep_or_first("outputs/calibration_intrinsics.json", intrinsics_options),
        )
        self.min_corners = pn.widgets.IntInput(name="Min corners per image", value=8, start=4)
        self.capture_port = pn.widgets.Select(
            name="USB serial port",
            options=port_options,
            value=keep_or_first("COM5", port_options),
        )
        self.capture_baud = pn.widgets.IntInput(name="Baud", value=115200, start=9600)
        self.capture_basename = pn.widgets.TextInput(name="Photo basename", value="charuco_usb")
        self.capture_count = pn.widgets.IntInput(name="Photos", value=1, start=1)
        self.capture_interval_s = pn.widgets.FloatInput(name="Interval (s)", value=1.0, start=0.0)
        self.capture_timeout_s = pn.widgets.FloatInput(name="Timeout (s)", value=30.0, start=1.0)
        self.capture_settle_delay_s = pn.widgets.FloatInput(
            name="USB settle delay (s)",
            value=2.0,
            start=0.0,
        )
        self.capture_usb_photo_button = pn.widgets.Button(
            name="Take USB calibration photo",
            button_type="primary",
            height=42,
        )
        self.refresh_intrinsics_files_button = pn.widgets.Button(name="Refresh files", height=38)
        self.calibrate_intrinsics_button = pn.widgets.Button(
            name="Calibrate intrinsics",
            button_type="primary",
            height=42,
        )
        self.capture_status = pn.pane.HTML(
            _status_html(
                "USB only",
                "Photos are requested from the local USB serial port and saved to the selected image folder.",
            )
        )
        self.intrinsics_status = pn.pane.HTML(
            _status_html("Ready", "Place calibration images in the selected folder.")
        )

        self.intrinsics_path = pn.widgets.Select(
            name="Intrinsics JSON",
            options=intrinsics_options,
            value=keep_or_first("outputs/calibration_intrinsics.json", intrinsics_options),
        )
        self.markers_csv = pn.widgets.Select(
            name="Ground markers CSV",
            options=marker_options,
            value=keep_or_first("data/ground_markers.csv", marker_options),
        )
        self.calibration_output_path = pn.widgets.Select(
            name="Calibration output JSON",
            options=calibration_output_options,
            value=keep_or_first("outputs/calibration.json", calibration_output_options),
        )
        self.ransac_threshold_m = pn.widgets.FloatInput(name="RANSAC threshold (m)", value=0.10)
        self.refresh_homography_files_button = pn.widgets.Button(name="Refresh files", height=38)
        self.compute_homography_button = pn.widgets.Button(
            name="Build calibration",
            button_type="primary",
            height=42,
        )
        self.homography_status = pn.pane.HTML(
            _status_html("Ready", "Use measured marker correspondences to build calibration.")
        )

        self.refresh_board_files_button.on_click(self._on_refresh_calibration_files)
        self.refresh_intrinsics_files_button.on_click(self._on_refresh_calibration_files)
        self.refresh_homography_files_button.on_click(self._on_refresh_calibration_files)
        self.generate_board_button.on_click(self._on_generate_board)
        self.capture_usb_photo_button.on_click(self._on_capture_usb_photo)
        self.calibrate_intrinsics_button.on_click(self._on_calibrate_intrinsics)
        self.compute_homography_button.on_click(self._on_compute_homography)

        self.board_preview = pn.Column(css_classes=["pedflow-preview"], sizing_mode="stretch_width")
        self._refresh_board_preview(
            _resolve_path(self.project_root, self.board_output_dir.value)
            / f"{self.board_basename.value}.png"
        )

    def panel(self) -> pn.Tabs:
        return pn.Tabs(
            ("ChArUco Board", self._board_panel()),
            ("Camera Intrinsics", self._intrinsics_panel()),
            ("Ground Homography", self._homography_panel()),
            dynamic=True,
            css_classes=["pedflow-tabs"],
        )

    def _board_panel(self) -> pn.Column:
        return pn.Column(
            pn.Row(
                pn.Column(
                    _section_title("Board output", "ChArUco"),
                    self.board_output_dir,
                    self.board_basename,
                    self.dictionary,
                    self.refresh_board_files_button,
                    _section_title("Current board", "Preview"),
                    self.board_preview,
                    css_classes=["pedflow-controls"],
                ),
                pn.Column(
                    _section_title("Board geometry", "Print"),
                    self.squares_x,
                    self.squares_y,
                    self.square_length_mm,
                    self.marker_length_mm,
                    self.dpi,
                    self.margin_mm,
                    self.generate_board_button,
                    css_classes=["pedflow-controls"],
                ),
                css_classes=["pedflow-layout"],
            ),
            self.board_status,
        )

    def _intrinsics_panel(self) -> pn.Column:
        return pn.Column(
            pn.Row(
                pn.Column(
                    _section_title("Image set", "Intrinsics"),
                    self.image_dir,
                    self.board_metadata_path,
                    self.intrinsics_output_path,
                    self.min_corners,
                    self.refresh_intrinsics_files_button,
                    self.calibrate_intrinsics_button,
                    css_classes=["pedflow-controls"],
                ),
                pn.Column(
                    _section_title("USB photo capture", "Calibration"),
                    self.capture_port,
                    self.capture_baud,
                    self.capture_basename,
                    self.capture_count,
                    self.capture_interval_s,
                    self.capture_timeout_s,
                    self.capture_settle_delay_s,
                    self.capture_usb_photo_button,
                    css_classes=["pedflow-controls"],
                ),
                css_classes=["pedflow-layout"],
            ),
            self.capture_status,
            self.intrinsics_status,
        )

    def _homography_panel(self) -> pn.Column:
        return pn.Column(
            pn.Row(
                pn.Column(
                    _section_title("Marker mapping", "Homography"),
                    self.intrinsics_path,
                    self.markers_csv,
                    self.calibration_output_path,
                    self.ransac_threshold_m,
                    self.refresh_homography_files_button,
                    self.compute_homography_button,
                    css_classes=["pedflow-controls"],
                ),
                css_classes=["pedflow-layout"],
            ),
            self.homography_status,
        )

    def _board_output_options(self) -> list[str]:
        return directory_options(self.project_root, ("outputs",), ("outputs/charuco_board",))

    def _board_metadata_options(self) -> list[str]:
        return file_options(
            self.project_root,
            ("outputs/charuco_board/*.json", "outputs/**/*.json"),
            ("outputs/charuco_board/charuco_board.json",),
        )

    def _calibration_image_dir_options(self) -> list[str]:
        return directory_options(
            self.project_root,
            ("data/calibration_images",),
            ("data/calibration_images/charuco",),
        )

    def _intrinsics_options(self) -> list[str]:
        return file_options(
            self.project_root,
            ("outputs/calibration_intrinsics*.json", "outputs/**/*.json"),
            ("outputs/calibration_intrinsics.json",),
        )

    def _marker_options(self) -> list[str]:
        return file_options(
            self.project_root,
            ("data/ground_markers*.csv", "data/**/*.csv"),
            ("data/ground_markers.csv",),
        )

    def _calibration_output_options(self) -> list[str]:
        return file_options(
            self.project_root,
            ("outputs/calibration*.json", "outputs/**/*.json"),
            ("outputs/calibration.json",),
        )

    def _on_refresh_calibration_files(self, _event: object) -> None:
        board_output_options = self._board_output_options()
        board_metadata_options = self._board_metadata_options()
        image_dir_options = self._calibration_image_dir_options()
        intrinsics_options = self._intrinsics_options()
        marker_options = self._marker_options()
        calibration_output_options = self._calibration_output_options()
        port_options = serial_port_options()

        self.board_output_dir.options = board_output_options
        self.board_output_dir.value = keep_or_first(str(self.board_output_dir.value), board_output_options)
        self.board_metadata_path.options = board_metadata_options
        self.board_metadata_path.value = keep_or_first(
            str(self.board_metadata_path.value),
            board_metadata_options,
        )
        self.image_dir.options = image_dir_options
        self.image_dir.value = keep_or_first(str(self.image_dir.value), image_dir_options)
        self.capture_port.options = port_options
        self.capture_port.value = keep_or_first(str(self.capture_port.value), port_options)
        self.intrinsics_output_path.options = intrinsics_options
        self.intrinsics_output_path.value = keep_or_first(
            str(self.intrinsics_output_path.value),
            intrinsics_options,
        )
        self.intrinsics_path.options = intrinsics_options
        self.intrinsics_path.value = keep_or_first(str(self.intrinsics_path.value), intrinsics_options)
        self.markers_csv.options = marker_options
        self.markers_csv.value = keep_or_first(str(self.markers_csv.value), marker_options)
        self.calibration_output_path.options = calibration_output_options
        self.calibration_output_path.value = keep_or_first(
            str(self.calibration_output_path.value),
            calibration_output_options,
        )

    def _on_generate_board(self, _event: object) -> None:
        self.generate_board_button.loading = True
        self.board_status.object = _status_html("Running", "Generating ChArUco board.")

        try:
            output_dir = _resolve_path(self.project_root, self.board_output_dir.value)
            metadata = generate_charuco_board(
                output_dir=output_dir,
                squares_x=int(self.squares_x.value),
                squares_y=int(self.squares_y.value),
                square_length_mm=float(self.square_length_mm.value),
                marker_length_mm=float(self.marker_length_mm.value),
                dictionary_name=str(self.dictionary.value),
                dpi=int(self.dpi.value),
                margin_mm=float(self.margin_mm.value),
                basename=str(self.board_basename.value),
            )
            message = (
                f"Generated PDF, PNG, and metadata in "
                f"{_display_path(self.project_root, output_dir)}. "
                f"Board size: {metadata['board_width_m']:.3f} m x {metadata['board_height_m']:.3f} m."
            )
            self._refresh_board_preview(Path(str(metadata["png_path"])))
            self.board_status.object = _status_html("Complete", message, kind="success")
        except Exception as exc:
            self.board_status.object = _status_html("Board generation failed", str(exc), kind="danger")
        finally:
            self.generate_board_button.loading = False

    def _refresh_board_preview(self, image_path: Path) -> None:
        if image_path.exists():
            preview = pn.pane.PNG(
                str(image_path),
                height=214,
                sizing_mode="scale_width",
                css_classes=["pedflow-board-image"],
            )
        else:
            preview = _preview_empty("Board preview appears after generation.")
        self.board_preview[:] = [preview]

    def _on_capture_usb_photo(self, _event: object) -> None:
        self.capture_usb_photo_button.loading = True
        self.capture_status.object = _status_html(
            "Running",
            f"Requesting photo capture over USB serial on {self.capture_port.value}.",
        )

        try:
            output_dir = _resolve_path(self.project_root, self.image_dir.value)
            captures = capture_usb_calibration_photos(
                port=str(self.capture_port.value),
                baud=int(self.capture_baud.value),
                output_dir=output_dir,
                basename=str(self.capture_basename.value),
                count=int(self.capture_count.value),
                interval_s=float(self.capture_interval_s.value),
                settle_delay_s=float(self.capture_settle_delay_s.value),
                timeout_s=float(self.capture_timeout_s.value),
            )
            saved_names = ", ".join(
                _display_path(self.project_root, capture.image_path) for capture in captures[:3]
            )
            if len(captures) > 3:
                saved_names = f"{saved_names}, ..."
            self._on_refresh_calibration_files(_event)
            self.capture_status.object = _status_html(
                "Complete",
                f"Saved {len(captures)} USB calibration photo(s): {saved_names}.",
                kind="success",
            )
        except Exception as exc:
            self.capture_status.object = _status_html(
                "USB photo capture failed",
                str(exc),
                kind="danger",
            )
        finally:
            self.capture_usb_photo_button.loading = False

    def _on_calibrate_intrinsics(self, _event: object) -> None:
        self.calibrate_intrinsics_button.loading = True
        self.intrinsics_status.object = _status_html("Running", "Calibrating camera intrinsics.")

        try:
            image_dir = _resolve_path(self.project_root, self.image_dir.value)
            image_paths = sorted(
                list(image_dir.glob("*.jpg"))
                + list(image_dir.glob("*.jpeg"))
                + list(image_dir.glob("*.png"))
            )
            if not image_paths:
                raise FileNotFoundError(
                    f"No JPG, JPEG, or PNG files found in {_display_path(self.project_root, image_dir)}"
                )

            metadata_path = _resolve_path(self.project_root, self.board_metadata_path.value)
            output_path = _resolve_path(self.project_root, self.intrinsics_output_path.value)
            intrinsics = calibrate_camera_from_charuco(
                image_paths=image_paths,
                board_metadata_path=metadata_path,
                min_corners=int(self.min_corners.value),
            )
            save_calibration(intrinsics, output_path)
            message = (
                f"Saved {_display_path(self.project_root, output_path)} from "
                f"{len(intrinsics['used_images'])} images. "
                f"RMS reprojection error: {intrinsics['rms_reprojection_error_px']:.3f} px."
            )
            self.intrinsics_status.object = _status_html("Complete", message, kind="success")
        except Exception as exc:
            self.intrinsics_status.object = _status_html(
                "Intrinsics calibration failed",
                str(exc),
                kind="danger",
            )
        finally:
            self.calibrate_intrinsics_button.loading = False

    def _on_compute_homography(self, _event: object) -> None:
        self.compute_homography_button.loading = True
        self.homography_status.object = _status_html("Running", "Building ground calibration.")

        try:
            intrinsics_path = _resolve_path(self.project_root, self.intrinsics_path.value)
            markers_path = _resolve_path(self.project_root, self.markers_csv.value)
            output_path = _resolve_path(self.project_root, self.calibration_output_path.value)

            intrinsics = load_calibration(intrinsics_path)
            markers = read_marker_csv(markers_path)
            homography = compute_ground_homography(
                marker_points=markers,
                camera_matrix=np.asarray(intrinsics["K"], dtype=np.float64),
                dist_coeffs=np.asarray(intrinsics["dist"], dtype=np.float64),
                ransac_threshold_m=float(self.ransac_threshold_m.value),
            )
            calibration = merge_and_save_calibration(intrinsics, homography, output_path)
            message = (
                f"Saved {_display_path(self.project_root, output_path)}. "
                f"Mean residual: {calibration['ground_marker_mean_residual_m']:.3f} m; "
                f"max residual: {calibration['ground_marker_max_residual_m']:.3f} m."
            )
            self.homography_status.object = _status_html("Complete", message, kind="success")
        except Exception as exc:
            self.homography_status.object = _status_html(
                "Ground calibration failed",
                str(exc),
                kind="danger",
            )
        finally:
            self.compute_homography_button.loading = False


class PedFlowDashboard:
    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.analysis = AnalysisPanel(project_root)
        self.calibration = CalibrationPanel(project_root)
        self.debug = UsbDebugPanel(project_root)

    def panel(self) -> pn.template.FastListTemplate:
        template = pn.template.FastListTemplate(
            title="Pedestrian Flow Logger",
            main=[
                pn.Tabs(
                    ("Analysis", self.analysis.panel()),
                    ("Calibration", self.calibration.panel()),
                    ("USB Debug", self.debug.panel()),
                    dynamic=True,
                    css_classes=["pedflow-tabs"],
                )
            ],
            accent_base_color="#256f78",
            header_background="#17212b",
        )
        return template


def build_app(project_root: Path = PROJECT_ROOT) -> pn.template.FastListTemplate:
    return PedFlowDashboard(project_root).panel()


app = build_app()
app.servable()
