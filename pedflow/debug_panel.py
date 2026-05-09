from __future__ import annotations

import html
from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn

from .geometry import load_calibration
from .live_debug import SerialDebugReader, analyze_live_debug_rows
from .ui_helpers import file_options, keep_or_first, resolve_path, serial_port_options

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pn.extension("tabulator")
hv.extension("bokeh")


def _status_html(title: str, message: str, kind: str = "info") -> str:
    safe_kind = kind if kind in {"info", "success", "danger"} else "info"
    return f"""
    <div class="pedflow-status pedflow-status-{safe_kind}">
      <div class="pedflow-status-title">{html.escape(title)}</div>
      <div class="pedflow-status-body">{html.escape(message)}</div>
    </div>
    """


def _metrics_html(values: dict[str, str]) -> str:
    items = []
    for label, value in values.items():
        items.append(
            f"""
            <div class="pedflow-metric">
              <div class="pedflow-metric-label">{html.escape(label)}</div>
              <div class="pedflow-metric-value">{html.escape(value)}</div>
            </div>
            """
        )
    return f'<div class="pedflow-metrics">{"".join(items)}</div>'


def _empty_plot(message: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"""
        <div class="pedflow-empty">
          <strong>{html.escape(message)}</strong>
          <span>Start USB debug capture to populate this view.</span>
        </div>
        """,
        styles={"min-height": "360px"},
    )


def _rounded_frame(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.round(digits)


def _latest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "frame_id" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["frame_id"] == frame["frame_id"].max()].copy()


def _bbox_plot(bbox_overlay: pd.DataFrame) -> hv.core.Dimensioned | pn.pane.HTML:
    latest = _latest_frame(bbox_overlay)
    if latest.empty:
        return _empty_plot("No live bounding boxes yet")

    rectangles = hv.Rectangles(
        latest,
        kdims=["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"],
        vdims=["frame_id", "detection_id", "confidence", "target"],
    ).opts(
        color="#1f7a8c",
        fill_alpha=0.08,
        height=420,
        line_width=2,
        responsive=True,
        tools=["hover"],
        title="Latest Bounding Boxes",
        xlabel="Image x",
        ylabel="Image y",
    )
    foot_points = hv.Points(
        latest,
        kdims=["foot_x", "foot_y"],
        vdims=["frame_id", "detection_id", "confidence"],
    ).opts(color="#d1495b", marker="cross", size=9, tools=["hover"])
    return (rectangles * foot_points).opts(aspect="equal", invert_yaxis=True, show_grid=True)


def _ground_plot(ground: pd.DataFrame, tracks: pd.DataFrame) -> hv.core.Dimensioned | pn.pane.HTML:
    if ground.empty and tracks.empty:
        return _empty_plot("No calibrated ground contact points yet")

    layers: list[hv.core.Dimensioned] = []
    if not tracks.empty:
        paths = tracks.hvplot.line(
            x="smooth_ground_x_m",
            y="smooth_ground_y_m",
            by="track_id",
            hover_cols=["track_id", "timestamp_ms", "speed_m_s"],
            line_width=2,
            alpha=0.7,
            legend=False,
            height=420,
            title="Ground Contact Points and Tracks",
            xlabel="Ground x (m)",
            ylabel="Ground y (m)",
        )
        layers.append(paths)

    latest_ground = _latest_frame(ground)
    if not latest_ground.empty:
        points = latest_ground.hvplot.scatter(
            x="ground_x_m",
            y="ground_y_m",
            hover_cols=["frame_id", "detection_id", "confidence", "foot_x", "foot_y"],
            color="#d1495b",
            marker="cross",
            size=95,
            height=420,
            title="Ground Contact Points and Tracks",
            xlabel="Ground x (m)",
            ylabel="Ground y (m)",
        )
        layers.append(points)

    plot = layers[0]
    for layer in layers[1:]:
        plot = plot * layer
    return plot.opts(responsive=True, aspect="equal", show_grid=True)


def _grid_plot(grid: pd.DataFrame) -> hv.core.Dimensioned | pn.pane.HTML:
    if grid.empty or "detection_count" not in grid.columns:
        return _empty_plot("No PedPy grid data yet")

    return grid.hvplot.heatmap(
        x="grid_x_m",
        y="grid_y_m",
        C="detection_count",
        cmap="magma",
        colorbar=True,
        clabel="Detections",
        height=420,
        tools=["hover"],
        title="PedPy Position Density",
        xlabel="Ground x (m)",
        ylabel="Ground y (m)",
    ).opts(responsive=True, aspect="equal", show_grid=True)


def _summary_values(summary: pd.DataFrame, rows_read: int, skipped_rows: int) -> dict[str, str]:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=float)

    def number(value: object, digits: int = 0) -> str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.0
        if np.isnan(parsed):
            parsed = 0.0
        return f"{parsed:,.{digits}f}"

    return {
        "Serial rows": f"{rows_read:,}",
        "Skipped": f"{skipped_rows:,}",
        "Tracks": number(row.get("pedestrian_count", 0)),
        "Median speed": f"{number(row.get('median_speed_m_s', 0), 2)} m/s",
    }


class UsbDebugPanel:
    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root
        self.reader: SerialDebugReader | None = None

        port_options = serial_port_options()
        calibration_options = self._calibration_options()
        self.port = pn.widgets.Select(
            name="USB serial port",
            options=port_options,
            value=keep_or_first("COM5", port_options),
        )
        self.baud = pn.widgets.IntInput(name="Baud", value=115200, start=9600)
        self.calibration_path = pn.widgets.Select(
            name="Calibration JSON",
            options=calibration_options,
            value=keep_or_first("outputs/calibration.json", calibration_options),
        )
        self.max_rows = pn.widgets.IntInput(name="Live row buffer", value=2000, start=100)
        self.refresh_button = pn.widgets.Button(name="Refresh ports/files", height=38)
        self.connection_toggle_button = pn.widgets.Button(name="Connection settings", height=38)
        self.start_button = pn.widgets.Button(
            name="Start USB debug", button_type="primary", height=42
        )
        self.stop_button = pn.widgets.Button(name="Stop", button_type="default", height=42)
        self.status = pn.pane.HTML(
            _status_html(
                "USB only",
                "Open the USB serial port to enable debug mode. Wi-Fi cannot turn it on.",
            )
        )
        self.metrics = pn.pane.HTML(
            _metrics_html(
                {
                    "Serial rows": "0",
                    "Skipped": "0",
                    "Tracks": "0",
                    "Median speed": "0.00 m/s",
                }
            ),
            sizing_mode="stretch_width",
        )

        self.bbox_pane = pn.Column(
            _empty_plot("No live bounding boxes yet"),
            sizing_mode="stretch_width",
        )
        self.ground_pane = pn.Column(
            _empty_plot("No calibrated ground contact points yet"),
            sizing_mode="stretch_width",
        )
        self.grid_pane = pn.Column(
            _empty_plot("No PedPy grid data yet"),
            sizing_mode="stretch_width",
        )
        self.summary_table = pn.widgets.Tabulator(pd.DataFrame(), pagination="remote", page_size=8)
        self.tracks_table = pn.widgets.Tabulator(pd.DataFrame(), pagination="remote", page_size=10)
        self.grid_table = pn.widgets.Tabulator(pd.DataFrame(), pagination="remote", page_size=10)
        self.comments = pn.widgets.TextAreaInput(
            name="Firmware status",
            value="",
            disabled=True,
            height=120,
        )
        self.connection_section = pn.Column(
            self.baud,
            self.max_rows,
            visible=False,
            css_classes=["pedflow-inline-section"],
        )

        self.refresh_button.on_click(self._on_refresh)
        self.connection_toggle_button.on_click(self._on_toggle_connection_settings)
        self.start_button.on_click(self._on_start)
        self.stop_button.on_click(self._on_stop)
        self._periodic = pn.state.add_periodic_callback(self._refresh, period=500, start=False)

    def panel(self) -> pn.Column:
        controls = pn.Column(
            self.port,
            self.calibration_path,
            self.refresh_button,
            pn.Row(self.start_button, self.stop_button, css_classes=["pedflow-button-row"]),
            self.connection_toggle_button,
            self.connection_section,
            self.comments,
            css_classes=["pedflow-controls"],
            max_width=390,
            sizing_mode="stretch_width",
        )
        plots = pn.Tabs(
            ("Bounding boxes", self.bbox_pane),
            ("Ground contact", self.ground_pane),
            ("PedPy density", self.grid_pane),
            dynamic=True,
            css_classes=["pedflow-tabs"],
        )
        tables = pn.Tabs(
            ("Summary", self.summary_table),
            ("Tracks", self.tracks_table),
            ("Grid", self.grid_table),
            dynamic=True,
            css_classes=["pedflow-tabs"],
        )
        return pn.Column(
            pn.Row(
                controls,
                pn.Column(
                    self.status,
                    self.metrics,
                    plots,
                    tables,
                    css_classes=["pedflow-results"],
                ),
                css_classes=["pedflow-layout"],
            ),
            sizing_mode="stretch_width",
        )

    def _calibration_options(self) -> list[str]:
        return file_options(
            self.project_root,
            ("outputs/calibration*.json", "outputs/**/*.json"),
            ("outputs/calibration.json",),
        )

    def _on_refresh(self, _event: object) -> None:
        port_options = serial_port_options()
        calibration_options = self._calibration_options()
        self.port.options = port_options
        self.port.value = keep_or_first(str(self.port.value), port_options)
        self.calibration_path.options = calibration_options
        self.calibration_path.value = keep_or_first(
            str(self.calibration_path.value),
            calibration_options,
        )

    def _on_toggle_connection_settings(self, _event: object) -> None:
        self.connection_section.visible = not self.connection_section.visible
        self.connection_toggle_button.name = (
            "Hide connection settings" if self.connection_section.visible else "Connection settings"
        )

    def _on_start(self, _event: object) -> None:
        self._on_stop(_event)
        self.reader = SerialDebugReader(
            port=str(self.port.value).strip(),
            baud=int(self.baud.value),
            max_rows=int(self.max_rows.value),
        )
        self.reader.start()
        self._periodic.start()
        self.status.object = _status_html(
            "USB debug active",
            f"Reading {self.port.value} at {self.baud.value} baud. Debug mode is active via USB only.",
            kind="success",
        )

    def _on_stop(self, _event: object) -> None:
        if self.reader is not None:
            self.reader.stop()
        self._periodic.stop()
        self.status.object = _status_html(
            "Stopped",
            "USB debug capture is stopped, so firmware debug output is off.",
        )

    def _load_calibration(self) -> dict | None:
        path = resolve_path(self.project_root, str(self.calibration_path.value))
        if not path.exists():
            return None
        return load_calibration(path)

    def _refresh(self) -> None:
        if self.reader is None:
            return

        snapshot = self.reader.snapshot()
        calibration = self._load_calibration()
        result = analyze_live_debug_rows(snapshot.rows, calibration)
        self.metrics.object = _metrics_html(
            _summary_values(result.summary, snapshot.rows_read, snapshot.skipped_rows)
        )
        self.comments.value = "\n".join(snapshot.comments[-8:])
        self.summary_table.value = _rounded_frame(result.summary)
        self.tracks_table.value = _rounded_frame(result.tracks.tail(200))
        self.grid_table.value = _rounded_frame(result.grid)
        self.bbox_pane[:] = [_bbox_plot(result.bbox_overlay)]
        self.ground_pane[:] = [_ground_plot(result.ground, result.tracks)]
        self.grid_pane[:] = [_grid_plot(result.grid)]

        if snapshot.error and not snapshot.running:
            self.status.object = _status_html("USB debug stopped", snapshot.error, kind="danger")
        elif result.error:
            self.status.object = _status_html("Debug analysis issue", result.error, kind="danger")
        elif calibration is None:
            self.status.object = _status_html(
                "USB debug active",
                "Bounding boxes are live. Add calibration JSON for ground points and PedPy metrics.",
            )
        else:
            self.status.object = _status_html(
                "USB debug active",
                "Bounding boxes, ground contact points, and PedPy metrics are updating.",
                kind="success",
            )


def build_debug_app(project_root: Path = PROJECT_ROOT) -> pn.template.FastListTemplate:
    template = pn.template.FastListTemplate(
        title="Pedestrian Flow USB Debug",
        main=[UsbDebugPanel(project_root).panel()],
        accent_base_color="#256f78",
        header_background="#17212b",
    )
    return template
