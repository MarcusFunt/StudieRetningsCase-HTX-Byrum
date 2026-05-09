from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import detections_to_ground
from .metrics import (
    add_dwell_flags,
    estimate_speeds,
    grid_statistics,
    summarize_flow,
    track_summaries,
)
from .tracking import filter_short_tracks, link_detections


@dataclass(frozen=True)
class FlowAnalysisSettings:
    confidence_threshold: float = 0.6
    max_matching_speed_m_s: float = 4.5
    close_after_s: float = 1.0
    min_track_duration_s: float = 1.5
    min_detections: int = 4
    smoothing_alpha: float = 0.3
    speed_window_s: float = 0.75
    stop_speed_threshold_m_s: float = 0.2
    stop_duration_threshold_s: float = 2.0
    grid_size_m: float = 0.5


@dataclass(frozen=True)
class FlowAnalysisResult:
    detections_ground: pd.DataFrame
    tracks: pd.DataFrame
    summary: pd.DataFrame
    track_summaries: pd.DataFrame
    grid: pd.DataFrame


def run_flow_analysis(
    detections: pd.DataFrame,
    calibration: dict,
    settings: FlowAnalysisSettings | None = None,
    logger: Callable[[str], None] | None = None,
) -> FlowAnalysisResult:
    settings = settings or FlowAnalysisSettings()
    log = logger or (lambda _message: None)

    log("OpenCV: undistorting bbox foot points and applying ground homography")
    ground = detections_to_ground(
        detections,
        calibration,
        confidence_threshold=settings.confidence_threshold,
    )
    log(f"OpenCV: kept {len(ground):,} calibrated ground detection row(s)")
    log("Tracking: linking detections into pedestrian tracks")
    tracks = link_detections(
        ground,
        max_matching_speed_m_s=settings.max_matching_speed_m_s,
        close_after_s=settings.close_after_s,
        smoothing_alpha=settings.smoothing_alpha,
    )
    raw_track_count = tracks["track_id"].nunique() if not tracks.empty else 0
    log(f"Tracking: built {raw_track_count:,} raw track(s)")
    tracks = filter_short_tracks(
        tracks,
        min_duration_s=settings.min_track_duration_s,
        min_detections=settings.min_detections,
    )
    filtered_track_count = tracks["track_id"].nunique() if not tracks.empty else 0
    log(f"Tracking: kept {filtered_track_count:,} track(s) after duration/count filters")
    log("PedPy: estimating individual speeds")
    tracks = estimate_speeds(tracks, window_s=settings.speed_window_s)
    log("Metrics: marking dwell points")
    tracks = add_dwell_flags(
        tracks,
        stop_speed_threshold_m_s=settings.stop_speed_threshold_m_s,
        stop_duration_threshold_s=settings.stop_duration_threshold_s,
    )
    log("Metrics: summarizing flow and per-track geometry")
    summary = summarize_flow(tracks)
    summaries = track_summaries(tracks)
    log("PedPy: computing density, speed, and dwell grid profiles")
    grid = grid_statistics(tracks, grid_size_m=settings.grid_size_m)
    log(f"PedPy: produced {len(grid):,} occupied grid cell(s)")

    return FlowAnalysisResult(
        detections_ground=ground,
        tracks=tracks,
        summary=summary,
        track_summaries=summaries,
        grid=grid,
    )


def write_analysis_outputs(
    result: FlowAnalysisResult,
    output_dir: str | Path,
    settings: FlowAnalysisSettings | None = None,
    detections_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    started_utc: str | None = None,
    ended_utc: str | None = None,
    write_visual_qa: bool = True,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    output_files = {
        "detections_ground_csv": output / "detections_ground.csv",
        "tracks_csv": output / "tracks.csv",
        "track_summaries_csv": output / "track_summaries.csv",
        "summary_metrics_csv": output / "summary_metrics.csv",
        "grid_metrics_csv": output / "grid_metrics.csv",
    }
    result.detections_ground.to_csv(output_files["detections_ground_csv"], index=False)
    result.tracks.to_csv(output_files["tracks_csv"], index=False)
    result.track_summaries.to_csv(output_files["track_summaries_csv"], index=False)
    result.summary.to_csv(output_files["summary_metrics_csv"], index=False)
    result.grid.to_csv(output_files["grid_metrics_csv"], index=False)

    if write_visual_qa:
        output_files.update(write_visual_qa_outputs(result, output))

    manifest_path = output / "analysis_manifest.json"
    now_utc = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": 1,
        "session_type": "analysis",
        "start_utc": started_utc or now_utc,
        "end_utc": ended_utc or now_utc,
        "firmware_version": "unknown",
        "calibration_json_path": str(calibration_path) if calibration_path is not None else None,
        "detections_csv_path": str(detections_path) if detections_path is not None else None,
        "settings": asdict(settings) if settings is not None else None,
        "row_counts": {
            "detections_ground": len(result.detections_ground),
            "tracks": len(result.tracks),
            "track_summaries": len(result.track_summaries),
            "grid_cells": len(result.grid),
        },
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    output_files["analysis_manifest_json"] = manifest_path
    return output_files


def write_visual_qa_outputs(result: FlowAnalysisResult, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    plot_specs = {
        "paths": ("Pedestrian Paths / Desire Lines", _plot_paths),
        "density": ("Pedestrian Position Density", _plot_density),
        "speed": ("Median Speed Heatmap", _plot_speed),
        "dwell": ("Stop / Dwell Map", _plot_dwell),
        "bottleneck": ("Bottleneck Index", _plot_bottleneck),
    }
    for name, (title, plotter) in plot_specs.items():
        png_path = output / f"{name}_qa.png"
        html_path = output / f"{name}_qa.html"
        fig, empty_message = plotter(result)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _write_plot_html(html_path, title, png_path.name, empty_message)
        outputs[f"{name}_qa_png"] = png_path
        outputs[f"{name}_qa_html"] = html_path
    return outputs


def _empty_figure(title: str, message: str):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return fig, message


def _plot_paths(result: FlowAnalysisResult):
    tracks = result.tracks
    title = "Pedestrian Paths / Desire Lines"
    if tracks.empty:
        return _empty_figure(title, "No path data")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for track_id, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values("timestamp_ms", kind="stable")
        ax.plot(
            ordered["smooth_ground_x_m"],
            ordered["smooth_ground_y_m"],
            linewidth=1.6,
            alpha=0.75,
            label=str(track_id),
        )
    ax.set_title(title)
    ax.set_xlabel("Ground x (m)")
    ax.set_ylabel("Ground y (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    return fig, None


def _plot_density(result: FlowAnalysisResult):
    return _plot_grid_value(
        result.grid,
        value_col="detection_count",
        title="Pedestrian Position Density",
        color_label="Detections",
    )


def _plot_speed(result: FlowAnalysisResult):
    grid = (
        result.grid.dropna(subset=["median_speed_m_s"])
        if "median_speed_m_s" in result.grid
        else result.grid
    )
    return _plot_grid_value(
        grid,
        value_col="median_speed_m_s",
        title="Median Speed Heatmap",
        color_label="Median speed (m/s)",
    )


def _plot_dwell(result: FlowAnalysisResult):
    return _plot_grid_value(
        result.grid,
        value_col="dwell_points",
        title="Stop / Dwell Map",
        color_label="Dwell points",
    )


def _plot_bottleneck(result: FlowAnalysisResult):
    grid = result.grid.copy()
    if not grid.empty and {"detection_count", "median_speed_m_s"}.issubset(grid.columns):
        grid["bottleneck_index"] = grid["detection_count"] / (
            grid["median_speed_m_s"].fillna(0) + 0.2
        )
    return _plot_grid_value(
        grid,
        value_col="bottleneck_index",
        title="Bottleneck Index",
        color_label="Density / speed",
    )


def _plot_grid_value(
    grid: pd.DataFrame,
    value_col: str,
    title: str,
    color_label: str,
):
    if grid.empty or value_col not in grid.columns or grid[value_col].dropna().empty:
        return _empty_figure(title, f"No {title.lower()} data")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    values = grid[value_col].to_numpy(dtype=np.float64)
    scatter = ax.scatter(
        grid["grid_x_m"],
        grid["grid_y_m"],
        c=values,
        cmap="viridis",
        marker="s",
        s=220,
        edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax, label=color_label)
    ax.set_title(title)
    ax.set_xlabel("Ground x (m)")
    ax.set_ylabel("Ground y (m)")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")
    return fig, None


def _write_plot_html(
    html_path: Path,
    title: str,
    image_name: str,
    empty_message: str | None,
) -> None:
    body = f'<img src="{image_name}" alt="{title}" style="max-width:100%;height:auto;">'
    if empty_message:
        body = f"<p>{empty_message}</p>{body}"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
  <h1>{title}</h1>
  {body}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
