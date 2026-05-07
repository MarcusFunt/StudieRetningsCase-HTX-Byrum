from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import grid_statistics


def _finish_axis(ax: plt.Axes, title: str, x_label: str = "Ground x (m)", y_label: str = "Ground y (m)") -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paths(tracks: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values("timestamp_ms", kind="stable")
        ax.plot(
            ordered["smooth_ground_x_m"],
            ordered["smooth_ground_y_m"],
            linewidth=1.2,
            alpha=0.7,
        )
    _finish_axis(ax, "Pedestrian Paths / Desire Lines")
    _save_figure(fig, output_path)


def plot_count_over_time(
    tracks: pd.DataFrame,
    output_path: str | Path,
    bin_seconds: int = 60,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    first_seen = tracks.groupby("track_id", as_index=False)["timestamp_ms"].min()
    if first_seen.empty:
        counts = pd.Series(dtype=int)
    else:
        elapsed_s = (first_seen["timestamp_ms"] - first_seen["timestamp_ms"].min()) / 1000.0
        bins = np.floor(elapsed_s / bin_seconds).astype(int)
        counts = bins.value_counts().sort_index()

    ax.bar(counts.index * bin_seconds / 60.0, counts.to_numpy(), width=bin_seconds / 60.0 * 0.9)
    ax.set_title("Pedestrian Count Over Time")
    ax.set_xlabel("Minutes since start")
    ax.set_ylabel("New tracks")
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, output_path)


def plot_grid_metric(
    grid: pd.DataFrame,
    value_col: str,
    output_path: str | Path,
    title: str,
    color_label: str,
    grid_size_m: float = 0.5,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    if not grid.empty and value_col in grid.columns:
        points = ax.scatter(
            grid["grid_x_m"] + grid_size_m / 2.0,
            grid["grid_y_m"] + grid_size_m / 2.0,
            c=grid[value_col],
            marker="s",
            s=420,
            cmap=cmap,
            alpha=0.85,
            edgecolors="none",
        )
        cbar = fig.colorbar(points, ax=ax)
        cbar.set_label(color_label)

    _finish_axis(ax, title)
    _save_figure(fig, output_path)


def plot_position_heatmap(tracks: pd.DataFrame, output_path: str | Path, grid_size_m: float = 0.5) -> None:
    grid = grid_statistics(tracks, grid_size_m=grid_size_m)
    plot_grid_metric(
        grid,
        "detection_count",
        output_path,
        "Pedestrian Position Heatmap",
        "Detections",
        grid_size_m=grid_size_m,
        cmap="magma",
    )


def plot_speed_heatmap(tracks: pd.DataFrame, output_path: str | Path, grid_size_m: float = 0.5) -> None:
    grid = grid_statistics(tracks, grid_size_m=grid_size_m)
    plot_grid_metric(
        grid.dropna(subset=["median_speed_m_s"]) if "median_speed_m_s" in grid.columns else grid,
        "median_speed_m_s",
        output_path,
        "Median Speed Heatmap",
        "Median speed (m/s)",
        grid_size_m=grid_size_m,
        cmap="plasma",
    )


def plot_dwell_map(tracks: pd.DataFrame, output_path: str | Path, grid_size_m: float = 0.5) -> None:
    grid = grid_statistics(tracks, grid_size_m=grid_size_m)
    plot_grid_metric(
        grid,
        "dwell_points",
        output_path,
        "Stop / Dwell Map",
        "Dwell points",
        grid_size_m=grid_size_m,
        cmap="cividis",
    )


def plot_bottleneck_map(tracks: pd.DataFrame, output_path: str | Path, grid_size_m: float = 0.5) -> None:
    grid = grid_statistics(tracks, grid_size_m=grid_size_m)
    if not grid.empty and {"detection_count", "median_speed_m_s"}.issubset(grid.columns):
        grid = grid.copy()
        grid["bottleneck_index"] = grid["detection_count"] / (grid["median_speed_m_s"].fillna(0) + 0.2)
    plot_grid_metric(
        grid,
        "bottleneck_index",
        output_path,
        "Bottleneck Index",
        "Density / speed",
        grid_size_m=grid_size_m,
        cmap="inferno",
    )


def write_standard_plots(
    tracks: pd.DataFrame,
    output_dir: str | Path,
    grid_size_m: float = 0.5,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plot_paths(tracks, output / "paths_desire_lines.png")
    plot_position_heatmap(tracks, output / "position_heatmap.png", grid_size_m=grid_size_m)
    plot_speed_heatmap(tracks, output / "speed_heatmap.png", grid_size_m=grid_size_m)
    plot_count_over_time(tracks, output / "count_over_time.png")
    plot_dwell_map(tracks, output / "dwell_map.png", grid_size_m=grid_size_m)
    plot_bottleneck_map(tracks, output / "bottleneck_map.png", grid_size_m=grid_size_m)
