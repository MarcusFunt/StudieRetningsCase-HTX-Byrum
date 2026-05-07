from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .geometry import detections_to_ground
from .metrics import add_dwell_flags, estimate_speeds, grid_statistics, summarize_flow, track_summaries
from .plotting import write_standard_plots
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
) -> FlowAnalysisResult:
    settings = settings or FlowAnalysisSettings()

    ground = detections_to_ground(
        detections,
        calibration,
        confidence_threshold=settings.confidence_threshold,
    )
    tracks = link_detections(
        ground,
        max_matching_speed_m_s=settings.max_matching_speed_m_s,
        close_after_s=settings.close_after_s,
        smoothing_alpha=settings.smoothing_alpha,
    )
    tracks = filter_short_tracks(
        tracks,
        min_duration_s=settings.min_track_duration_s,
        min_detections=settings.min_detections,
    )
    tracks = estimate_speeds(tracks, window_s=settings.speed_window_s)
    tracks = add_dwell_flags(
        tracks,
        stop_speed_threshold_m_s=settings.stop_speed_threshold_m_s,
        stop_duration_threshold_s=settings.stop_duration_threshold_s,
    )

    return FlowAnalysisResult(
        detections_ground=ground,
        tracks=tracks,
        summary=summarize_flow(tracks),
        track_summaries=track_summaries(tracks),
        grid=grid_statistics(tracks, grid_size_m=settings.grid_size_m),
    )


def write_analysis_outputs(
    result: FlowAnalysisResult,
    output_dir: str | Path,
    grid_size_m: float = 0.5,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    result.detections_ground.to_csv(output / "detections_ground.csv", index=False)
    result.tracks.to_csv(output / "tracks.csv", index=False)
    result.track_summaries.to_csv(output / "track_summaries.csv", index=False)
    result.summary.to_csv(output / "summary_metrics.csv", index=False)
    result.grid.to_csv(output / "grid_metrics.csv", index=False)
    write_standard_plots(result.tracks, output, grid_size_m=grid_size_m)
