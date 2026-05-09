from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pedpy import (
    FRAME_COL,
    ID_COL,
    SPEED_COL,
    X_COL,
    Y_COL,
    AxisAlignedMeasurementArea,
    DensityMethod,
    SpeedCalculation,
    SpeedMethod,
    TrajectoryData,
    compute_density_profile,
    compute_individual_speed,
    compute_speed_profile,
    get_grid_cells,
)

_PEDPY_FRAME_COL = "_pedpy_frame"
_TIMING_JITTER_WARNING_RATIO = 0.25


@dataclass(frozen=True)
class _PedPyTracks:
    tracks: pd.DataFrame
    trajectory: TrajectoryData | None


def _infer_frame_rate(timestamps_ms: pd.Series) -> float:
    unique_timestamps = np.sort(timestamps_ms.dropna().to_numpy(dtype=np.float64))
    unique_timestamps = np.unique(unique_timestamps)
    deltas_ms = np.diff(unique_timestamps)
    positive_deltas_ms = deltas_ms[deltas_ms > 0]
    if len(positive_deltas_ms) == 0:
        return 1.0
    return float(1000.0 / np.median(positive_deltas_ms))


def _positive_timestamp_deltas_ms(timestamps_ms: pd.Series) -> np.ndarray:
    unique_timestamps = np.sort(timestamps_ms.dropna().to_numpy(dtype=np.float64))
    unique_timestamps = np.unique(unique_timestamps)
    deltas_ms = np.diff(unique_timestamps)
    return deltas_ms[deltas_ms > 0]


def _warn_if_jittery_timestamps(timestamps_ms: pd.Series) -> None:
    positive_deltas_ms = _positive_timestamp_deltas_ms(timestamps_ms)
    if len(positive_deltas_ms) < 3:
        return

    median_delta_ms = float(np.median(positive_deltas_ms))
    if median_delta_ms <= 0:
        return

    max_relative_jitter = float(
        np.max(np.abs(positive_deltas_ms - median_delta_ms)) / median_delta_ms
    )
    if max_relative_jitter > _TIMING_JITTER_WARNING_RATIO:
        warnings.warn(
            "Timestamp intervals are jittery; PedPy frame mapping uses the median interval, "
            "so speed and grid metrics may be approximate.",
            RuntimeWarning,
            stacklevel=3,
        )


def _timestamp_frame_lookup(timestamps_ms: pd.Series) -> dict[float, int]:
    unique_timestamps = np.sort(
        np.unique(timestamps_ms.dropna().to_numpy(dtype=np.float64))
    )
    return {
        float(timestamp): frame_index
        for frame_index, timestamp in enumerate(unique_timestamps)
    }


def _prepare_pedpy_tracks(tracks: pd.DataFrame) -> _PedPyTracks:
    required = {"track_id", "timestamp_ms", "smooth_ground_x_m", "smooth_ground_y_m"}
    missing = required.difference(tracks.columns)
    if missing:
        raise ValueError(f"Missing required track columns: {sorted(missing)}")

    output = tracks.sort_values(["track_id", "timestamp_ms"], kind="stable").copy()
    if output.empty:
        return _PedPyTracks(output, None)

    _warn_if_jittery_timestamps(output["timestamp_ms"])
    frame_lookup = _timestamp_frame_lookup(output["timestamp_ms"])
    output[_PEDPY_FRAME_COL] = output["timestamp_ms"].map(frame_lookup)

    pedpy_data = pd.DataFrame(
        {
            ID_COL: output["track_id"],
            FRAME_COL: output[_PEDPY_FRAME_COL],
            X_COL: output["smooth_ground_x_m"],
            Y_COL: output["smooth_ground_y_m"],
        }
    )
    pedpy_data = pedpy_data.dropna(subset=[ID_COL, FRAME_COL, X_COL, Y_COL])
    if pedpy_data.empty:
        return _PedPyTracks(output, None)

    pedpy_data = pedpy_data.astype(
        {
            ID_COL: "int64",
            FRAME_COL: "int64",
            X_COL: "float64",
            Y_COL: "float64",
        }
    )
    pedpy_data = pedpy_data.drop_duplicates(subset=[ID_COL, FRAME_COL], keep="last")

    return _PedPyTracks(
        output,
        TrajectoryData(
            data=pedpy_data,
            frame_rate=_infer_frame_rate(output["timestamp_ms"]),
        ),
    )


def _frame_step_from_window(window_s: float, frame_rate: float) -> int:
    if window_s <= 0:
        raise ValueError("window_s must be greater than zero")
    return max(1, round(window_s * frame_rate / 2.0))


def _profile_measurement_area(tracks: pd.DataFrame, grid_size_m: float) -> AxisAlignedMeasurementArea:
    if grid_size_m <= 0:
        raise ValueError("grid_size_m must be greater than zero")

    x_min = math.floor(float(tracks["smooth_ground_x_m"].min()) / grid_size_m) * grid_size_m
    y_min = math.floor(float(tracks["smooth_ground_y_m"].min()) / grid_size_m) * grid_size_m
    x_max = math.floor(float(tracks["smooth_ground_x_m"].max()) / grid_size_m) * grid_size_m + grid_size_m
    y_max = math.floor(float(tracks["smooth_ground_y_m"].max()) / grid_size_m) * grid_size_m + grid_size_m
    if x_max <= x_min:
        x_max = x_min + grid_size_m
    if y_max <= y_min:
        y_max = y_min + grid_size_m
    return AxisAlignedMeasurementArea(x_min, y_min, x_max, y_max)


def _profile_counts(
    profile_data: pd.DataFrame,
    measurement_area: AxisAlignedMeasurementArea,
    grid_size_m: float,
    cells_count: int,
) -> np.ndarray:
    if profile_data.empty:
        return np.zeros(cells_count, dtype=np.int64)

    density_profiles = compute_density_profile(
        data=profile_data,
        axis_aligned_measurement_area=measurement_area,
        grid_size=grid_size_m,
        density_method=DensityMethod.CLASSIC,
    )
    if not density_profiles:
        return np.zeros(cells_count, dtype=np.int64)

    density_stack = np.asarray(density_profiles, dtype=np.float64).reshape(len(density_profiles), -1)
    counts = np.nansum(density_stack, axis=0) * (grid_size_m * grid_size_m)
    return np.rint(counts).astype(np.int64)


def _profile_medians(
    profile_data: pd.DataFrame,
    measurement_area: AxisAlignedMeasurementArea,
    grid_size_m: float,
    cells_count: int,
) -> np.ndarray:
    speed_profiles = compute_speed_profile(
        data=profile_data,
        axis_aligned_measurement_area=measurement_area,
        grid_size=grid_size_m,
        speed_method=SpeedMethod.MEAN,
        fill_value=np.nan,
    )
    if not speed_profiles:
        return np.full(cells_count, np.nan, dtype=np.float64)

    speed_stack = np.asarray(speed_profiles, dtype=np.float64).reshape(len(speed_profiles), -1)
    medians = np.full(speed_stack.shape[1], np.nan, dtype=np.float64)
    for cell_index in range(speed_stack.shape[1]):
        cell_values = speed_stack[:, cell_index]
        cell_values = cell_values[~np.isnan(cell_values)]
        if len(cell_values) > 0:
            medians[cell_index] = float(np.median(cell_values))
    return medians


def estimate_speeds(tracks: pd.DataFrame, window_s: float = 0.75) -> pd.DataFrame:
    prepared = _prepare_pedpy_tracks(tracks)
    output = prepared.tracks.drop(columns=["speed_m_s"], errors="ignore")
    if output.empty or prepared.trajectory is None:
        output["speed_m_s"] = np.nan
        return output.drop(columns=[_PEDPY_FRAME_COL], errors="ignore").reset_index(drop=True)

    pedpy_speed = compute_individual_speed(
        traj_data=prepared.trajectory,
        frame_step=_frame_step_from_window(window_s, prepared.trajectory.frame_rate),
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    ).rename(
        columns={
            ID_COL: "track_id",
            FRAME_COL: _PEDPY_FRAME_COL,
            SPEED_COL: "speed_m_s",
        }
    )

    output = output.merge(
        pedpy_speed[["track_id", _PEDPY_FRAME_COL, "speed_m_s"]],
        on=["track_id", _PEDPY_FRAME_COL],
        how="left",
    )
    return output.drop(columns=[_PEDPY_FRAME_COL], errors="ignore").reset_index(drop=True)


def add_dwell_flags(
    tracks: pd.DataFrame,
    stop_speed_threshold_m_s: float = 0.2,
    stop_duration_threshold_s: float = 2.0,
) -> pd.DataFrame:
    """Mark consecutive slow samples as dwell points.

    A point is slow when ``speed_m_s`` is below ``stop_speed_threshold_m_s``. A run of slow points is
    marked as dwell only when its elapsed timestamp duration reaches ``stop_duration_threshold_s``.
    """

    if "speed_m_s" not in tracks.columns:
        raise ValueError("tracks must contain speed_m_s; call estimate_speeds first")

    output = tracks.sort_values(["track_id", "timestamp_ms"], kind="stable").copy()
    output["is_slow"] = output["speed_m_s"].fillna(np.inf) < stop_speed_threshold_m_s
    output["is_dwell"] = False

    for _, group in output.groupby("track_id", sort=False):
        run_indices: list[int] = []
        run_start_ms: float | None = None

        for index, row in group.iterrows():
            if bool(row["is_slow"]):
                if not run_indices:
                    run_start_ms = float(row["timestamp_ms"])
                run_indices.append(index)
                continue

            if run_indices and run_start_ms is not None:
                duration_s = (float(output.loc[run_indices[-1], "timestamp_ms"]) - run_start_ms) / 1000.0
                if duration_s >= stop_duration_threshold_s:
                    output.loc[run_indices, "is_dwell"] = True
            run_indices = []
            run_start_ms = None

        if run_indices and run_start_ms is not None:
            duration_s = (float(output.loc[run_indices[-1], "timestamp_ms"]) - run_start_ms) / 1000.0
            if duration_s >= stop_duration_threshold_s:
                output.loc[run_indices, "is_dwell"] = True

    return output.reset_index(drop=True)


def _track_path_length(group: pd.DataFrame) -> float:
    xs = group["smooth_ground_x_m"].to_numpy(dtype=np.float64)
    ys = group["smooth_ground_y_m"].to_numpy(dtype=np.float64)
    if len(xs) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


def _track_heading_change_degrees(group: pd.DataFrame) -> float:
    xs = group["smooth_ground_x_m"].to_numpy(dtype=np.float64)
    ys = group["smooth_ground_y_m"].to_numpy(dtype=np.float64)
    if len(xs) < 3:
        return 0.0

    headings = np.arctan2(np.diff(ys), np.diff(xs))
    if len(headings) < 2:
        return 0.0

    changes = np.diff(headings)
    wrapped = (changes + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.mean(np.abs(wrapped))))


def track_summaries(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "duration_s",
                "detections",
                "path_length_m",
                "straight_line_m",
                "detour_ratio",
                "median_speed_m_s",
                "mean_heading_change_deg",
            ]
        )

    rows = []
    for track_id, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values("timestamp_ms", kind="stable")
        duration_s = (ordered["timestamp_ms"].max() - ordered["timestamp_ms"].min()) / 1000.0
        path_length_m = _track_path_length(ordered)
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        straight_line_m = float(
            math.hypot(
                float(last["smooth_ground_x_m"] - first["smooth_ground_x_m"]),
                float(last["smooth_ground_y_m"] - first["smooth_ground_y_m"]),
            )
        )
        detour_ratio = path_length_m / straight_line_m if straight_line_m > 0.25 else np.nan

        rows.append(
            {
                "track_id": int(track_id),
                "duration_s": float(duration_s),
                "detections": len(ordered),
                "path_length_m": float(path_length_m),
                "straight_line_m": float(straight_line_m),
                "detour_ratio": float(detour_ratio) if not np.isnan(detour_ratio) else np.nan,
                "median_speed_m_s": float(ordered["speed_m_s"].median())
                if "speed_m_s" in ordered.columns
                else np.nan,
                "mean_heading_change_deg": _track_heading_change_degrees(ordered),
            }
        )

    return pd.DataFrame(rows)


def summarize_flow(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        rows = {
            "pedestrian_count": 0,
            "duration_min": 0.0,
            "people_per_minute": 0.0,
            "people_per_hour": 0.0,
            "median_speed_m_s": np.nan,
            "median_detour_ratio": np.nan,
            "dwell_points": 0,
        }
        return pd.DataFrame([rows])

    duration_min = (
        float(tracks["timestamp_ms"].max()) - float(tracks["timestamp_ms"].min())
    ) / 1000.0 / 60.0
    duration_min = max(duration_min, 0.0)
    pedestrian_count = int(tracks["track_id"].nunique())
    people_per_minute = pedestrian_count / duration_min if duration_min > 0 else np.nan
    summaries = track_summaries(tracks)
    detour_ratios = summaries["detour_ratio"].dropna()

    return pd.DataFrame(
        [
            {
                "pedestrian_count": pedestrian_count,
                "duration_min": duration_min,
                "people_per_minute": people_per_minute,
                "people_per_hour": people_per_minute * 60.0,
                "median_speed_m_s": float(tracks["speed_m_s"].median())
                if "speed_m_s" in tracks.columns
                else np.nan,
                "median_detour_ratio": float(detour_ratios.median())
                if not detour_ratios.empty
                else np.nan,
                "dwell_points": int(tracks["is_dwell"].sum())
                if "is_dwell" in tracks.columns
                else 0,
            }
        ]
    )


def grid_statistics(
    tracks: pd.DataFrame,
    grid_size_m: float = 0.5,
) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame()

    prepared = _prepare_pedpy_tracks(tracks)
    if prepared.trajectory is None:
        return pd.DataFrame()

    measurement_area = _profile_measurement_area(prepared.tracks, grid_size_m)
    grid_cells, _, _ = get_grid_cells(
        axis_aligned_measurement_area=measurement_area,
        grid_size=grid_size_m,
    )
    cells_count = len(grid_cells)

    profile_data = prepared.trajectory.data
    detection_count = _profile_counts(
        profile_data,
        measurement_area,
        grid_size_m,
        cells_count,
    )

    stats = pd.DataFrame(
        {
            "grid_x_m": [float(cell.bounds[0]) for cell in grid_cells],
            "grid_y_m": [float(cell.bounds[1]) for cell in grid_cells],
            "detection_count": detection_count,
        }
    )

    occupied = stats["detection_count"] > 0

    if "speed_m_s" in prepared.tracks.columns:
        speed_data = prepared.tracks[
            ["track_id", _PEDPY_FRAME_COL, "speed_m_s"]
        ].rename(
            columns={
                "track_id": ID_COL,
                _PEDPY_FRAME_COL: FRAME_COL,
                "speed_m_s": SPEED_COL,
            }
        )
        speed_data = speed_data.drop_duplicates(subset=[ID_COL, FRAME_COL], keep="last")
        profile_speed_data = profile_data.merge(speed_data, on=[ID_COL, FRAME_COL], how="left")
        stats["median_speed_m_s"] = _profile_medians(
            profile_speed_data,
            measurement_area,
            grid_size_m,
            cells_count,
        )
        occupied = occupied | stats["median_speed_m_s"].notna()

    if "is_dwell" in prepared.tracks.columns:
        dwell_keys = prepared.tracks.loc[
            prepared.tracks["is_dwell"].fillna(False),
            ["track_id", _PEDPY_FRAME_COL],
        ].rename(
            columns={
                "track_id": ID_COL,
                _PEDPY_FRAME_COL: FRAME_COL,
            }
        )
        dwell_keys = dwell_keys.drop_duplicates(subset=[ID_COL, FRAME_COL], keep="last")
        dwell_data = profile_data.merge(dwell_keys, on=[ID_COL, FRAME_COL], how="inner")
        stats["dwell_points"] = _profile_counts(
            dwell_data,
            measurement_area,
            grid_size_m,
            cells_count,
        )
        occupied = occupied | (stats["dwell_points"] > 0)

    return (
        stats.loc[occupied]
        .sort_values(["grid_x_m", "grid_y_m"], kind="stable")
        .reset_index(drop=True)
    )
