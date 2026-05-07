from __future__ import annotations

import math

import numpy as np
import pandas as pd


def estimate_speeds(tracks: pd.DataFrame, window_s: float = 0.75) -> pd.DataFrame:
    required = {"track_id", "timestamp_ms", "smooth_ground_x_m", "smooth_ground_y_m"}
    missing = required.difference(tracks.columns)
    if missing:
        raise ValueError(f"Missing required track columns: {sorted(missing)}")

    output = tracks.sort_values(["track_id", "timestamp_ms"], kind="stable").copy()
    output["speed_m_s"] = np.nan

    for _, group in output.groupby("track_id", sort=False):
        indices = group.index.to_numpy()
        times_s = group["timestamp_ms"].to_numpy(dtype=np.float64) / 1000.0
        xs = group["smooth_ground_x_m"].to_numpy(dtype=np.float64)
        ys = group["smooth_ground_y_m"].to_numpy(dtype=np.float64)

        for position in range(1, len(group)):
            target_time = times_s[position] - window_s
            previous_position = int(np.searchsorted(times_s, target_time, side="left"))
            if previous_position >= position:
                previous_position = position - 1

            dt_s = times_s[position] - times_s[previous_position]
            if dt_s <= 0:
                continue

            distance_m = math.hypot(
                xs[position] - xs[previous_position],
                ys[position] - ys[previous_position],
            )
            output.loc[indices[position], "speed_m_s"] = distance_m / dt_s

    return output.reset_index(drop=True)


def add_dwell_flags(
    tracks: pd.DataFrame,
    stop_speed_threshold_m_s: float = 0.2,
    stop_duration_threshold_s: float = 2.0,
) -> pd.DataFrame:
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
                "detections": int(len(ordered)),
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
    people_per_minute = pedestrian_count / duration_min if duration_min > 0 else float(pedestrian_count)
    summaries = track_summaries(tracks)

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
                "median_detour_ratio": float(summaries["detour_ratio"].median()),
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

    output = tracks.copy()
    output["grid_x_m"] = np.floor(output["smooth_ground_x_m"] / grid_size_m) * grid_size_m
    output["grid_y_m"] = np.floor(output["smooth_ground_y_m"] / grid_size_m) * grid_size_m

    aggregations = {"track_id": "count"}
    if "speed_m_s" in output.columns:
        aggregations["speed_m_s"] = "median"
    if "is_dwell" in output.columns:
        aggregations["is_dwell"] = "sum"

    stats = output.groupby(["grid_x_m", "grid_y_m"], as_index=False).agg(aggregations)
    stats = stats.rename(
        columns={
            "track_id": "detection_count",
            "speed_m_s": "median_speed_m_s",
            "is_dwell": "dwell_points",
        }
    )
    return stats
