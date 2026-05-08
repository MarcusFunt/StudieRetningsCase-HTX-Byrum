from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class _TrackState:
    track_id: int
    last_timestamp_ms: float
    smooth_x: float
    smooth_y: float
    velocity_x_m_s: float = 0.0
    velocity_y_m_s: float = 0.0


def _predicted_position(track: _TrackState, timestamp_ms: float) -> tuple[float, float, float]:
    dt_s = max((timestamp_ms - track.last_timestamp_ms) / 1000.0, 0.0)
    return (
        track.smooth_x + track.velocity_x_m_s * dt_s,
        track.smooth_y + track.velocity_y_m_s * dt_s,
        dt_s,
    )


def _global_assignments(
    detections_xy: list[tuple[float, float]],
    active: dict[int, _TrackState],
    timestamp_ms: float,
    max_matching_speed_m_s: float,
    min_gate_m: float,
) -> dict[int, int]:
    if not detections_xy or not active:
        return {}

    track_ids = list(active)
    blocked_cost = 1e9
    costs = np.full((len(detections_xy), len(track_ids)), blocked_cost, dtype=np.float64)

    for row_position, (x, y) in enumerate(detections_xy):
        for column_position, track_id in enumerate(track_ids):
            predicted_x, predicted_y, dt_s = _predicted_position(active[track_id], timestamp_ms)
            gate_m = max(min_gate_m, max_matching_speed_m_s * dt_s)
            distance_m = float(np.hypot(x - predicted_x, y - predicted_y))
            if distance_m <= gate_m:
                costs[row_position, column_position] = distance_m

    row_indices, column_indices = linear_sum_assignment(costs)
    assignments: dict[int, int] = {}
    for row_position, column_position in zip(row_indices, column_indices, strict=True):
        if costs[row_position, column_position] >= blocked_cost:
            continue
        assignments[int(row_position)] = int(track_ids[int(column_position)])
    return assignments


def _required_ground_columns(detections: pd.DataFrame) -> None:
    required = {"timestamp_ms", "ground_x_m", "ground_y_m"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Missing required ground columns: {sorted(missing)}")


def link_detections(
    detections: pd.DataFrame,
    max_matching_speed_m_s: float = 4.5,
    close_after_s: float = 1.0,
    smoothing_alpha: float = 0.3,
    min_gate_m: float = 0.75,
    velocity_alpha: float = 0.5,
) -> pd.DataFrame:
    _required_ground_columns(detections)
    if not 0.0 <= smoothing_alpha <= 1.0:
        raise ValueError("smoothing_alpha must be between 0.0 and 1.0")
    if not 0.0 <= velocity_alpha <= 1.0:
        raise ValueError("velocity_alpha must be between 0.0 and 1.0")
    if max_matching_speed_m_s <= 0:
        raise ValueError("max_matching_speed_m_s must be greater than zero")
    if close_after_s <= 0:
        raise ValueError("close_after_s must be greater than zero")
    if min_gate_m < 0:
        raise ValueError("min_gate_m must be non-negative")

    if detections.empty:
        return detections.assign(
            track_id=pd.Series(dtype="int64"),
            smooth_ground_x_m=pd.Series(dtype="float64"),
            smooth_ground_y_m=pd.Series(dtype="float64"),
        )

    ordered = detections.reset_index(drop=True).sort_values(
        ["timestamp_ms", "frame_id", "detection_id"],
        kind="stable",
    )
    active: dict[int, _TrackState] = {}
    next_track_id = 1
    assignments: list[dict[str, float | int]] = []

    for timestamp_ms, group in ordered.groupby("timestamp_ms", sort=True):
        timestamp = float(timestamp_ms)
        stale_ids = [
            track_id
            for track_id, track in active.items()
            if timestamp - track.last_timestamp_ms > close_after_s * 1000.0
        ]
        for track_id in stale_ids:
            del active[track_id]

        detections_xy = [
            (float(detection["ground_x_m"]), float(detection["ground_y_m"]))
            for _, detection in group.iterrows()
        ]
        row_to_track = _global_assignments(
            detections_xy,
            active,
            timestamp,
            max_matching_speed_m_s,
            min_gate_m,
        )

        for row_position, (original_index, detection) in enumerate(group.iterrows()):
            x = float(detection["ground_x_m"])
            y = float(detection["ground_y_m"])

            if row_position in row_to_track:
                track_id = row_to_track[row_position]
                track = active[track_id]
                dt_s = max((timestamp - track.last_timestamp_ms) / 1000.0, 0.0)
                if dt_s > 0:
                    measured_vx = (x - track.smooth_x) / dt_s
                    measured_vy = (y - track.smooth_y) / dt_s
                    velocity_x = track.velocity_x_m_s + velocity_alpha * (
                        measured_vx - track.velocity_x_m_s
                    )
                    velocity_y = track.velocity_y_m_s + velocity_alpha * (
                        measured_vy - track.velocity_y_m_s
                    )
                else:
                    velocity_x = track.velocity_x_m_s
                    velocity_y = track.velocity_y_m_s
                smooth_x = track.smooth_x + smoothing_alpha * (x - track.smooth_x)
                smooth_y = track.smooth_y + smoothing_alpha * (y - track.smooth_y)
                active[track_id] = _TrackState(
                    track_id,
                    timestamp,
                    smooth_x,
                    smooth_y,
                    float(velocity_x),
                    float(velocity_y),
                )
            else:
                track_id = next_track_id
                next_track_id += 1
                smooth_x = x
                smooth_y = y
                active[track_id] = _TrackState(track_id, timestamp, smooth_x, smooth_y)

            assignments.append(
                {
                    "_source_index": int(original_index),
                    "track_id": int(track_id),
                    "smooth_ground_x_m": float(smooth_x),
                    "smooth_ground_y_m": float(smooth_y),
                }
            )

    assignment_frame = pd.DataFrame(assignments).set_index("_source_index")
    tracked = ordered.join(assignment_frame, how="left")
    tracked["track_id"] = tracked["track_id"].astype(int)
    return tracked.reset_index(drop=True)


def filter_short_tracks(
    tracks: pd.DataFrame,
    min_duration_s: float = 1.5,
    min_detections: int = 4,
) -> pd.DataFrame:
    if tracks.empty:
        return tracks.copy()
    if "track_id" not in tracks.columns:
        raise ValueError("tracks must contain a 'track_id' column")

    valid_track_ids: list[int] = []
    for track_id, group in tracks.groupby("track_id", sort=True):
        duration_s = (group["timestamp_ms"].max() - group["timestamp_ms"].min()) / 1000.0
        if duration_s >= min_duration_s and len(group) >= min_detections:
            valid_track_ids.append(int(track_id))

    filtered = tracks.loc[tracks["track_id"].isin(valid_track_ids)].copy()
    return filtered.reset_index(drop=True)
