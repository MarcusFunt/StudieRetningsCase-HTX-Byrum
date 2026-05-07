from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class _TrackState:
    track_id: int
    last_timestamp_ms: float
    smooth_x: float
    smooth_y: float


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
) -> pd.DataFrame:
    _required_ground_columns(detections)
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

        candidates: list[tuple[float, int, int]] = []
        for row_position, (_, detection) in enumerate(group.iterrows()):
            x = float(detection["ground_x_m"])
            y = float(detection["ground_y_m"])
            for track_id, track in active.items():
                dt_s = max((timestamp - track.last_timestamp_ms) / 1000.0, 0.0)
                gate_m = max(min_gate_m, max_matching_speed_m_s * dt_s)
                distance_m = float(np.hypot(x - track.smooth_x, y - track.smooth_y))
                if distance_m <= gate_m:
                    candidates.append((distance_m, row_position, track_id))

        assigned_rows: set[int] = set()
        assigned_tracks: set[int] = set()
        row_to_track: dict[int, int] = {}

        for _, row_position, track_id in sorted(candidates, key=lambda item: item[0]):
            if row_position in assigned_rows or track_id in assigned_tracks:
                continue
            row_to_track[row_position] = track_id
            assigned_rows.add(row_position)
            assigned_tracks.add(track_id)

        for row_position, (original_index, detection) in enumerate(group.iterrows()):
            x = float(detection["ground_x_m"])
            y = float(detection["ground_y_m"])

            if row_position in row_to_track:
                track_id = row_to_track[row_position]
                track = active[track_id]
                smooth_x = track.smooth_x + smoothing_alpha * (x - track.smooth_x)
                smooth_y = track.smooth_y + smoothing_alpha * (y - track.smooth_y)
                active[track_id] = _TrackState(track_id, timestamp, smooth_x, smooth_y)
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
