import numpy as np
import pandas as pd

from pedflow.metrics import add_dwell_flags, estimate_speeds, summarize_flow, track_summaries
from pedflow.tracking import filter_short_tracks, link_detections


def _ground_row(timestamp_ms, frame_id, detection_id, x, y):
    return {
        "timestamp_ms": timestamp_ms,
        "frame_id": frame_id,
        "detection_id": detection_id,
        "bbox_x": 0.0,
        "bbox_y": 0.0,
        "bbox_w": 0.0,
        "bbox_h": 0.0,
        "confidence": 0.9,
        "target": 0,
        "ground_x_m": x,
        "ground_y_m": y,
    }


def test_nearest_neighbor_tracking_and_speed():
    detections = pd.DataFrame(
        [
            _ground_row(0, 0, 0, 0.0, 0.0),
            _ground_row(500, 1, 0, 0.5, 0.0),
            _ground_row(1000, 2, 0, 1.0, 0.0),
            _ground_row(1500, 3, 0, 1.5, 0.0),
            _ground_row(2000, 4, 0, 2.0, 0.0),
        ]
    )

    linked = link_detections(detections, smoothing_alpha=1.0)
    filtered = filter_short_tracks(linked, min_duration_s=1.5, min_detections=4)
    speeds = estimate_speeds(filtered, window_s=0.75)

    assert speeds["track_id"].nunique() == 1
    assert len(speeds) == 5
    np.testing.assert_allclose(speeds["speed_m_s"].dropna().to_numpy(), [1.0, 1.0, 1.0, 1.0, 1.0])


def test_short_tracks_are_removed():
    detections = pd.DataFrame(
        [
            _ground_row(0, 0, 0, 0.0, 0.0),
            _ground_row(100, 1, 0, 0.1, 0.0),
        ]
    )

    linked = link_detections(detections, smoothing_alpha=1.0)
    filtered = filter_short_tracks(linked, min_duration_s=1.0, min_detections=4)

    assert filtered.empty


def test_dwell_flags_and_summary_metrics():
    detections = pd.DataFrame(
        [
            _ground_row(0, 0, 0, 1.0, 1.0),
            _ground_row(1000, 1, 0, 1.0, 1.0),
            _ground_row(2000, 2, 0, 1.0, 1.0),
            _ground_row(3000, 3, 0, 1.0, 1.0),
        ]
    )

    tracks = link_detections(detections, smoothing_alpha=1.0)
    tracks = filter_short_tracks(tracks, min_duration_s=1.0, min_detections=4)
    tracks = estimate_speeds(tracks, window_s=0.75)
    tracks = add_dwell_flags(tracks, stop_speed_threshold_m_s=0.2, stop_duration_threshold_s=2.0)
    summary = summarize_flow(tracks)
    per_track = track_summaries(tracks)

    assert tracks["is_dwell"].sum() == 4
    assert int(summary.loc[0, "pedestrian_count"]) == 1
    assert int(summary.loc[0, "dwell_points"]) == 4
    assert np.isnan(per_track.loc[0, "detour_ratio"])
