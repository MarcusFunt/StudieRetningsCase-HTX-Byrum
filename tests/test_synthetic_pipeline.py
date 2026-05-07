import numpy as np
import pandas as pd

from pedflow.geometry import detections_to_ground
from pedflow.metrics import add_dwell_flags, estimate_speeds, summarize_flow
from pedflow.plotting import write_standard_plots
from pedflow.tracking import filter_short_tracks, link_detections


def _bbox_from_ground(timestamp_ms, frame_id, detection_id, ground_x, ground_y):
    bbox_w = 4.0
    bbox_h = 8.0
    foot_x = ground_x * 10.0
    foot_y = ground_y * 10.0
    return {
        "timestamp_ms": timestamp_ms,
        "frame_id": frame_id,
        "detection_id": detection_id,
        "bbox_x": foot_x - bbox_w / 2.0,
        "bbox_y": foot_y - bbox_h,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "confidence": 0.95,
        "target": 0,
    }


def test_synthetic_end_to_end_pipeline_counts_two_people_and_writes_plots(tmp_path):
    rows = []
    for frame_id, timestamp_ms in enumerate([0, 500, 1000, 1500, 2000]):
        rows.append(_bbox_from_ground(timestamp_ms, frame_id, 0, frame_id * 0.5, 0.0))
        rows.append(_bbox_from_ground(timestamp_ms, frame_id, 1, 5.0, frame_id * 0.4))

    detections = pd.DataFrame(rows)
    calibration = {
        "K": np.eye(3).tolist(),
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "H_image_to_ground": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]],
    }

    ground = detections_to_ground(detections, calibration, confidence_threshold=0.6)
    tracks = link_detections(ground, smoothing_alpha=1.0)
    tracks = filter_short_tracks(tracks, min_duration_s=1.5, min_detections=4)
    tracks = estimate_speeds(tracks, window_s=0.75)
    tracks = add_dwell_flags(tracks)
    summary = summarize_flow(tracks)
    write_standard_plots(tracks, tmp_path, grid_size_m=0.5)

    assert len(ground) == 10
    assert tracks["track_id"].nunique() == 2
    assert int(summary.loc[0, "pedestrian_count"]) == 2
    assert summary.loc[0, "median_speed_m_s"] > 0.0
    assert (tmp_path / "paths_desire_lines.png").exists()
    assert (tmp_path / "position_heatmap.png").exists()
    assert (tmp_path / "speed_heatmap.png").exists()
    assert (tmp_path / "count_over_time.png").exists()
    assert (tmp_path / "dwell_map.png").exists()
    assert (tmp_path / "bottleneck_map.png").exists()
