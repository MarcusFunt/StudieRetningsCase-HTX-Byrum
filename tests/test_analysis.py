import json

import numpy as np
import pandas as pd

from pedflow.analysis import FlowAnalysisSettings, run_flow_analysis, write_analysis_outputs


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


def test_run_flow_analysis_returns_tables_and_writes_outputs(tmp_path):
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

    result = run_flow_analysis(
        detections,
        calibration,
        FlowAnalysisSettings(smoothing_alpha=1.0),
    )
    detections_path = tmp_path / "input.csv"
    calibration_path = tmp_path / "calibration.json"
    settings = FlowAnalysisSettings(smoothing_alpha=1.0)
    outputs = write_analysis_outputs(
        result,
        tmp_path,
        settings=settings,
        detections_path=detections_path,
        calibration_path=calibration_path,
        started_utc="2026-05-09T10:00:00+00:00",
        ended_utc="2026-05-09T10:01:00+00:00",
    )

    assert int(result.summary.loc[0, "pedestrian_count"]) == 2
    assert result.track_summaries["track_id"].tolist() == [1, 2]
    assert not result.grid.empty
    assert (tmp_path / "detections_ground.csv").exists()
    assert (tmp_path / "tracks.csv").exists()
    assert (tmp_path / "track_summaries.csv").exists()
    assert (tmp_path / "summary_metrics.csv").exists()
    assert (tmp_path / "grid_metrics.csv").exists()
    assert (tmp_path / "paths_qa.png").exists()
    assert (tmp_path / "density_qa.html").exists()
    assert len(list(tmp_path.glob("*_qa.png"))) == 5
    assert len(list(tmp_path.glob("*_qa.html"))) == 5

    manifest = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert outputs["analysis_manifest_json"] == tmp_path / "analysis_manifest.json"
    assert manifest["schema_version"] == 1
    assert manifest["session_type"] == "analysis"
    assert manifest["start_utc"] == "2026-05-09T10:00:00+00:00"
    assert manifest["end_utc"] == "2026-05-09T10:01:00+00:00"
    assert manifest["calibration_json_path"] == str(calibration_path)
    assert manifest["detections_csv_path"] == str(detections_path)
    assert manifest["settings"]["smoothing_alpha"] == settings.smoothing_alpha
    assert manifest["output_files"]["paths_qa_png"].endswith("paths_qa.png")
