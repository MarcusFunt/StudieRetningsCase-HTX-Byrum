import json

import numpy as np
import pytest

from pedflow.analysis import FlowAnalysisSettings, run_flow_analysis, write_analysis_outputs
from pedflow.geometry import apply_homography
from pedflow.manual import (
    build_manual_calibration,
    clip_manual_paths_to_measurement,
    manual_measurement_grid_lines,
    manual_path_summaries,
    manual_paths_to_detections,
    run_manual_analysis,
)


def _manual_settings(close_after_s: float = 1.0) -> FlowAnalysisSettings:
    return FlowAnalysisSettings(
        confidence_threshold=0.5,
        max_matching_speed_m_s=50.0,
        close_after_s=close_after_s,
        min_track_duration_s=0.0,
        min_detections=2,
        smoothing_alpha=1.0,
    )


def test_manual_calibration_maps_four_road_corners_to_ground_rectangle():
    corners = [(10.0, 20.0), (110.0, 20.0), (110.0, 70.0), (10.0, 70.0)]

    calibration = build_manual_calibration(corners, length_m=10.0, width_m=5.0, image_size=(160, 90))

    projected = apply_homography(np.asarray(corners, dtype=np.float64), calibration["H_image_to_ground"])
    np.testing.assert_allclose(
        projected,
        [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
        atol=1e-8,
    )
    assert calibration["K"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert calibration["dist"] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert calibration["calibration_quality"]["status"] == "warn"
    assert calibration["homography_quality"]["status"] == "pass"


def test_manual_calibration_accepts_column_pair_corner_order():
    clicked = [(10.0, 20.0), (10.0, 70.0), (110.0, 20.0), (110.0, 70.0)]

    calibration = build_manual_calibration(clicked, length_m=10.0, width_m=5.0, image_size=(160, 90))

    ordered = np.asarray([clicked[index - 1] for index in [1, 3, 4, 2]], dtype=np.float64)
    projected = apply_homography(ordered, calibration["H_image_to_ground"])
    np.testing.assert_allclose(
        projected,
        [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
        atol=1e-8,
    )
    assert calibration["manual_corner_order"] == "paired_width_edges"
    assert calibration["manual_corner_order_source_indices"] == [1, 3, 4, 2]


def test_manual_calibration_accepts_row_pair_corner_order():
    clicked = [(10.0, 20.0), (110.0, 20.0), (10.0, 70.0), (110.0, 70.0)]

    calibration = build_manual_calibration(clicked, length_m=10.0, width_m=5.0, image_size=(160, 90))

    ordered = np.asarray([clicked[index - 1] for index in [1, 2, 4, 3]], dtype=np.float64)
    projected = apply_homography(ordered, calibration["H_image_to_ground"])
    np.testing.assert_allclose(
        projected,
        [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
        atol=1e-8,
    )
    assert calibration["manual_corner_order"] == "paired_length_edges"
    assert calibration["manual_corner_order_source_indices"] == [1, 2, 4, 3]


@pytest.mark.parametrize(
    ("corners", "length_m", "width_m", "match"),
    [
        ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], 1.0, 1.0, "exactly four"),
        ([(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)], 1.0, 1.0, "unique"),
        ([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], 1.0, 1.0, "non-degenerate"),
        ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 0.0, 1.0, "greater than zero"),
    ],
)
def test_manual_calibration_rejects_invalid_input(corners, length_m, width_m, match):
    with pytest.raises(ValueError, match=match):
        build_manual_calibration(corners, length_m=length_m, width_m=width_m, image_size=(100, 100))


def test_manual_paths_become_monotonic_synthetic_detection_rows():
    paths = [
        [(10.0, 50.0, 5.0), (20.0, 50.0, 5.2), (20.0, 60.0, 5.2)],
        [(30.0, 60.0, 0.0), (40.0, 60.0, 1.0)],
    ]

    detections = manual_paths_to_detections(paths, bbox_width_px=24.0, bbox_height_px=64.0)

    assert detections["timestamp_ms"].tolist() == [0, 200, 300, 2300, 3300]
    assert detections["timestamp_ms"].is_monotonic_increasing
    np.testing.assert_allclose(
        detections["bbox_x"].to_numpy() + detections["bbox_w"].to_numpy() / 2.0,
        [10.0, 20.0, 20.0, 30.0, 40.0],
    )
    np.testing.assert_allclose(
        detections["bbox_y"].to_numpy() + detections["bbox_h"].to_numpy(),
        [50.0, 50.0, 60.0, 60.0, 60.0],
    )
    assert detections["confidence"].tolist() == [1.0] * 5


def test_manual_paths_can_preserve_absolute_synthetic_timestamps():
    paths = [
        [(10.0, 50.0, 20.0), (20.0, 50.0, 21.0)],
        [(30.0, 60.0, 5.0), (40.0, 60.0, 6.0)],
    ]

    detections = manual_paths_to_detections(paths, preserve_timestamps=True)

    assert detections["timestamp_ms"].tolist() == [5000, 6000, 20000, 21000]
    assert detections["frame_id"].tolist() == [0, 1, 2, 3]


def test_manual_grid_lines_project_to_image_space():
    calibration = build_manual_calibration(
        [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)],
        length_m=10.0,
        width_m=5.0,
        image_size=(100, 50),
    )

    grid = manual_measurement_grid_lines(calibration, grid_size_m=5.0)

    assert grid.iloc[0]["kind"] == "boundary"
    assert len(grid) == 6
    assert grid["xs"].map(len).min() >= 2


def test_manual_paths_are_clipped_to_measurement_grid():
    calibration = build_manual_calibration(
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        length_m=10.0,
        width_m=10.0,
        image_size=(100, 100),
    )
    paths = [[(-50.0, 50.0, 0.0), (50.0, 50.0, 5.0), (150.0, 50.0, 10.0)]]

    clipped = clip_manual_paths_to_measurement(paths, calibration)
    detections = manual_paths_to_detections(clipped, preserve_timestamps=True)

    assert len(clipped) == 1
    assert clipped[0][0]["x"] == pytest.approx(0.0)
    assert clipped[0][-1]["x"] == pytest.approx(100.0)
    assert detections["timestamp_ms"].tolist() == [2500, 5000, 7500]


def test_manual_path_gaps_create_separate_tracks():
    calibration = build_manual_calibration(
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        length_m=10.0,
        width_m=10.0,
        image_size=(100, 100),
    )
    detections = manual_paths_to_detections(
        [
            [(10.0, 10.0, 0.0), (20.0, 10.0, 1.0)],
            [(10.0, 20.0, 0.0), (20.0, 20.0, 1.0)],
        ],
        path_gap_s=1.2,
    )

    result = run_flow_analysis(detections, calibration, _manual_settings(close_after_s=1.1))

    assert result.tracks["track_id"].nunique() == 2


def test_manual_paths_report_drag_duration_length_and_speed():
    calibration = build_manual_calibration(
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        length_m=10.0,
        width_m=10.0,
        image_size=(100, 100),
    )
    paths = [[(0.0, 0.0, 0.0), (30.0, 40.0, 2.0)]]

    summary = manual_path_summaries(paths, calibration)

    assert summary.loc[0, "duration_s"] == 2.0
    assert summary.loc[0, "path_length_m"] == pytest.approx(5.0)
    assert summary.loc[0, "mean_speed_m_s"] == pytest.approx(2.5)


def test_manual_analysis_can_write_standard_outputs(tmp_path):
    corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    calibration = build_manual_calibration(corners, length_m=10.0, width_m=10.0, image_size=(100, 100))
    paths = [
        [(10.0, 10.0, 0.0), (20.0, 10.0, 1.0), (30.0, 10.0, 2.0), (40.0, 10.0, 3.0)]
    ]
    detections = manual_paths_to_detections(paths)
    settings = _manual_settings()

    result = run_flow_analysis(detections, calibration, settings)
    outputs = write_analysis_outputs(
        result,
        tmp_path,
        settings=settings,
        detections_path=tmp_path / "manual_detections.csv",
        calibration_path=tmp_path / "manual_calibration.json",
        started_utc="2026-05-15T10:00:00+00:00",
        ended_utc="2026-05-15T10:00:03+00:00",
    )

    assert int(result.summary.loc[0, "pedestrian_count"]) == 1
    assert not result.tracks.empty
    assert outputs["analysis_manifest_json"] == tmp_path / "analysis_manifest.json"
    assert (tmp_path / "paths_qa.png").exists()
    assert (tmp_path / "density_qa.html").exists()
    manifest = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["detections_csv_path"].endswith("manual_detections.csv")
    assert manifest["calibration_json_path"].endswith("manual_calibration.json")


def test_run_manual_analysis_wrapper_uses_image_metadata():
    result = run_manual_analysis(
        image_metadata={"image_width": 100, "image_height": 100},
        corners_px=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        paths=[[(10.0, 10.0, 0.0), (20.0, 10.0, 1.0)]],
        length_m=10.0,
        width_m=10.0,
        settings=_manual_settings(),
    )

    assert int(result.summary.loc[0, "pedestrian_count"]) == 1
