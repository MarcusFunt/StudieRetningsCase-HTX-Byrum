import numpy as np
import pandas as pd

from pedflow.calibration import compute_ground_homography
from pedflow.geometry import bbox_foot_points, detections_to_ground


def test_bbox_foot_points_use_bottom_center():
    detections = pd.DataFrame(
        [
            {
                "bbox_x": 10.0,
                "bbox_y": 20.0,
                "bbox_w": 4.0,
                "bbox_h": 8.0,
            }
        ]
    )

    points = bbox_foot_points(detections)

    np.testing.assert_allclose(points, [[12.0, 28.0]])


def test_detections_are_undistorted_before_homography():
    detections = pd.DataFrame(
        [
            {
                "timestamp_ms": 0,
                "frame_id": 0,
                "detection_id": 0,
                "bbox_x": 10.0,
                "bbox_y": 20.0,
                "bbox_w": 4.0,
                "bbox_h": 8.0,
                "confidence": 0.9,
                "target": 0,
            }
        ]
    )
    calibration = {
        "K": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "H_image_to_ground": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]],
    }

    ground = detections_to_ground(detections, calibration, confidence_threshold=0.6)

    np.testing.assert_allclose(ground[["foot_x", "foot_y"]].to_numpy(), [[12.0, 28.0]])
    np.testing.assert_allclose(ground[["ground_x_m", "ground_y_m"]].to_numpy(), [[1.2, 2.8]])


def test_compute_ground_homography_reports_residuals():
    markers = pd.DataFrame(
        {
            "image_x": [0.0, 10.0, 0.0, 10.0, 5.0],
            "image_y": [0.0, 0.0, 10.0, 10.0, 5.0],
            "ground_x_m": [0.0, 1.0, 0.0, 1.0, 0.5],
            "ground_y_m": [0.0, 0.0, 1.0, 1.0, 0.5],
        }
    )
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros(5)

    result = compute_ground_homography(markers, camera_matrix, dist_coeffs)

    assert result["ground_marker_count"] == 5
    assert result["ground_marker_max_residual_m"] < 1e-8
    np.testing.assert_allclose(
        result["H_image_to_ground"],
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-8,
    )
