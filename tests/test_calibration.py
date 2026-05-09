import json

import cv2
import numpy as np
import pytest

import pedflow.calibration as calibration_module
from pedflow.calibration import (
    _MAX_SAFE_BASENAME_LENGTH,
    _reprojection_rms_error,
    _safe_output_basename,
    calibrate_camera_from_charuco,
    calibrate_camera_from_checkerboard,
    calibration_quality_report,
    generate_charuco_board,
    homography_quality_report,
    intrinsics_quality_report,
    read_marker_csv,
)


def test_checkerboard_calibration_ignores_unusable_mixed_image_resolutions(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    cv2.imwrite(str(first), np.zeros((20, 20, 3), dtype=np.uint8))
    cv2.imwrite(str(second), np.zeros((30, 30, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="No usable checkerboard"):
        calibrate_camera_from_checkerboard(
            [first, second],
            pattern_size=(3, 3),
            square_size_m=0.1,
        )


def test_checkerboard_calibration_rejects_mixed_usable_image_resolutions(tmp_path, monkeypatch):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    cv2.imwrite(str(first), np.zeros((20, 20, 3), dtype=np.uint8))
    cv2.imwrite(str(second), np.zeros((30, 30, 3), dtype=np.uint8))

    corners = np.zeros((9, 1, 2), dtype=np.float32)
    monkeypatch.setattr(cv2, "findChessboardCorners", lambda _gray, _pattern_size: (True, corners))
    monkeypatch.setattr(cv2, "cornerSubPix", lambda _gray, found, *_args: found)

    with pytest.raises(ValueError, match=r"usable checkerboard.*same resolution"):
        calibrate_camera_from_checkerboard(
            [first, second],
            pattern_size=(3, 3),
            square_size_m=0.1,
        )


def test_charuco_calibration_ignores_unusable_image_resolution(tmp_path, monkeypatch):
    metadata_path = tmp_path / "charuco_board.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dictionary": "DICT_5X5_100",
                "squares_x": 5,
                "squares_y": 4,
                "square_length_m": 0.03,
                "marker_length_m": 0.022,
            }
        ),
        encoding="utf-8",
    )
    image_paths = [tmp_path / "skipped.png"] + [
        tmp_path / f"usable_{index}.png" for index in range(3)
    ]
    cv2.imwrite(str(image_paths[0]), np.zeros((20, 20, 3), dtype=np.uint8))
    for path in image_paths[1:]:
        cv2.imwrite(str(path), np.zeros((30, 30, 3), dtype=np.uint8))

    object_points = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    image_points = np.array(
        [
            [[0.0, 0.0]],
            [[10.0, 0.0]],
            [[0.0, 10.0]],
            [[10.0, 10.0]],
        ],
        dtype=np.float32,
    )

    def fake_detect(image, _board, min_corners=8):
        if image.shape[:2] == (20, 20):
            raise ValueError(f"Detected fewer than {min_corners} ChArUco corners")
        return object_points, image_points, 4

    def fake_calibrate_camera(obj_points, _img_points, _image_size, *_args):
        rvecs = [np.zeros((3, 1), dtype=np.float64) for _ in obj_points]
        tvecs = [np.zeros((3, 1), dtype=np.float64) for _ in obj_points]
        return 0.0, np.eye(3), np.zeros((5, 1)), rvecs, tvecs

    monkeypatch.setattr(calibration_module, "detect_charuco_image_points", fake_detect)
    monkeypatch.setattr(cv2, "calibrateCamera", fake_calibrate_camera)
    monkeypatch.setattr(calibration_module, "_reprojection_rms_error", lambda *_args: 0.0)

    result = calibrate_camera_from_charuco(image_paths, metadata_path, min_corners=4)

    assert result["image_width"] == 30
    assert result["skipped_image_count"] == 1
    assert len(result["used_images"]) == 3
    assert result["intrinsics_quality"]["status"] == "warn"


def test_reprojection_error_reports_rms_pixel_error():
    object_points = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    camera_matrix = np.array(
        [
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [2.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    residuals = np.array(
        [
            [[3.0, 4.0]],
            [[0.0, 5.0]],
            [[5.0, 0.0]],
            [[4.0, 3.0]],
        ],
        dtype=np.float64,
    )

    error = _reprojection_rms_error(
        object_points,
        projected + residuals,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    assert error == pytest.approx(5.0)


def test_calibration_quality_reports_include_status_and_issues():
    intrinsics = {
        "K": np.eye(3).tolist(),
        "rms_reprojection_error_px": 0.45,
        "per_image_reprojection_error_px": [0.4, 0.5, 0.45],
        "used_images": ["a.png", "b.png", "c.png"],
        "skipped_image_count": 0,
    }
    homography = {
        "H_image_to_ground": np.eye(3).tolist(),
        "ground_marker_count": 4,
        "ground_marker_mean_residual_m": 0.02,
        "ground_marker_max_residual_m": 0.05,
        "ground_marker_inliers": [True, True, True, True],
    }

    assert intrinsics_quality_report(intrinsics)["status"] == "pass"
    assert homography_quality_report(homography)["status"] == "pass"
    merged = intrinsics | homography
    assert calibration_quality_report(merged)["status"] == "pass"

    intrinsics["skipped_image_count"] = 2
    report = intrinsics_quality_report(intrinsics)
    assert report["status"] == "warn"
    assert "2 image(s) were skipped." in report["issues"]


@pytest.mark.parametrize("basename", ["../escape", "nested/name", r"nested\name", "bad name"])
def test_generate_charuco_board_rejects_path_like_basename(tmp_path, basename):
    with pytest.raises(ValueError, match="basename"):
        generate_charuco_board(tmp_path, basename=basename)


def test_charuco_board_basename_is_deterministically_truncated():
    basename = "charuco_" + ("a" * 200)

    shortened = _safe_output_basename(basename)

    assert len(shortened) == _MAX_SAFE_BASENAME_LENGTH
    assert shortened == _safe_output_basename(basename)
    assert shortened != _safe_output_basename(f"{basename}b")
    assert shortened.startswith("charuco_")


def test_marker_csv_rejects_non_finite_values(tmp_path):
    markers_path = tmp_path / "markers.csv"
    markers_path.write_text(
        "\n".join(
            [
                "image_x,image_y,ground_x_m,ground_y_m",
                "120.0,200.0,0.0,0.0",
                "200.0,200.0,1.0,0.0",
                "120.0,140.0,0.0,1.0",
                "200.0,140.0,inf,1.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite numeric"):
        read_marker_csv(markers_path)


def test_marker_csv_rejects_duplicate_points(tmp_path):
    markers_path = tmp_path / "markers.csv"
    markers_path.write_text(
        "\n".join(
            [
                "image_x,image_y,ground_x_m,ground_y_m",
                "120.0,200.0,0.0,0.0",
                "120.0,200.0,1.0,0.0",
                "120.0,140.0,0.0,1.0",
                "200.0,140.0,1.0,1.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="4 unique image"):
        read_marker_csv(markers_path)


def test_marker_csv_rejects_collinear_points(tmp_path):
    markers_path = tmp_path / "markers.csv"
    markers_path.write_text(
        "\n".join(
            [
                "image_x,image_y,ground_x_m,ground_y_m",
                "100.0,100.0,0.0,0.0",
                "200.0,100.0,1.0,0.0",
                "300.0,100.0,2.0,0.0",
                "400.0,100.0,3.0,0.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="marker points must not be collinear"):
        read_marker_csv(markers_path)
