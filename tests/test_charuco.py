import json
import subprocess
import sys

import cv2
import numpy as np

from pedflow.calibration import (
    DEFAULT_CHARUCO_DPI,
    DEFAULT_CHARUCO_MARGIN_MM,
    DEFAULT_CHARUCO_MARKER_LENGTH_MM,
    DEFAULT_CHARUCO_PAPER_HEIGHT_MM,
    DEFAULT_CHARUCO_PAPER_ORIENTATION,
    DEFAULT_CHARUCO_PAPER_SIZE,
    DEFAULT_CHARUCO_PAPER_WIDTH_MM,
    DEFAULT_CHARUCO_SQUARE_LENGTH_MM,
    DEFAULT_CHARUCO_SQUARES_X,
    DEFAULT_CHARUCO_SQUARES_Y,
    _charuco_object_points_to_ground,
    charuco_board_from_metadata,
    compute_ground_homography_from_charuco,
    detect_charuco_image_points,
    generate_charuco_board,
)


def test_generate_charuco_board_script_writes_printable_board_and_metadata(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_charuco_board.py",
            "--output-dir",
            str(tmp_path),
            "--squares-x",
            "5",
            "--squares-y",
            "4",
            "--square-length-mm",
            "30",
            "--marker-length-mm",
            "22",
            "--dpi",
            "150",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metadata_path = tmp_path / "charuco_board.json"
    png_path = tmp_path / "charuco_board.png"
    pdf_path = tmp_path / "charuco_board.pdf"

    assert metadata_path.exists()
    assert png_path.exists()
    assert pdf_path.exists()
    assert "Print the PDF at 100% scale" in result.stdout

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["board_type"] == "charuco"
    assert metadata["squares_x"] == 5
    assert metadata["squares_y"] == 4
    assert metadata["square_length_mm"] == 30.0
    assert metadata["marker_length_mm"] == 22.0
    assert metadata["dictionary"] == "DICT_5X5_100"
    assert len(metadata["png_sha256"]) == 64

    image = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None

    board = charuco_board_from_metadata(metadata)
    object_points, image_points, marker_count = detect_charuco_image_points(
        image,
        board,
        min_corners=4,
    )

    assert len(object_points) == len(image_points)
    assert len(image_points) >= 4
    assert marker_count > 0


def test_generate_charuco_board_defaults_to_a3_landscape_board(tmp_path):
    metadata = generate_charuco_board(tmp_path)

    assert metadata["paper_size"] == DEFAULT_CHARUCO_PAPER_SIZE
    assert metadata["paper_orientation"] == DEFAULT_CHARUCO_PAPER_ORIENTATION
    assert metadata["paper_width_mm"] == DEFAULT_CHARUCO_PAPER_WIDTH_MM
    assert metadata["paper_height_mm"] == DEFAULT_CHARUCO_PAPER_HEIGHT_MM
    assert metadata["squares_x"] == DEFAULT_CHARUCO_SQUARES_X
    assert metadata["squares_y"] == DEFAULT_CHARUCO_SQUARES_Y
    assert metadata["square_length_mm"] == DEFAULT_CHARUCO_SQUARE_LENGTH_MM
    assert metadata["marker_length_mm"] == DEFAULT_CHARUCO_MARKER_LENGTH_MM
    assert metadata["margin_mm"] == DEFAULT_CHARUCO_MARGIN_MM
    assert metadata["dpi"] == DEFAULT_CHARUCO_DPI
    assert metadata["charuco_corners_x"] * metadata["charuco_corners_y"] == 35
    assert metadata["board_width_m"] == 0.356
    assert metadata["board_height_m"] == 0.267
    assert "A3 landscape" in metadata["print_instructions"]


def test_charuco_object_points_rotate_into_ground_coordinates():
    object_points = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )

    ground = _charuco_object_points_to_ground(
        object_points,
        board_origin_x_m=10.0,
        board_origin_y_m=20.0,
        board_rotation_deg=90.0,
    )

    np.testing.assert_allclose(
        ground,
        [
            [10.0, 20.0],
            [10.0, 21.0],
            [9.0, 20.0],
            [9.0, 21.0],
        ],
        atol=1e-12,
    )


def test_ground_homography_can_be_built_from_flat_charuco_photo(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_charuco_board.py",
            "--output-dir",
            str(tmp_path),
            "--squares-x",
            "5",
            "--squares-y",
            "4",
            "--square-length-mm",
            "30",
            "--marker-length-mm",
            "22",
            "--dpi",
            "150",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = compute_ground_homography_from_charuco(
        image_path=tmp_path / "charuco_board.png",
        board_metadata_path=tmp_path / "charuco_board.json",
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        board_origin_x_m=1.0,
        board_origin_y_m=2.0,
        board_rotation_deg=15.0,
        min_corners=4,
    )

    assert result["ground_homography_source"] == "charuco_board"
    assert result["ground_charuco_detected_corner_count"] >= 4
    assert result["ground_marker_count"] == result["ground_charuco_detected_corner_count"]
    assert result["ground_marker_max_residual_m"] < 1e-4
