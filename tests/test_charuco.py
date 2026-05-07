import json
import subprocess
import sys

import cv2

from pedflow.calibration import charuco_board_from_metadata, detect_charuco_image_points


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
