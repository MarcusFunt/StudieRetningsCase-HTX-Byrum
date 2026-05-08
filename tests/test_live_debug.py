import numpy as np

from pedflow.live_debug import analyze_live_debug_rows, build_bbox_overlay, parse_debug_serial_line


def test_debug_serial_parser_keeps_rows_and_ignores_control_lines():
    assert parse_debug_serial_line("#status,usb_debug_on").kind == "comment"
    assert (
        parse_debug_serial_line(
            "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target"
        ).kind
        == "header"
    )

    parsed = parse_debug_serial_line("100,1,0,10.0,20.0,4,8,0.700,0")
    assert parsed.kind == "row"
    assert parsed.row == ["100", "1", "0", "10.0", "20.0", "4", "8", "0.700", "0"]

    assert parse_debug_serial_line("bad,row").kind == "invalid"


def test_bbox_overlay_adds_ground_contact_foot_point_in_image_coordinates():
    rows = [["100", "1", "0", "10.0", "20.0", "4", "8", "0.700", "0"]]
    result = analyze_live_debug_rows(rows, calibration=None)

    overlay = build_bbox_overlay(result.detections)
    assert overlay.loc[0, "bbox_x1"] == 14.0
    assert overlay.loc[0, "bbox_y1"] == 28.0
    assert overlay.loc[0, "foot_x"] == 12.0
    assert overlay.loc[0, "foot_y"] == 28.0
    assert not result.calibration_loaded


def test_live_debug_analysis_uses_calibration_for_pedpy_metrics():
    rows = [
        ["0", "0", "0", "8.0", "12.0", "4", "8", "0.900", "0"],
        ["500", "1", "0", "13.0", "12.0", "4", "8", "0.900", "0"],
    ]
    calibration = {
        "K": np.eye(3).tolist(),
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "H_image_to_ground": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]],
    }

    result = analyze_live_debug_rows(rows, calibration=calibration)

    assert result.calibration_loaded
    assert len(result.ground) == 2
    assert result.ground.loc[0, "ground_x_m"] == 1.0
    assert int(result.summary.loc[0, "pedestrian_count"]) == 1
