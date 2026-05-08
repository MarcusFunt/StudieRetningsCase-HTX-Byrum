import subprocess
import sys

from scripts.capture_serial import CSV_COLUMNS, validate_csv_row
from scripts.capture_wifi import DEFAULT_UDP_PORT, parse_udp_payload_lines


def test_udp_payload_parser_splits_non_empty_lines():
    payload = b"#status,wifi_ap_ready\n\n100,1,0,10.5,20.0,4,8,0.700,0\n"

    assert parse_udp_payload_lines(payload) == [
        "#status,wifi_ap_ready",
        "100,1,0,10.5,20.0,4,8,0.700,0",
    ]


def test_wifi_rows_use_shared_csv_schema():
    row = ["100", "1", "0", "10.5", "20.0", "4", "8", "0.700", "0"]

    assert CSV_COLUMNS == [
        "timestamp_ms",
        "frame_id",
        "detection_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "confidence",
        "target",
    ]
    assert validate_csv_row(row) == row


def test_default_udp_port_matches_firmware():
    assert DEFAULT_UDP_PORT == 4210


def test_capture_wifi_script_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/capture_wifi.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PedFlow sensor Wi-Fi UDP feed" in result.stdout
