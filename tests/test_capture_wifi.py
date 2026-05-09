import json
import socket
import subprocess
import sys
import time

from pedflow.capture import UdpCsvCaptureWorker
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


def test_udp_capture_worker_writes_rows_and_metadata(tmp_path):
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    output_path = tmp_path / "capture.csv"
    worker = UdpCsvCaptureWorker(output_path=output_path, host="127.0.0.1", port=port)
    worker.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            snapshot = worker.snapshot()
            if any("capturing UDP CSV" in line for line in snapshot.log_lines):
                break
            assert snapshot.error is None
            time.sleep(0.05)
        else:
            raise AssertionError("UDP worker did not start")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(
                b"#status,firmware_version,0.1.0\n100,1,0,10.5,20.0,4,8,0.700,0\n",
                ("127.0.0.1", port),
            )

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            snapshot = worker.snapshot()
            if snapshot.rows_written == 1:
                break
            assert snapshot.error is None
            time.sleep(0.05)
        else:
            raise AssertionError("UDP worker did not write the sample row")
    finally:
        worker.stop()

    text = output_path.read_text(encoding="utf-8")
    assert CSV_COLUMNS[0] in text
    assert "100,1,0,10.5,20.0,4,8,0.700,0" in text
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["session_type"] == "detection_capture"
    assert metadata["firmware_version"] == "0.1.0"
    assert metadata["calibration_json_path"] is None
    assert metadata["settings"]["transport"] == "udp_wifi_ap"
    assert metadata["settings"]["port"] == port
    assert metadata["output_files"]["detections_csv"] == str(output_path)
    assert metadata["output_files"]["metadata_json"] == str(metadata_path)
