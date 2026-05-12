import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pedflow.usb_calibration_capture import (
    _MAX_SAFE_BASENAME_LENGTH,
    CALIBRATION_CAPTURE_COMMAND,
    _capture_settings_from_status_line,
    _safe_basename,
    _unique_capture_path,
    _write_capture,
    decode_calibration_image_payload,
    metadata_path_for,
    request_calibration_image,
)


class FakeSerial:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines
        self.writes: list[bytes] = []
        self.flushed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        self.flushed = True

    def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        return b""


def test_request_calibration_image_sends_usb_command_and_decodes_jpeg():
    device = FakeSerial(
        [
            b"#status,usb_debug_on\n",
            b"timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target\n",
            b"#status,calibration_capture_started\n",
            b"#calibration_image_begin,8\n",
            b"/9j/2Q==\n",
            b"#calibration_image_end\n",
        ]
    )

    assert request_calibration_image(device, timeout_s=0.5) == b"\xff\xd8\xff\xd9"
    assert device.writes == [f"{CALIBRATION_CAPTURE_COMMAND}\n".encode("ascii")]
    assert device.flushed


def test_request_calibration_image_reconstructs_chunked_payload():
    device = FakeSerial(
        [
            b"#status,calibration_capture_started\n",
            b"#calibration_image_begin,8\n",
            b"/9j/\n",
            b"2Q==\n",
            b"#calibration_image_end\n",
        ]
    )

    assert request_calibration_image(device, timeout_s=0.5) == b"\xff\xd8\xff\xd9"


def test_request_calibration_image_rejects_truncated_chunk_payload():
    device = FakeSerial(
        [
            b"#status,calibration_capture_started\n",
            b"#calibration_image_begin,12\n",
            b"/9j/\n",
            b"2Q==\n",
            b"#calibration_image_end\n",
        ]
    )

    with pytest.raises(ValueError, match="payload length mismatch"):
        request_calibration_image(device, timeout_s=0.5)


def test_request_calibration_image_fails_on_malformed_chunk_error():
    device = FakeSerial(
        [
            b"#status,calibration_capture_started\n",
            b"#calibration_image_begin,8\n",
            b"/9j/\n",
            b"#error,calibration_capture_failed,chunk_order_1_2\n",
        ]
    )

    with pytest.raises(RuntimeError, match="chunk_order_1_2"):
        request_calibration_image(device, timeout_s=0.5)


def test_request_calibration_image_collects_post_capture_status_lines():
    device = FakeSerial(
        [
            b"#status,calibration_capture_started\n",
            b"#calibration_image_begin,8\n",
            b"/9j/2Q==\n",
            b"#calibration_image_end\n",
            b"#status,calibration_sensor_restored,0,240x240 Auto\n",
        ]
    )
    statuses: list[str] = []

    assert request_calibration_image(device, timeout_s=0.5, status_handler=statuses.append) == (
        b"\xff\xd8\xff\xd9"
    )

    assert "#status,calibration_sensor_restored,0,240x240 Auto" in statuses


def test_decode_calibration_image_payload_rejects_non_jpeg_payload():
    with pytest.raises(ValueError, match="JPEG"):
        decode_calibration_image_payload("bm90LWpwZWc=")


def test_decode_calibration_image_payload_allows_nul_padding_after_jpeg_eoi():
    assert decode_calibration_image_payload("/9j/2QAAAA==") == b"\xff\xd8\xff\xd9"


def test_usb_calibration_metadata_path_uses_json_sidecar():
    assert metadata_path_for(Path("data/calibration_images/charuco/photo.jpg")) == Path(
        "data/calibration_images/charuco/photo.metadata.json"
    )


def test_usb_capture_basename_is_truncated_before_timestamp(tmp_path):
    basename = "charuco_usb_" + ("b" * 200)

    shortened = _safe_basename(basename)
    image_path = _unique_capture_path(
        tmp_path,
        shortened,
        index=1,
        total_count=1,
        captured_at=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
    )

    assert len(shortened) == _MAX_SAFE_BASENAME_LENGTH
    assert shortened == _safe_basename(basename)
    assert shortened != _safe_basename(f"{basename}b")
    assert image_path.name.startswith(shortened)
    assert len(image_path.name) < 255


def test_usb_capture_sidecar_records_manifest_fields(tmp_path):
    capture = _write_capture(
        output_dir=tmp_path,
        basename="charuco_usb",
        index=1,
        total_count=1,
        started_at=datetime(2026, 5, 9, 10, 0, tzinfo=UTC),
        captured_at=datetime(2026, 5, 9, 10, 0, 2, tzinfo=UTC),
        jpeg=b"\xff\xd8\xff\xd9",
        port="COM5",
        baud=115200,
        timeout_s=12.5,
        firmware_version="0.1.0",
        calibration_path=tmp_path / "calibration.json",
        settings={
            "workflow": "ground_homography",
            "requested_sensor_opt_id": 5,
            "requested_sensor_detail": "640x480 Calibration HQ",
            "previous_sensor_opt_id": 0,
            "previous_sensor_detail": "240x240 Auto",
            "selected_sensor_opt_id": 5,
            "selected_sensor_detail": "640x480 Calibration HQ",
            "restored_sensor_opt_id": 0,
            "restored_sensor_detail": "240x240 Auto",
            "sensor_restore_status": "restored",
            "sscma_transport": "uart_921600",
            "jpeg_qtable": "JPEG_ENC_QTABLE_4X",
            "image_width": 640,
            "image_height": 480,
            "reported_jpeg_byte_count": 4,
            "reported_base64_length": 8,
            "reported_chunk_count": 1,
        },
    )

    metadata = json.loads(capture.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["session_type"] == "calibration_image_capture"
    assert metadata["start_utc"] == "2026-05-09T10:00:00+00:00"
    assert metadata["end_utc"] == "2026-05-09T10:00:02+00:00"
    assert metadata["firmware_version"] == "0.1.0"
    assert metadata["calibration_json_path"].endswith("calibration.json")
    assert metadata["settings"]["transport"] == "usb_serial"
    assert metadata["settings"]["workflow"] == "ground_homography"
    assert metadata["settings"]["requested_sensor_opt_id"] == 5
    assert metadata["settings"]["requested_sensor_detail"] == "640x480 Calibration HQ"
    assert metadata["settings"]["previous_sensor_opt_id"] == 0
    assert metadata["settings"]["selected_sensor_opt_id"] == 5
    assert metadata["settings"]["sensor_restore_status"] == "restored"
    assert metadata["settings"]["restored_sensor_detail"] == "240x240 Auto"
    assert metadata["settings"]["sscma_transport"] == "uart_921600"
    assert metadata["settings"]["jpeg_qtable"] == "JPEG_ENC_QTABLE_4X"
    assert metadata["settings"]["image_width"] == 640
    assert metadata["settings"]["reported_chunk_count"] == 1
    assert metadata["output_files"]["calibration_image"] == str(capture.image_path)
    assert metadata["output_files"]["metadata_json"] == str(capture.metadata_path)


def test_usb_capture_status_lines_are_mapped_to_metadata_settings():
    settings = {}
    for line in [
        "#status,calibration_sensor_requested,5,640x480 Calibration HQ",
        "#status,calibration_sensor_previous,0,240x240 Auto",
        "#status,calibration_sensor_selected,5,640x480 Calibration HQ",
        "#status,calibration_sensor_restored,0,240x240 Auto",
        "#status,calibration_capture_transport,uart_921600",
        "#status,calibration_image_resolution,640,480",
        "#status,calibration_jpeg_byte_count,12345",
        "#status,calibration_base64_length,16460",
        "#status,calibration_chunk_count,5",
        "#status,calibration_sample_sensor_opt_id,5",
        "#status,calibration_jpeg_qtable,JPEG_ENC_QTABLE_4X",
    ]:
        settings.update(_capture_settings_from_status_line(line))

    assert settings == {
        "requested_sensor_opt_id": 5,
        "requested_sensor_detail": "640x480 Calibration HQ",
        "previous_sensor_opt_id": 0,
        "previous_sensor_detail": "240x240 Auto",
        "selected_sensor_opt_id": 5,
        "selected_sensor_detail": "640x480 Calibration HQ",
        "restored_sensor_opt_id": 0,
        "restored_sensor_detail": "240x240 Auto",
        "sensor_restore_status": "restored",
        "sscma_transport": "uart_921600",
        "image_width": 640,
        "image_height": 480,
        "reported_jpeg_byte_count": 12345,
        "reported_base64_length": 16460,
        "reported_chunk_count": 5,
        "calibsample_sensor_opt_id": 5,
        "jpeg_qtable": "JPEG_ENC_QTABLE_4X",
    }

    assert _capture_settings_from_status_line(
        "#error,calibration_sensor_restore_failed,response_timeout"
    ) == {
        "sensor_restore_error": "response_timeout",
        "sensor_restore_status": "failed",
    }
