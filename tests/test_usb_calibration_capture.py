from pathlib import Path

import pytest

from pedflow.usb_calibration_capture import (
    CALIBRATION_CAPTURE_COMMAND,
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


def test_decode_calibration_image_payload_rejects_non_jpeg_payload():
    with pytest.raises(ValueError, match="JPEG"):
        decode_calibration_image_payload("bm90LWpwZWc=")


def test_decode_calibration_image_payload_allows_nul_padding_after_jpeg_eoi():
    assert decode_calibration_image_payload("/9j/2QAAAA==") == b"\xff\xd8\xff\xd9"


def test_usb_calibration_metadata_path_uses_json_sidecar():
    assert metadata_path_for(Path("data/calibration_images/charuco/photo.jpg")) == Path(
        "data/calibration_images/charuco/photo.metadata.json"
    )
