from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

CALIBRATION_CAPTURE_COMMAND = "CALIB_CAPTURE"
IMAGE_BEGIN_PREFIX = "#calibration_image_begin"
IMAGE_END = "#calibration_image_end"
CAPTURE_ERROR_PREFIX = "#error,calibration_capture"
DEFAULT_CAPTURE_TIMEOUT_S = 30.0

_SAFE_BASENAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAX_SAFE_BASENAME_LENGTH = 120
_TRUNCATION_DIGEST_LENGTH = 10


class SerialLike(Protocol):
    def write(self, data: bytes) -> object: ...
    def flush(self) -> object: ...
    def readline(self) -> bytes: ...


@dataclass(frozen=True)
class UsbCalibrationCapture:
    image_path: Path
    metadata_path: Path
    captured_utc: str
    jpeg_byte_count: int
    image_sha256: str


def metadata_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".metadata.json")


def decode_calibration_image_payload(payload: str) -> bytes:
    compact_payload = "".join(payload.split())
    try:
        image = base64.b64decode(compact_payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("Calibration image payload is not valid base64") from exc

    image = image.rstrip(b"\x00")
    if len(image) < 4 or not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
        raise ValueError("Calibration image payload is not a JPEG")
    return image


def request_calibration_image(
    device: SerialLike,
    timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S,
) -> bytes:
    deadline = time.monotonic() + float(timeout_s)
    device.write(f"{CALIBRATION_CAPTURE_COMMAND}\n".encode("ascii"))
    device.flush()

    expected_payload_length: int | None = None
    while True:
        line = _read_serial_line(device, deadline, "calibration image header")
        if line.startswith(CAPTURE_ERROR_PREFIX):
            raise RuntimeError(line)
        if line.startswith(IMAGE_BEGIN_PREFIX):
            expected_payload_length = _parse_begin_length(line)
            break

    payload = _read_serial_line(device, deadline, "calibration image payload")
    if payload.startswith("#error,"):
        raise RuntimeError(payload)
    if expected_payload_length is not None and len(payload) != expected_payload_length:
        raise ValueError(
            "Calibration image payload length mismatch: "
            f"expected {expected_payload_length}, got {len(payload)}"
        )

    end_line = _read_serial_line(device, deadline, "calibration image footer")
    if end_line != IMAGE_END:
        raise ValueError(f"Unexpected calibration image footer: {end_line}")

    return decode_calibration_image_payload(payload)


def capture_usb_calibration_photos(
    port: str,
    baud: int,
    output_dir: str | Path,
    basename: str = "charuco_usb",
    count: int = 1,
    interval_s: float = 1.0,
    settle_delay_s: float = 2.0,
    timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S,
) -> list[UsbCalibrationCapture]:
    import serial

    if int(count) < 1:
        raise ValueError("count must be at least 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_basename = _safe_basename(basename)
    captures: list[UsbCalibrationCapture] = []

    with serial.Serial(str(port).strip(), int(baud), timeout=0.5) as device:
        if settle_delay_s > 0:
            time.sleep(float(settle_delay_s))

        for index in range(1, int(count) + 1):
            if hasattr(device, "reset_input_buffer"):
                device.reset_input_buffer()

            jpeg = request_calibration_image(device, timeout_s=timeout_s)
            captured_at = datetime.now(UTC)
            captures.append(
                _write_capture(
                    output_dir=output_path,
                    basename=safe_basename,
                    index=index,
                    total_count=int(count),
                    captured_at=captured_at,
                    jpeg=jpeg,
                    port=str(port).strip(),
                    baud=int(baud),
                )
            )

            if index < int(count) and interval_s > 0:
                time.sleep(float(interval_s))

    return captures


def _read_serial_line(device: SerialLike, deadline: float, label: str) -> str:
    while time.monotonic() < deadline:
        raw_line = device.readline()
        if not raw_line:
            continue
        return raw_line.decode("utf-8", errors="replace").strip()
    raise TimeoutError(f"Timed out waiting for {label}")


def _parse_begin_length(line: str) -> int | None:
    if line == IMAGE_BEGIN_PREFIX:
        return None
    prefix = f"{IMAGE_BEGIN_PREFIX},"
    if not line.startswith(prefix):
        raise ValueError(f"Unexpected calibration image header: {line}")
    value = line[len(prefix) :]
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid calibration image payload length: {value}") from exc
    if parsed <= 0:
        raise ValueError("Calibration image payload length must be positive")
    return parsed


def _safe_basename(basename: str) -> str:
    if not basename:
        raise ValueError("basename must not be empty")
    path = Path(basename)
    if path.is_absolute() or path.name != basename or basename in {".", ".."}:
        raise ValueError("basename must be a filename stem, not a path")
    if not _SAFE_BASENAME.fullmatch(basename):
        raise ValueError("basename may only contain letters, numbers, dots, dashes, and underscores")
    return _truncate_with_digest(basename, _MAX_SAFE_BASENAME_LENGTH)


def _truncate_with_digest(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:_TRUNCATION_DIGEST_LENGTH]
    prefix_length = max_length - len(digest) - 1
    if prefix_length < 1:
        raise ValueError("max_length is too short for deterministic truncation")
    return f"{value[:prefix_length]}-{digest}"


def _write_capture(
    output_dir: Path,
    basename: str,
    index: int,
    total_count: int,
    captured_at: datetime,
    jpeg: bytes,
    port: str,
    baud: int,
) -> UsbCalibrationCapture:
    image_path = _unique_capture_path(output_dir, basename, index, total_count, captured_at)
    image_path.write_bytes(jpeg)

    captured_utc = captured_at.isoformat()
    image_sha256 = hashlib.sha256(jpeg).hexdigest()
    metadata_path = metadata_path_for(image_path)
    metadata = {
        "capture_utc": captured_utc,
        "transport": "usb_serial",
        "firmware_command": CALIBRATION_CAPTURE_COMMAND,
        "port": port,
        "baud": int(baud),
        "output_path": str(image_path),
        "jpeg_byte_count": len(jpeg),
        "image_sha256": image_sha256,
        "privacy_note": (
            "Calibration image capture is requested over local USB serial only. "
            "The firmware does not expose Wi-Fi image capture or streaming."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    return UsbCalibrationCapture(
        image_path=image_path,
        metadata_path=metadata_path,
        captured_utc=captured_utc,
        jpeg_byte_count=len(jpeg),
        image_sha256=image_sha256,
    )


def _unique_capture_path(
    output_dir: Path,
    basename: str,
    index: int,
    total_count: int,
    captured_at: datetime,
) -> Path:
    timestamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    index_suffix = f"_{index:03d}" if total_count > 1 else ""
    candidate = output_dir / f"{basename}_{timestamp}{index_suffix}.jpg"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{basename}_{timestamp}{index_suffix}_{suffix}.jpg"
        suffix += 1
    return candidate
