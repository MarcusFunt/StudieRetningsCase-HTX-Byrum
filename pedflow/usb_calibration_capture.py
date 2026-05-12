from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .serial_protocol import UNKNOWN_FIRMWARE_VERSION, firmware_version_from_status_line

CALIBRATION_CAPTURE_COMMAND = "CALIB_CAPTURE"
IMAGE_BEGIN_PREFIX = "#calibration_image_begin"
IMAGE_END = "#calibration_image_end"
CAPTURE_ERROR_PREFIX = "#error,calibration_capture"
DEFAULT_CAPTURE_TIMEOUT_S = 60.0
POST_CAPTURE_STATUS_TIMEOUT_S = 1.0

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
    started_utc: str
    captured_utc: str
    jpeg_byte_count: int
    image_sha256: str
    firmware_version: str


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
    status_handler: Callable[[str], None] | None = None,
) -> bytes:
    deadline = time.monotonic() + float(timeout_s)
    device.write(f"{CALIBRATION_CAPTURE_COMMAND}\n".encode("ascii"))
    device.flush()

    expected_payload_length: int | None = None
    while True:
        line = _read_serial_line(device, deadline, "calibration image header")
        if line.startswith("#") and status_handler is not None:
            status_handler(line)
        if line.startswith(CAPTURE_ERROR_PREFIX):
            raise RuntimeError(line)
        if line.startswith(IMAGE_BEGIN_PREFIX):
            expected_payload_length = _parse_begin_length(line)
            break

    payload_parts: list[str] = []
    while True:
        line = _read_serial_line(device, deadline, "calibration image payload")
        if line.startswith("#") and status_handler is not None:
            status_handler(line)
        if line.startswith("#error,"):
            raise RuntimeError(line)
        if line == IMAGE_END:
            break
        if line.startswith("#"):
            continue
        payload_parts.append(line)

    payload = "".join(payload_parts)
    if expected_payload_length is not None and len(payload) != expected_payload_length:
        raise ValueError(
            "Calibration image payload length mismatch: "
            f"expected {expected_payload_length}, got {len(payload)}"
        )

    _drain_post_capture_status(
        device,
        min(deadline, time.monotonic() + POST_CAPTURE_STATUS_TIMEOUT_S),
        status_handler,
    )
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
    firmware_version: str = UNKNOWN_FIRMWARE_VERSION,
    calibration_path: str | Path | None = None,
    settings: dict | None = None,
) -> list[UsbCalibrationCapture]:
    import serial

    if int(count) < 1:
        raise ValueError("count must be at least 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_basename = _safe_basename(basename)
    captures: list[UsbCalibrationCapture] = []
    current_firmware_version = firmware_version or UNKNOWN_FIRMWARE_VERSION

    current_capture_settings: dict[str, object] = {}

    def handle_status(line: str) -> None:
        nonlocal current_firmware_version
        detected_version = firmware_version_from_status_line(line)
        if detected_version is not None:
            current_firmware_version = detected_version
        current_capture_settings.update(_capture_settings_from_status_line(line))

    with serial.Serial(str(port).strip(), int(baud), timeout=0.5) as device:
        if settle_delay_s > 0:
            time.sleep(float(settle_delay_s))

        for index in range(1, int(count) + 1):
            if hasattr(device, "reset_input_buffer"):
                device.reset_input_buffer()

            current_capture_settings = {}
            started_at = datetime.now(UTC)
            jpeg = request_calibration_image(
                device,
                timeout_s=timeout_s,
                status_handler=handle_status,
            )
            captured_at = datetime.now(UTC)
            captures.append(
                _write_capture(
                    output_dir=output_path,
                    basename=safe_basename,
                    index=index,
                    total_count=int(count),
                    started_at=started_at,
                    captured_at=captured_at,
                    jpeg=jpeg,
                    port=str(port).strip(),
                    baud=int(baud),
                    timeout_s=float(timeout_s),
                    firmware_version=current_firmware_version,
                    calibration_path=calibration_path,
                    settings=(settings or {}) | current_capture_settings,
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


def _drain_post_capture_status(
    device: SerialLike,
    deadline: float,
    status_handler: Callable[[str], None] | None,
) -> None:
    if status_handler is None:
        return

    while time.monotonic() < deadline:
        raw_line = device.readline()
        if not raw_line:
            return
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("#"):
            return
        status_handler(line)


def _capture_settings_from_status_line(line: str) -> dict[str, object]:
    parts = line.split(",", 3)
    if len(parts) < 2:
        return {}

    level = parts[0].lstrip("#")
    code = parts[1]
    value = parts[2] if len(parts) >= 3 else ""
    detail = parts[3] if len(parts) >= 4 else ""
    if code == "calibration_sensor_requested":
        return _sensor_option_settings("requested_sensor", value, detail)
    if code == "calibration_sensor_previous":
        return _sensor_option_settings("previous_sensor", value, detail)
    if code == "calibration_sensor_selected":
        return _sensor_option_settings("selected_sensor", value, detail)
    if code == "calibration_sensor_restored":
        return _sensor_option_settings("restored_sensor", value, detail) | {
            "sensor_restore_status": "restored"
        }
    if code == "calibration_capture_transport":
        return {"sscma_transport": value}
    if code == "calibration_image_resolution":
        return _image_resolution_settings(value, detail)
    if code == "calibration_jpeg_byte_count":
        return _int_setting("reported_jpeg_byte_count", value)
    if code == "calibration_base64_length":
        return _int_setting("reported_base64_length", value)
    if code == "calibration_chunk_count":
        return _int_setting("reported_chunk_count", value)
    if code == "calibration_sample_sensor_opt_id":
        return _int_setting("calibsample_sensor_opt_id", value)
    if code == "calibration_jpeg_qtable":
        return {"jpeg_qtable": value} if value else {}
    if level == "status" and code == "calibration_sensor_restore_not_needed":
        return {"sensor_restore_status": "not_needed"}
    if level != "error":
        return {}
    if code == "calibration_sensor_query_failed":
        return {"sensor_query_error": value}
    if code == "calibration_sensor_select_failed":
        return {"sensor_select_error": value}
    if code == "calibration_sensor_restore_failed":
        return {
            "sensor_restore_error": value,
            "sensor_restore_status": "failed",
        }
    return {}


def _sensor_option_settings(prefix: str, opt_id: str, detail: str) -> dict[str, object]:
    settings: dict[str, object] = {}
    try:
        settings[f"{prefix}_opt_id"] = int(opt_id)
    except ValueError:
        settings[f"{prefix}_opt_id_raw"] = opt_id
    if detail:
        settings[f"{prefix}_detail"] = detail
    return settings


def _int_setting(key: str, value: str) -> dict[str, object]:
    try:
        return {key: int(value)}
    except ValueError:
        return {f"{key}_raw": value}


def _image_resolution_settings(width_or_resolution: str, height: str) -> dict[str, object]:
    if height:
        raw_width = width_or_resolution
        raw_height = height
    elif "x" in width_or_resolution.lower():
        raw_width, raw_height = re.split("x", width_or_resolution, maxsplit=1, flags=re.IGNORECASE)
    else:
        return {"image_resolution_raw": width_or_resolution}

    try:
        return {"image_width": int(raw_width), "image_height": int(raw_height)}
    except ValueError:
        return {"image_resolution_raw": f"{raw_width}x{raw_height}"}


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
        raise ValueError(
            "basename may only contain letters, numbers, dots, dashes, and underscores"
        )
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
    started_at: datetime,
    captured_at: datetime,
    jpeg: bytes,
    port: str,
    baud: int,
    timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S,
    firmware_version: str = UNKNOWN_FIRMWARE_VERSION,
    calibration_path: str | Path | None = None,
    settings: dict | None = None,
) -> UsbCalibrationCapture:
    image_path = _unique_capture_path(output_dir, basename, index, total_count, captured_at)
    image_path.write_bytes(jpeg)

    started_utc = started_at.isoformat()
    captured_utc = captured_at.isoformat()
    image_sha256 = hashlib.sha256(jpeg).hexdigest()
    metadata_path = metadata_path_for(image_path)
    capture_settings = {
        "transport": "usb_serial",
        "firmware_command": CALIBRATION_CAPTURE_COMMAND,
        "port": port,
        "baud": int(baud),
        "basename": basename,
        "index": int(index),
        "total_count": int(total_count),
        "timeout_s": float(timeout_s),
    }
    if settings:
        capture_settings.update(settings)
    metadata = {
        "schema_version": 1,
        "session_type": "calibration_image_capture",
        "start_utc": started_utc,
        "end_utc": captured_utc,
        "capture_utc": captured_utc,
        "firmware_version": firmware_version or UNKNOWN_FIRMWARE_VERSION,
        "calibration_json_path": str(calibration_path) if calibration_path is not None else None,
        "settings": capture_settings,
        "output_path": str(image_path),
        "output_files": {
            "calibration_image": str(image_path),
            "metadata_json": str(metadata_path),
        },
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
        started_utc=started_utc,
        captured_utc=captured_utc,
        jpeg_byte_count=len(jpeg),
        image_sha256=image_sha256,
        firmware_version=firmware_version or UNKNOWN_FIRMWARE_VERSION,
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
