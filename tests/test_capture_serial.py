from pathlib import Path

import pytest

from scripts.capture_serial import _metadata_path_for, parse_serial_csv_line, validate_csv_row


def test_serial_parser_ignores_privacy_safe_comment_lines():
    assert parse_serial_csv_line("#error,ai_begin_failed") is None
    assert parse_serial_csv_line("#status,ai_ready") is None


def test_serial_row_validation_rejects_malformed_values():
    valid = ["100", "1", "0", "10.5", "20.0", "4", "8", "0.700", "0"]

    assert validate_csv_row(valid) == valid

    invalid = valid.copy()
    invalid[5] = "-1"
    with pytest.raises(ValueError, match="bbox_w"):
        validate_csv_row(invalid)

    invalid = valid.copy()
    invalid[7] = "1.4"
    with pytest.raises(ValueError, match="confidence"):
        validate_csv_row(invalid)


def test_capture_metadata_path_uses_json_sidecar():
    assert _metadata_path_for(Path("data/detections/session.csv")) == Path(
        "data/detections/session.metadata.json"
    )
