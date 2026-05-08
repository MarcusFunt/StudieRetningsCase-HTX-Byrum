from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/generate_secrets.py")


def run_generator(project_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(project_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_generator_writes_matching_device_and_laptop_secrets(tmp_path: Path):
    result = run_generator(
        tmp_path,
        "--ssid",
        "PedFlowTest",
        "--password",
        "ClassroomPass42",
    )

    source_json = tmp_path / "secrets" / "pedflow_wifi.json"
    firmware_header = tmp_path / "GroveAIV2_Box_AP" / "pedflow_secrets.h"
    laptop_env = tmp_path / "secrets" / "pedflow_wifi.env"
    laptop_text = tmp_path / "secrets" / "pedflow_wifi.txt"
    windows_profile = tmp_path / "secrets" / "PedFlowTest-wifi-profile.xml"

    data = json.loads(source_json.read_text(encoding="utf-8"))
    assert data["wifi_ap_ssid"] == "PedFlowTest"
    assert data["wifi_ap_password"] == "ClassroomPass42"
    assert 'PEDFLOW_WIFI_AP_SSID[] = "PedFlowTest";' in firmware_header.read_text(
        encoding="utf-8"
    )
    assert 'PEDFLOW_WIFI_AP_PASSWORD[] = "ClassroomPass42";' in firmware_header.read_text(
        encoding="utf-8"
    )
    assert "PEDFLOW_WIFI_AP_SSID=PedFlowTest" in laptop_env.read_text(encoding="utf-8")
    assert "PEDFLOW_WIFI_AP_PASSWORD=ClassroomPass42" in laptop_env.read_text(encoding="utf-8")
    assert "SSID: PedFlowTest" in laptop_text.read_text(encoding="utf-8")
    assert "Password: ClassroomPass42" in laptop_text.read_text(encoding="utf-8")
    assert "<keyMaterial>ClassroomPass42</keyMaterial>" in windows_profile.read_text(
        encoding="utf-8"
    )
    assert "ClassroomPass42" not in result.stdout


def test_generator_reuses_existing_password_until_rotated(tmp_path: Path):
    run_generator(tmp_path, "--password-length", "12")
    source_json = tmp_path / "secrets" / "pedflow_wifi.json"
    original_password = json.loads(source_json.read_text(encoding="utf-8"))["wifi_ap_password"]

    run_generator(tmp_path)
    reused_password = json.loads(source_json.read_text(encoding="utf-8"))["wifi_ap_password"]

    run_generator(tmp_path, "--rotate", "--password-length", "12")
    rotated_password = json.loads(source_json.read_text(encoding="utf-8"))["wifi_ap_password"]

    assert reused_password == original_password
    assert rotated_password != original_password


def test_generated_secret_paths_are_gitignored():
    ignored_lines = {
        line.strip() for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert "secrets/" in ignored_lines
    assert "GroveAIV2_Box_AP/pedflow_secrets.h" in ignored_lines
