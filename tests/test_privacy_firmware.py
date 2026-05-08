from pathlib import Path


SKETCH = Path("GroveAIV2_Box_AP/GroveAIV2_Box_AP.ino")


def test_field_firmware_has_no_live_stream_or_image_capture_api():
    source = SKETCH.read_text(encoding="utf-8")
    lower_source = source.lower()

    forbidden = [
        "#include <webserver.h>",
        "wifiserver",
        "webserver",
        "server.on",
        "stream",
        "last_image",
        "save_jpeg",
        "jpeg",
        "camera",
    ]

    for token in forbidden:
        assert token not in lower_source


def test_field_firmware_invokes_detection_without_image_payloads():
    source = SKETCH.read_text(encoding="utf-8")

    assert "AI.invoke(1, false, false)" in source
    assert "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target" in source
    assert "#error,ai_invoke_failed" in source
    assert "ai_begin_failed" in source


def test_field_firmware_sends_anonymous_rows_over_wifi_udp():
    source = SKETCH.read_text(encoding="utf-8")

    assert "#include <WiFi.h>" in source
    assert "#include <WiFiUdp.h>" in source
    assert "WiFi.softAP(" in source
    assert "udp.beginPacket(WIFI_UDP_BROADCAST, WIFI_UDP_PORT)" in source
    assert "PedFlowSensor" in source
    assert "4210" in source


def test_debug_mode_is_usb_only_and_not_wifi_controlled():
    source = SKETCH.read_text(encoding="utf-8")
    lower_source = source.lower()

    assert "isUsbDebugActive" in source
    assert "static_cast<bool>(Serial)" in source
    assert "#status,usb_debug_on" in source
    assert "parsepacket" not in lower_source
    assert "udp.read" not in lower_source
    assert "udp.parsepacket" not in lower_source
