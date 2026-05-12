from pathlib import Path

SKETCH = Path("GroveAIV2_Box_AP/GroveAIV2_Box_AP.ino")


def test_field_firmware_has_no_live_stream_or_network_capture_api():
    source = SKETCH.read_text(encoding="utf-8")
    lower_source = source.lower()

    forbidden = [
        "#include <webserver.h>",
        "wifiserver",
        "webserver",
        "server.on",
        "stream",
        "save_jpeg",
        "camera",
    ]

    for token in forbidden:
        assert token not in lower_source


def test_field_firmware_invokes_detection_without_image_payloads():
    source = SKETCH.read_text(encoding="utf-8")

    assert 'SSCMA_INVOKE_DETECTIONS_COMMAND[] = "AT+INVOKE=1,0,1\\r\\n"' in source
    assert "invokeForDetections" in source
    assert (
        "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target" in source
    )
    assert "#error,ai_invoke_failed" in source
    assert "ai_begin_failed" in source
    assert "SSCMA_RESET_PIN = D3" in source
    assert "resetSscmaModule" in source
    assert "AI.invoke(" not in source


def test_calibration_image_capture_is_usb_serial_only():
    source = SKETCH.read_text(encoding="utf-8")
    lower_source = source.lower()

    assert 'USB_CALIBRATION_CAPTURE_COMMAND[] = "CALIB_CAPTURE"' in source
    assert 'SSCMA_CALIB_SAMPLE_IMAGE_COMMAND[] = "AT+CALIBSAMPLE=1\\r\\n"' in source
    assert 'SSCMA_SENSOR_QUERY_COMMAND[] = "AT+SENSOR?\\r\\n"' in source
    assert "SSCMA_CALIBRATION_SENSOR_OPT_ID = SSCMA_SENSOR_OPT_640X480_CALIBRATION_HQ" in source
    assert "SSCMA_SENSOR_OPT_640X480 = 2" in source
    assert "SSCMA_SENSOR_OPT_640X480_CALIBRATION_HQ = 5" in source
    assert "HardwareSerial sscmaSerial(1)" in source
    assert "sscmaSerial.begin(SSCMA_UART_BAUD, SERIAL_8N1, SSCMA_UART_RX_PIN, SSCMA_UART_TX_PIN)" in source
    assert "AT+SENSOR=1,1,%d" in source
    assert "calibration_sensor_selected" in source
    assert "calibration_sensor_restored" in source
    assert 'strcmp(name, "CALIBSAMPLE")' in source
    assert 'data["image_chunk"]' in source
    assert "calibration_capture_transport" in source
    assert "calibration_jpeg_qtable" in source
    assert "calibration_chunk_count" in source
    assert "#calibration_image_begin" in source
    assert "Serial.println(CALIBRATION_IMAGE_END)" in source
    assert "#include <Wire.h>" not in source
    assert "Wire." not in source
    assert "parsepacket" not in lower_source
    assert "udp.read" not in lower_source
    assert "sendUdpLine(image" not in source
    assert "emitTelemetryLine(image" not in source


def test_field_firmware_sends_anonymous_rows_over_wifi_udp():
    source = SKETCH.read_text(encoding="utf-8")

    assert "#include <WiFi.h>" in source
    assert "#include <WiFiUdp.h>" in source
    assert '#include "pedflow_secrets.h"' in source
    assert "WiFi.softAP(" in source
    assert "PEDFLOW_WIFI_AP_SSID" in source
    assert "PEDFLOW_WIFI_AP_PASSWORD" in source
    assert "udp.beginPacket(destination, WIFI_UDP_PORT)" in source
    assert "udp.endPacket()" in source
    assert "WIFI_UDP_BROADCAST" in source
    assert "WiFi.softAPgetStationNum()" in source
    assert "IPAddress(192, 168, 4, host)" in source
    assert "udpPacketFailureCount" in source
    assert "#status,udp_heartbeat" in source
    assert "PEDFLOW_FIRMWARE_VERSION" in source
    assert '"firmware_version"' in source
    assert "4210" in source
    assert "WIFI_AP_PASSWORD[]" not in source


def test_debug_mode_is_usb_only_and_not_wifi_controlled():
    source = SKETCH.read_text(encoding="utf-8")
    lower_source = source.lower()

    assert "isUsbDebugActive" in source
    assert "static_cast<bool>(Serial)" in source
    assert "#status,usb_debug_on" in source
    assert "parsepacket" not in lower_source
    assert "udp.read" not in lower_source
    assert "udp.parsepacket" not in lower_source
