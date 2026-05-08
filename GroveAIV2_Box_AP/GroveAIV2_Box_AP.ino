#include <Seeed_Arduino_SSCMA.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#include <stdio.h>
#include <string.h>

#include "pedflow_secrets.h"

namespace {
constexpr unsigned long SERIAL_BAUD = 115200;
constexpr unsigned long INFERENCE_INTERVAL_MS = 100;
constexpr unsigned long ERROR_LOG_INTERVAL_MS = 5000;
constexpr uint16_t WIFI_UDP_PORT = 4210;
constexpr size_t STATUS_BUFFER_SIZE = 80;
constexpr size_t CSV_ROW_BUFFER_SIZE = 160;
constexpr size_t USB_COMMAND_BUFFER_SIZE = 48;
constexpr const char CSV_HEADER[] =
    "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target";
constexpr const char USB_CALIBRATION_CAPTURE_COMMAND[] = "CALIB_CAPTURE";
constexpr const char CALIBRATION_IMAGE_BEGIN_PREFIX[] = "#calibration_image_begin";
constexpr const char CALIBRATION_IMAGE_END[] = "#calibration_image_end";

SSCMA AI;
WiFiUDP udp;

const IPAddress WIFI_AP_IP(192, 168, 4, 1);
const IPAddress WIFI_AP_GATEWAY(192, 168, 4, 1);
const IPAddress WIFI_AP_SUBNET(255, 255, 255, 0);
const IPAddress WIFI_UDP_BROADCAST(192, 168, 4, 255);

bool udpReady = false;
bool usbDebugActive = false;
unsigned long lastInferenceMs = 0;
unsigned long lastInvokeErrorLogMs = 0;
uint32_t frameId = 0;
uint32_t invokeFailureCount = 0;
bool aiReady = false;

void sendUdpLine(const char *line)
{
  if (!udpReady) {
    return;
  }

  udp.beginPacket(WIFI_UDP_BROADCAST, WIFI_UDP_PORT);
  udp.print(line);
  udp.endPacket();
}

bool isUsbDebugActive()
{
  return static_cast<bool>(Serial);
}

void emitTelemetryLine(const char *line)
{
  if (usbDebugActive) {
    Serial.println(line);
  }
  sendUdpLine(line);
}

void updateUsbDebugMode()
{
  const bool active = isUsbDebugActive();
  if (active == usbDebugActive) {
    return;
  }

  usbDebugActive = active;
  if (usbDebugActive) {
    Serial.println("#status,usb_debug_on");
    Serial.println(CSV_HEADER);
  }
}

void printCsvHeader()
{
  emitTelemetryLine(CSV_HEADER);
}

void printStatus(const char *level, const char *code)
{
  char line[STATUS_BUFFER_SIZE];
  snprintf(line, sizeof(line), "#%s,%s", level, code);
  emitTelemetryLine(line);
}

void printUsbOnlyStatus(const char *level, const char *code)
{
  if (!usbDebugActive) {
    return;
  }

  char line[STATUS_BUFFER_SIZE];
  snprintf(line, sizeof(line), "#%s,%s", level, code);
  Serial.println(line);
}

void startWifiAccessPoint()
{
  WiFi.mode(WIFI_AP);

  if (!WiFi.softAPConfig(WIFI_AP_IP, WIFI_AP_GATEWAY, WIFI_AP_SUBNET)) {
    printStatus("error", "wifi_ap_config_failed");
    return;
  }

  if (!WiFi.softAP(PEDFLOW_WIFI_AP_SSID, PEDFLOW_WIFI_AP_PASSWORD)) {
    printStatus("error", "wifi_ap_start_failed");
    return;
  }

  if (udp.begin(WIFI_UDP_PORT) == 0) {
    printStatus("error", "udp_begin_failed");
    return;
  }

  udpReady = true;
  printStatus("status", "wifi_ap_ready");
}

void printInvokeError(unsigned long now, int result)
{
  ++invokeFailureCount;
  if (lastInvokeErrorLogMs != 0 && now - lastInvokeErrorLogMs < ERROR_LOG_INTERVAL_MS) {
    return;
  }

  lastInvokeErrorLogMs = now;
  char line[STATUS_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "#error,ai_invoke_failed,%d,%lu",
      result,
      static_cast<unsigned long>(invokeFailureCount));
  emitTelemetryLine(line);
}

void printCsvRow(
    unsigned long timestampMs,
    uint32_t currentFrameId,
    uint16_t detectionId,
    const boxes_t &box)
{
  const float bboxX = static_cast<float>(box.x) - static_cast<float>(box.w) / 2.0f;
  const float bboxY = static_cast<float>(box.y) - static_cast<float>(box.h) / 2.0f;
  const float confidence = static_cast<float>(box.score) / 100.0f;

  char line[CSV_ROW_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "%lu,%lu,%u,%.2f,%.2f,%d,%d,%.3f,%d",
      timestampMs,
      static_cast<unsigned long>(currentFrameId),
      static_cast<unsigned int>(detectionId),
      static_cast<double>(bboxX),
      static_cast<double>(bboxY),
      static_cast<int>(box.w),
      static_cast<int>(box.h),
      static_cast<double>(confidence),
      static_cast<int>(box.target));
  emitTelemetryLine(line);
}

void logDetections()
{
  if (!aiReady) {
    return;
  }

  const unsigned long now = millis();

  if (now - lastInferenceMs < INFERENCE_INTERVAL_MS) {
    return;
  }

  lastInferenceMs = now;

  const int invokeResult = AI.invoke(1, false, false);
  if (invokeResult != 0) {
    printInvokeError(now, invokeResult);
    return;
  }

  const uint32_t currentFrameId = frameId++;

  for (size_t i = 0; i < AI.boxes().size(); ++i) {
    printCsvRow(now, currentFrameId, static_cast<uint16_t>(i), AI.boxes()[i]);
  }
}

void captureCalibrationImageUsbOnly()
{
  if (!usbDebugActive) {
    return;
  }

  if (!aiReady) {
    printUsbOnlyStatus("error", "calibration_capture_ai_not_ready");
    return;
  }

  printUsbOnlyStatus("status", "calibration_capture_started");

  const int invokeResult = AI.invoke(1, false, true);
  if (invokeResult != 0) {
    char line[STATUS_BUFFER_SIZE];
    snprintf(line, sizeof(line), "#error,calibration_capture_failed,%d", invokeResult);
    Serial.println(line);
    return;
  }

  const String image = AI.last_image();
  if (image.length() == 0) {
    printUsbOnlyStatus("error", "calibration_capture_empty_image");
    return;
  }

  char beginLine[STATUS_BUFFER_SIZE];
  snprintf(
      beginLine,
      sizeof(beginLine),
      "%s,%lu",
      CALIBRATION_IMAGE_BEGIN_PREFIX,
      static_cast<unsigned long>(image.length()));
  Serial.println(beginLine);
  Serial.println(image);
  Serial.println(CALIBRATION_IMAGE_END);
}

void handleUsbCommand(const char *command)
{
  if (strcmp(command, USB_CALIBRATION_CAPTURE_COMMAND) == 0) {
    captureCalibrationImageUsbOnly();
    return;
  }

  printUsbOnlyStatus("error", "unknown_usb_command");
}

void handleUsbSerialCommands()
{
  if (!usbDebugActive) {
    return;
  }

  static char commandBuffer[USB_COMMAND_BUFFER_SIZE];
  static size_t commandLength = 0;

  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\r') {
      continue;
    }

    if (ch == '\n') {
      commandBuffer[commandLength] = '\0';
      if (commandLength > 0) {
        handleUsbCommand(commandBuffer);
      }
      commandLength = 0;
      continue;
    }

    if (commandLength + 1 >= sizeof(commandBuffer)) {
      commandLength = 0;
      printUsbOnlyStatus("error", "usb_command_too_long");
      continue;
    }

    commandBuffer[commandLength++] = ch;
  }
}
}  // namespace

void setup()
{
  Serial.begin(SERIAL_BAUD);
  updateUsbDebugMode();
  startWifiAccessPoint();
  printCsvHeader();
  if (!AI.begin()) {
    printStatus("error", "ai_begin_failed");
    return;
  }

  aiReady = true;
  printStatus("status", "ai_ready");
}

void loop()
{
  updateUsbDebugMode();
  handleUsbSerialCommands();
  logDetections();
}
