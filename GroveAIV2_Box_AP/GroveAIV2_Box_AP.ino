#include <Seeed_Arduino_SSCMA.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#include <stdio.h>

namespace {
constexpr unsigned long SERIAL_BAUD = 115200;
constexpr unsigned long INFERENCE_INTERVAL_MS = 100;
constexpr unsigned long ERROR_LOG_INTERVAL_MS = 5000;
constexpr uint16_t WIFI_UDP_PORT = 4210;
constexpr size_t STATUS_BUFFER_SIZE = 80;
constexpr size_t CSV_ROW_BUFFER_SIZE = 160;
constexpr const char WIFI_AP_SSID[] = "PedFlowSensor";
constexpr const char WIFI_AP_PASSWORD[] = "pedflow1234";
constexpr const char CSV_HEADER[] =
    "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target";

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

void startWifiAccessPoint()
{
  WiFi.mode(WIFI_AP);

  if (!WiFi.softAPConfig(WIFI_AP_IP, WIFI_AP_GATEWAY, WIFI_AP_SUBNET)) {
    printStatus("error", "wifi_ap_config_failed");
    return;
  }

  if (!WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD)) {
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
  logDetections();
}
