#include <ArduinoJson.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

#include <stdio.h>
#include <string.h>

#include "pedflow_secrets.h"

namespace {
constexpr unsigned long SERIAL_BAUD = 115200;
constexpr const char PEDFLOW_FIRMWARE_VERSION[] = "0.1.0";
constexpr unsigned long INFERENCE_INTERVAL_MS = 100;
constexpr unsigned long ERROR_LOG_INTERVAL_MS = 5000;
constexpr unsigned long SSCMA_RESPONSE_TIMEOUT_MS = 1000;
constexpr unsigned long SSCMA_INVOKE_EVENT_TIMEOUT_MS = 5000;
constexpr unsigned long SSCMA_IMAGE_EVENT_TIMEOUT_MS = 15000;
constexpr unsigned long AI_INIT_SETTLE_MS = 1500;
constexpr unsigned long AI_INIT_RETRY_INTERVAL_MS = 3000;
constexpr unsigned long UDP_HEARTBEAT_INTERVAL_MS = 1000;
constexpr uint16_t WIFI_UDP_PORT = 4210;
constexpr uint8_t WIFI_AP_CHANNEL = 6;
constexpr bool WIFI_AP_HIDDEN = false;
constexpr uint8_t WIFI_AP_MAX_CLIENTS = 2;
constexpr uint8_t SSCMA_I2C_ADDRESS = 0x62;
constexpr uint32_t SSCMA_I2C_CLOCK = 100000;
constexpr uint8_t SSCMA_I2C_WAIT_DELAY_MS = 2;
constexpr uint8_t SSCMA_I2C_RECOVERY_CLOCKS = 18;
constexpr int32_t SSCMA_RESET_PIN = D3;
constexpr unsigned long SSCMA_RESET_LOW_MS = 50;
constexpr unsigned long SSCMA_RESET_SETTLE_MS = 1500;
constexpr uint8_t SSCMA_MAX_PAYLOAD_LEN = 250;
constexpr size_t STATUS_BUFFER_SIZE = 120;
constexpr size_t INVOKE_ERROR_BUFFER_SIZE = 192;
constexpr size_t SSCMA_ERROR_DETAIL_BUFFER_SIZE = 80;
constexpr size_t CSV_ROW_BUFFER_SIZE = 160;
constexpr size_t USB_COMMAND_BUFFER_SIZE = 64;
constexpr size_t MODULE_TEXT_BUFFER_SIZE = 128;
constexpr size_t SSCMA_RX_BUFFER_SIZE = 64 * 1024;
constexpr const char CSV_HEADER[] =
    "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target";
constexpr const char USB_CALIBRATION_CAPTURE_COMMAND[] = "CALIB_CAPTURE";
constexpr const char USB_MODULE_INFO_COMMAND[] = "MODULE_INFO";
constexpr const char USB_WIFI_STATUS_COMMAND[] = "WIFI_STATUS";
constexpr const char USB_UDP_TEST_COMMAND[] = "UDP_TEST";
constexpr const char USB_RAW_AT_COMMAND_PREFIX[] = "AT:";
constexpr const char CALIBRATION_IMAGE_BEGIN_PREFIX[] = "#calibration_image_begin";
constexpr const char CALIBRATION_IMAGE_END[] = "#calibration_image_end";
constexpr const char SSCMA_INVOKE_DETECTIONS_COMMAND[] = "AT+INVOKE=1,0,1\r\n";
constexpr const char SSCMA_SAMPLE_IMAGE_COMMAND[] = "AT+SAMPLE=1\r\n";
constexpr const char SSCMA_ID_COMMAND[] = "AT+ID?\r\n";
constexpr const char SSCMA_NAME_COMMAND[] = "AT+NAME?\r\n";
constexpr const char SSCMA_INFO_COMMAND[] = "AT+INFO?\r\n";
constexpr const char JSON_RESPONSE_PREFIX[] = "\r{";
constexpr const char JSON_RESPONSE_SUFFIX[] = "}\n";

constexpr uint8_t FEATURE_TRANSPORT = 0x10;
constexpr uint8_t FEATURE_TRANSPORT_CMD_READ = 0x01;
constexpr uint8_t FEATURE_TRANSPORT_CMD_WRITE = 0x02;
constexpr uint8_t FEATURE_TRANSPORT_CMD_AVAILABLE = 0x03;

constexpr int CMD_OK = 0;
constexpr int CMD_TYPE_RESPONSE = 0;
constexpr int CMD_TYPE_EVENT = 1;
constexpr int CMD_TYPE_LOG = 2;

WiFiUDP udp;

const IPAddress WIFI_AP_IP(192, 168, 4, 1);
const IPAddress WIFI_AP_GATEWAY(192, 168, 4, 1);
const IPAddress WIFI_AP_SUBNET(255, 255, 255, 0);
const IPAddress WIFI_UDP_BROADCAST(192, 168, 4, 255);

bool udpReady = false;
bool usbDebugActive = false;
unsigned long lastInferenceMs = 0;
unsigned long lastInvokeErrorLogMs = 0;
unsigned long lastEmptyFrameLogMs = 0;
unsigned long lastUdpHeartbeatMs = 0;
unsigned long lastAiInitAttemptMs = 0;
uint32_t frameId = 0;
uint32_t invokeFailureCount = 0;
uint32_t aiInitAttemptCount = 0;
uint32_t udpPacketAttemptCount = 0;
uint32_t udpPacketSuccessCount = 0;
uint32_t udpPacketFailureCount = 0;
uint32_t udpBeginFailureCount = 0;
uint32_t udpWriteFailureCount = 0;
uint32_t udpEndFailureCount = 0;
uint32_t udpHeartbeatCount = 0;
bool aiReady = false;
uint8_t lastSscmaProbeError = 0;
uint8_t lastSscmaWriteError = 0;
uint8_t lastSscmaAvailableError = 0;
uint8_t lastSscmaReadError = 0;
char aiId[MODULE_TEXT_BUFFER_SIZE] = "";
char aiName[MODULE_TEXT_BUFFER_SIZE] = "";
char aiInfo[MODULE_TEXT_BUFFER_SIZE] = "";
char wifiApDetails[STATUS_BUFFER_SIZE] = "";
char sscmaRxBuffer[SSCMA_RX_BUFFER_SIZE];
size_t sscmaRxLength = 0;

struct RawSscmaMessage {
  char *json = nullptr;
  size_t jsonLength = 0;
  size_t consumeLength = 0;
};

bool sendUdpPacket(const IPAddress &destination, const char *line)
{
  ++udpPacketAttemptCount;
  if (!udp.beginPacket(destination, WIFI_UDP_PORT)) {
    ++udpPacketFailureCount;
    ++udpBeginFailureCount;
    return false;
  }

  const size_t expectedLength = strlen(line);
  const size_t writtenLength = udp.print(line);
  const bool writeSucceeded = writtenLength == expectedLength;
  if (!writeSucceeded) {
    ++udpWriteFailureCount;
  }

  if (!udp.endPacket()) {
    ++udpPacketFailureCount;
    ++udpEndFailureCount;
    return false;
  }

  if (!writeSucceeded) {
    ++udpPacketFailureCount;
    return false;
  }

  ++udpPacketSuccessCount;
  return true;
}

bool sendUdpLine(const char *line)
{
  if (!udpReady) {
    return false;
  }

  return sendUdpPacket(WIFI_UDP_BROADCAST, line);
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

void printStatusValue(const char *level, const char *code, const char *value)
{
  char line[STATUS_BUFFER_SIZE];
  snprintf(line, sizeof(line), "#%s,%s,%s", level, code, value ? value : "");
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

void printUsbOnlyStatusValue(const char *level, const char *code, const char *value)
{
  if (!usbDebugActive) {
    return;
  }

  char line[STATUS_BUFFER_SIZE];
  snprintf(line, sizeof(line), "#%s,%s,%s", level, code, value ? value : "");
  Serial.println(line);
}

void copyText(char *destination, size_t destinationSize, const char *source)
{
  if (destinationSize == 0) {
    return;
  }

  if (source == nullptr) {
    destination[0] = '\0';
    return;
  }

  snprintf(destination, destinationSize, "%s", source);
}

void releaseI2cLine(uint8_t pin)
{
  pinMode(pin, INPUT_PULLUP);
}

void driveI2cLineLow(uint8_t pin)
{
  digitalWrite(pin, LOW);
  pinMode(pin, OUTPUT);
}

void recoverSscmaI2cBus()
{
  Wire.end();
  releaseI2cLine(SDA);
  releaseI2cLine(SCL);
  delay(5);

  for (uint8_t i = 0; i < SSCMA_I2C_RECOVERY_CLOCKS; ++i) {
    driveI2cLineLow(SCL);
    delayMicroseconds(5);
    releaseI2cLine(SCL);
    delayMicroseconds(5);
  }

  driveI2cLineLow(SDA);
  delayMicroseconds(5);
  releaseI2cLine(SCL);
  delayMicroseconds(5);
  releaseI2cLine(SDA);
  delay(5);
}

void resetSscmaModule()
{
  if (SSCMA_RESET_PIN < 0) {
    return;
  }

  pinMode(SSCMA_RESET_PIN, OUTPUT);
  digitalWrite(SSCMA_RESET_PIN, LOW);
  delay(SSCMA_RESET_LOW_MS);
  pinMode(SSCMA_RESET_PIN, INPUT);
  delay(SSCMA_RESET_SETTLE_MS);
}

void printBootSummaryUsbOnly()
{
  if (!usbDebugActive) {
    return;
  }

  printUsbOnlyStatusValue("status", "firmware_version", PEDFLOW_FIRMWARE_VERSION);
  printUsbOnlyStatus("status", udpReady ? "wifi_ap_ready" : "wifi_ap_not_ready");
  printUsbOnlyStatus("status", aiReady ? "ai_ready" : "ai_not_ready");
  if (aiId[0] != '\0') {
    printUsbOnlyStatusValue("status", "ai_id", aiId);
  }
  if (aiName[0] != '\0') {
    printUsbOnlyStatusValue("status", "ai_name", aiName);
  }
  if (aiInfo[0] != '\0') {
    printUsbOnlyStatusValue("status", "ai_info", aiInfo);
  }
  if (wifiApDetails[0] != '\0') {
    Serial.println(wifiApDetails);
  }
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
    printBootSummaryUsbOnly();
  }
}

void startWifiAccessPoint()
{
  WiFi.mode(WIFI_AP);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  if (!WiFi.softAPConfig(WIFI_AP_IP, WIFI_AP_GATEWAY, WIFI_AP_SUBNET)) {
    printStatus("error", "wifi_ap_config_failed");
    return;
  }

  if (!WiFi.softAP(
          PEDFLOW_WIFI_AP_SSID,
          PEDFLOW_WIFI_AP_PASSWORD,
          WIFI_AP_CHANNEL,
          WIFI_AP_HIDDEN,
          WIFI_AP_MAX_CLIENTS)) {
    printStatus("error", "wifi_ap_start_failed");
    return;
  }

  if (udp.begin(WIFI_UDP_PORT) == 0) {
    printStatus("error", "udp_begin_failed");
    return;
  }

  udpReady = true;
  printStatusValue("status", "firmware_version", PEDFLOW_FIRMWARE_VERSION);
  printStatus("status", "wifi_ap_ready");
  snprintf(
      wifiApDetails,
      sizeof(wifiApDetails),
      "#status,wifi_ap_details,%s,%s,%u",
      WiFi.softAPIP().toString().c_str(),
      WiFi.softAPmacAddress().c_str(),
      static_cast<unsigned int>(WIFI_AP_CHANNEL));
  emitTelemetryLine(wifiApDetails);
}

bool sscmaWritePacket(const char *data, uint8_t length)
{
  Wire.beginTransmission(SSCMA_I2C_ADDRESS);
  Wire.write(FEATURE_TRANSPORT);
  Wire.write(FEATURE_TRANSPORT_CMD_WRITE);
  Wire.write(static_cast<uint8_t>(length >> 8));
  Wire.write(static_cast<uint8_t>(length & 0xFF));
  Wire.write(reinterpret_cast<const uint8_t *>(data), length);
  Wire.write(static_cast<uint8_t>(0));
  Wire.write(static_cast<uint8_t>(0));
  lastSscmaWriteError = Wire.endTransmission();
  return lastSscmaWriteError == 0;
}

bool sscmaWrite(const char *data, size_t length)
{
  size_t offset = 0;
  while (offset < length) {
    const uint8_t chunkLength = static_cast<uint8_t>(
        min(static_cast<size_t>(SSCMA_MAX_PAYLOAD_LEN), length - offset));
    if (!sscmaWritePacket(data + offset, chunkLength)) {
      return false;
    }
    offset += chunkLength;
    delay(SSCMA_I2C_WAIT_DELAY_MS);
  }

  return true;
}

int sscmaAvailable()
{
  uint8_t data[2] = {0, 0};

  delay(SSCMA_I2C_WAIT_DELAY_MS);
  Wire.beginTransmission(SSCMA_I2C_ADDRESS);
  Wire.write(FEATURE_TRANSPORT);
  Wire.write(FEATURE_TRANSPORT_CMD_AVAILABLE);
  Wire.write(static_cast<uint8_t>(0));
  Wire.write(static_cast<uint8_t>(0));
  Wire.write(static_cast<uint8_t>(0));
  Wire.write(static_cast<uint8_t>(0));
  lastSscmaAvailableError = Wire.endTransmission();
  if (lastSscmaAvailableError != 0) {
    return 0;
  }

  delay(SSCMA_I2C_WAIT_DELAY_MS);
  const uint8_t received = Wire.requestFrom(SSCMA_I2C_ADDRESS, static_cast<uint8_t>(2));
  if (received != 2) {
    return 0;
  }

  data[0] = static_cast<uint8_t>(Wire.read());
  data[1] = static_cast<uint8_t>(Wire.read());
  return (static_cast<int>(data[0]) << 8) | data[1];
}

int sscmaRead(char *data, int length)
{
  int totalRead = 0;

  while (totalRead < length) {
    const uint8_t chunkLength = static_cast<uint8_t>(
        min(static_cast<int>(SSCMA_MAX_PAYLOAD_LEN), length - totalRead));

    delay(SSCMA_I2C_WAIT_DELAY_MS);
    Wire.beginTransmission(SSCMA_I2C_ADDRESS);
    Wire.write(FEATURE_TRANSPORT);
    Wire.write(FEATURE_TRANSPORT_CMD_READ);
    Wire.write(static_cast<uint8_t>(chunkLength >> 8));
    Wire.write(static_cast<uint8_t>(chunkLength & 0xFF));
    Wire.write(static_cast<uint8_t>(0));
    Wire.write(static_cast<uint8_t>(0));
    lastSscmaReadError = Wire.endTransmission();
    if (lastSscmaReadError != 0) {
      break;
    }

    delay(SSCMA_I2C_WAIT_DELAY_MS);
    const uint8_t requested = Wire.requestFrom(SSCMA_I2C_ADDRESS, chunkLength);
    int chunkRead = 0;
    while (Wire.available() > 0 && chunkRead < requested) {
      data[totalRead++] = static_cast<char>(Wire.read());
      ++chunkRead;
    }

    if (chunkRead == 0 || chunkRead < chunkLength) {
      break;
    }
  }

  return totalRead;
}

void drainSscmaBytes(int byteCount)
{
  char discard[64];
  int remaining = byteCount;
  while (remaining > 0) {
    const int chunk = min(remaining, static_cast<int>(sizeof(discard)));
    const int readCount = sscmaRead(discard, chunk);
    if (readCount <= 0) {
      return;
    }
    remaining -= readCount;
  }
}

void resetSscmaParser()
{
  sscmaRxLength = 0;
  sscmaRxBuffer[0] = '\0';
}

void flushSscmaInput()
{
  int available = sscmaAvailable();
  while (available > 0) {
    drainSscmaBytes(available);
    delay(SSCMA_I2C_WAIT_DELAY_MS);
    available = sscmaAvailable();
  }
  resetSscmaParser();
}

bool fillSscmaRxBuffer(char *error, size_t errorSize)
{
  int available = sscmaAvailable();
  while (available > 0) {
    if (sscmaRxLength + static_cast<size_t>(available) >= sizeof(sscmaRxBuffer)) {
      drainSscmaBytes(available);
      sscmaRxLength = 0;
      sscmaRxBuffer[0] = '\0';
      snprintf(error, errorSize, "response_oversized");
      return false;
    }

    const int readCount = sscmaRead(
        sscmaRxBuffer + sscmaRxLength,
        min(available, static_cast<int>(sizeof(sscmaRxBuffer) - sscmaRxLength - 1)));
    if (readCount <= 0) {
      snprintf(error, errorSize, "read_failed");
      return false;
    }

    sscmaRxLength += static_cast<size_t>(readCount);
    sscmaRxBuffer[sscmaRxLength] = '\0';
    available -= readCount;
  }

  return true;
}

bool findRawSscmaMessage(RawSscmaMessage &message, char *error, size_t errorSize)
{
  char *prefix = strstr(sscmaRxBuffer, JSON_RESPONSE_PREFIX);
  if (prefix == nullptr) {
    if (sscmaRxLength > sizeof(JSON_RESPONSE_PREFIX)) {
      const size_t keep = sizeof(JSON_RESPONSE_PREFIX) - 1;
      memmove(sscmaRxBuffer, sscmaRxBuffer + sscmaRxLength - keep, keep);
      sscmaRxLength = keep;
      sscmaRxBuffer[sscmaRxLength] = '\0';
    }
    return false;
  }

  if (prefix != sscmaRxBuffer) {
    const size_t discardLength = static_cast<size_t>(prefix - sscmaRxBuffer);
    memmove(sscmaRxBuffer, prefix, sscmaRxLength - discardLength);
    sscmaRxLength -= discardLength;
    sscmaRxBuffer[sscmaRxLength] = '\0';
    prefix = sscmaRxBuffer;
  }

  char *suffix = strstr(prefix + 1, JSON_RESPONSE_SUFFIX);
  if (suffix == nullptr) {
    if (sscmaRxLength + 1 >= sizeof(sscmaRxBuffer)) {
      sscmaRxLength = 0;
      sscmaRxBuffer[0] = '\0';
      snprintf(error, errorSize, "response_oversized");
      return false;
    }
    return false;
  }

  message.json = prefix + 1;
  message.jsonLength = static_cast<size_t>(suffix - message.json) + 1;
  message.consumeLength = static_cast<size_t>((suffix + strlen(JSON_RESPONSE_SUFFIX)) - sscmaRxBuffer);
  message.json[message.jsonLength] = '\0';
  return true;
}

void consumeRawSscmaMessage(const RawSscmaMessage &message)
{
  if (message.consumeLength >= sscmaRxLength) {
    sscmaRxLength = 0;
    sscmaRxBuffer[0] = '\0';
    return;
  }

  memmove(
      sscmaRxBuffer,
      sscmaRxBuffer + message.consumeLength,
      sscmaRxLength - message.consumeLength);
  sscmaRxLength -= message.consumeLength;
  sscmaRxBuffer[sscmaRxLength] = '\0';
}

bool waitForRawSscmaMessage(
    RawSscmaMessage &message,
    unsigned long timeoutMs,
    char *error,
    size_t errorSize)
{
  const unsigned long startMs = millis();
  while (millis() - startMs <= timeoutMs) {
    if (!fillSscmaRxBuffer(error, errorSize)) {
      return false;
    }

    if (findRawSscmaMessage(message, error, errorSize)) {
      return true;
    }

    delay(SSCMA_I2C_WAIT_DELAY_MS);
  }

  snprintf(error, errorSize, "timeout");
  return false;
}

bool waitForExpectedJson(
    int expectedType,
    const char *expectedName,
    unsigned long timeoutMs,
    JsonDocument &document,
    char *error,
    size_t errorSize)
{
  const unsigned long startMs = millis();
  while (millis() - startMs <= timeoutMs) {
    RawSscmaMessage message;
    const unsigned long elapsedMs = millis() - startMs;
    const unsigned long remainingMs = elapsedMs >= timeoutMs ? 0 : timeoutMs - elapsedMs;
    if (!waitForRawSscmaMessage(message, remainingMs, error, errorSize)) {
      return false;
    }

    document.clear();
    const DeserializationError parseError =
        deserializeJson(document, static_cast<const char *>(message.json));
    consumeRawSscmaMessage(message);
    if (parseError) {
      snprintf(error, errorSize, "json_parse_failed");
      return false;
    }

    const int type = document["type"] | -1;
    const int code = document["code"] | -1;
    const char *name = document["name"] | "";
    if (type == CMD_TYPE_LOG) {
      continue;
    }

    if (type == expectedType && strcmp(name, expectedName) == 0) {
      if (code == CMD_OK) {
        return true;
      }

      snprintf(error, errorSize, "response_code_%d", code);
      return false;
    }
  }

  snprintf(error, errorSize, "timeout");
  return false;
}

bool sendSscmaCommand(const char *command, char *error, size_t errorSize)
{
  if (!sscmaWrite(command, strlen(command))) {
    snprintf(error, errorSize, "write_failed_%u", static_cast<unsigned int>(lastSscmaWriteError));
    return false;
  }

  return true;
}

bool probeSscmaDevice(char *error, size_t errorSize)
{
  Wire.beginTransmission(SSCMA_I2C_ADDRESS);
  lastSscmaProbeError = Wire.endTransmission();
  if (lastSscmaProbeError != 0) {
    snprintf(
        error,
        errorSize,
        "i2c_probe_failed_%u",
        static_cast<unsigned int>(lastSscmaProbeError));
    return false;
  }

  return true;
}

bool queryModuleText(
    const char *command,
    const char *expectedName,
    const char *nestedKey,
    char *destination,
    size_t destinationSize)
{
  char error[STATUS_BUFFER_SIZE] = "";
  JsonDocument response;

  if (!sendSscmaCommand(command, error, sizeof(error))) {
    printStatusValue("error", "ai_query_failed", error);
    return false;
  }

  if (!waitForExpectedJson(
          CMD_TYPE_RESPONSE,
          expectedName,
          SSCMA_RESPONSE_TIMEOUT_MS,
          response,
          error,
          sizeof(error))) {
    printStatusValue("error", "ai_query_failed", error);
    return false;
  }

  const char *value = nullptr;
  if (nestedKey != nullptr) {
    value = response["data"][nestedKey] | "";
  } else {
    value = response["data"] | "";
  }

  copyText(destination, destinationSize, value);
  return destination[0] != '\0';
}

bool initializeAiModule()
{
  lastAiInitAttemptMs = millis();
  ++aiInitAttemptCount;
  aiId[0] = '\0';
  aiName[0] = '\0';
  aiInfo[0] = '\0';

  recoverSscmaI2cBus();
  Wire.begin(SDA, SCL);
  Wire.setClock(SSCMA_I2C_CLOCK);
  resetSscmaModule();
  resetSscmaParser();

  const unsigned long now = millis();
  if (now < AI_INIT_SETTLE_MS) {
    delay(AI_INIT_SETTLE_MS - now);
  }

  char error[STATUS_BUFFER_SIZE] = "";
  if (!probeSscmaDevice(error, sizeof(error))) {
    printStatusValue("error", "ai_query_failed", error);
    printStatus("error", "ai_begin_failed");
    return false;
  }

  if (!queryModuleText(SSCMA_ID_COMMAND, "ID?", nullptr, aiId, sizeof(aiId))) {
    printStatus("error", "ai_begin_failed");
    return false;
  }

  if (!queryModuleText(SSCMA_NAME_COMMAND, "NAME?", nullptr, aiName, sizeof(aiName))) {
    printStatus("error", "ai_begin_failed");
    return false;
  }

  queryModuleText(SSCMA_INFO_COMMAND, "INFO?", "info", aiInfo, sizeof(aiInfo));
  printStatus("status", "ai_ready");
  printStatusValue("status", "ai_id", aiId);
  printStatusValue("status", "ai_name", aiName);
  if (aiInfo[0] != '\0') {
    printStatusValue("status", "ai_info", aiInfo);
  }
  return true;
}

void retryAiInitialization()
{
  if (aiReady) {
    return;
  }

  const unsigned long now = millis();
  if (lastAiInitAttemptMs != 0 && now - lastAiInitAttemptMs < AI_INIT_RETRY_INTERVAL_MS) {
    return;
  }

  char line[STATUS_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "#status,ai_init_retry,%lu",
      static_cast<unsigned long>(aiInitAttemptCount + 1));
  emitTelemetryLine(line);
  aiReady = initializeAiModule();
}

void printInvokeError(unsigned long now, const char *stage, const char *reason)
{
  ++invokeFailureCount;
  if (lastInvokeErrorLogMs != 0 && now - lastInvokeErrorLogMs < ERROR_LOG_INTERVAL_MS) {
    return;
  }

  lastInvokeErrorLogMs = now;
  char line[INVOKE_ERROR_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "#error,ai_invoke_failed,%s,%s,%lu",
      stage,
      reason,
      static_cast<unsigned long>(invokeFailureCount));
  emitTelemetryLine(line);
}

void printCsvRow(
    unsigned long timestampMs,
    uint32_t currentFrameId,
    uint16_t detectionId,
    float centerX,
    float centerY,
    float width,
    float height,
    float score,
    int target)
{
  const float bboxX = centerX - width / 2.0f;
  const float bboxY = centerY - height / 2.0f;
  const float confidence = score / 100.0f;

  char line[CSV_ROW_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "%lu,%lu,%u,%.2f,%.2f,%.2f,%.2f,%.3f,%d",
      timestampMs,
      static_cast<unsigned long>(currentFrameId),
      static_cast<unsigned int>(detectionId),
      static_cast<double>(bboxX),
      static_cast<double>(bboxY),
      static_cast<double>(width),
      static_cast<double>(height),
      static_cast<double>(confidence),
      target);
  emitTelemetryLine(line);
}

bool invokeForDetections(JsonDocument &event, char *error, size_t errorSize)
{
  JsonDocument response;
  char detail[SSCMA_ERROR_DETAIL_BUFFER_SIZE] = "";

  if (!sendSscmaCommand(SSCMA_INVOKE_DETECTIONS_COMMAND, detail, sizeof(detail))) {
    snprintf(error, errorSize, "write_%s", detail);
    return false;
  }

  if (!waitForExpectedJson(
          CMD_TYPE_RESPONSE,
          "INVOKE",
          SSCMA_RESPONSE_TIMEOUT_MS,
          response,
          detail,
          sizeof(detail))) {
    snprintf(error, errorSize, "response_%s", detail);
    return false;
  }

  if (!waitForExpectedJson(
          CMD_TYPE_EVENT,
          "INVOKE",
          SSCMA_INVOKE_EVENT_TIMEOUT_MS,
          event,
          detail,
          sizeof(detail))) {
    snprintf(error, errorSize, "event_%s", detail);
    return false;
  }

  return true;
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

  char error[STATUS_BUFFER_SIZE] = "";
  JsonDocument event;
  if (!invokeForDetections(event, error, sizeof(error))) {
    printInvokeError(now, "invoke", error);
    return;
  }

  const uint32_t currentFrameId = frameId++;
  uint16_t detectionId = 0;
  JsonArrayConst boxes = event["data"]["boxes"].as<JsonArrayConst>();
  for (JsonArrayConst box : boxes) {
    const float centerX = box[0] | 0.0f;
    const float centerY = box[1] | 0.0f;
    const float width = box[2] | 0.0f;
    const float height = box[3] | 0.0f;
    const float score = box[4] | 0.0f;
    const int target = box[5] | 0;
    printCsvRow(now, currentFrameId, detectionId++, centerX, centerY, width, height, score, target);
  }

  if (detectionId == 0 && (lastEmptyFrameLogMs == 0 || now - lastEmptyFrameLogMs >= ERROR_LOG_INTERVAL_MS)) {
    lastEmptyFrameLogMs = now;
    char line[STATUS_BUFFER_SIZE];
    snprintf(
        line,
        sizeof(line),
        "#status,ai_invoke_ok_no_detections,%lu",
        static_cast<unsigned long>(currentFrameId));
    emitTelemetryLine(line);
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

  printUsbOnlyStatusValue("status", "firmware_version", PEDFLOW_FIRMWARE_VERSION);
  printUsbOnlyStatus("status", "calibration_capture_started");

  char error[STATUS_BUFFER_SIZE] = "";
  JsonDocument response;
  flushSscmaInput();
  if (!sendSscmaCommand(SSCMA_SAMPLE_IMAGE_COMMAND, error, sizeof(error)) ||
      !waitForExpectedJson(
          CMD_TYPE_RESPONSE,
          "SAMPLE",
          SSCMA_RESPONSE_TIMEOUT_MS,
          response,
          error,
          sizeof(error))) {
    printUsbOnlyStatusValue("error", "calibration_capture_failed", error);
    return;
  }

  const unsigned long startMs = millis();
  while (millis() - startMs <= SSCMA_IMAGE_EVENT_TIMEOUT_MS) {
    RawSscmaMessage message;
    const unsigned long elapsedMs = millis() - startMs;
    const unsigned long remainingMs =
        elapsedMs >= SSCMA_IMAGE_EVENT_TIMEOUT_MS ? 0 : SSCMA_IMAGE_EVENT_TIMEOUT_MS - elapsedMs;
    if (!waitForRawSscmaMessage(message, remainingMs, error, sizeof(error))) {
      printUsbOnlyStatusValue("error", "calibration_capture_failed", error);
      return;
    }

    JsonDocument sampleEvent;
    const DeserializationError parseError =
        deserializeJson(sampleEvent, static_cast<const char *>(message.json));
    if (parseError) {
      consumeRawSscmaMessage(message);
      continue;
    }

    const int type = sampleEvent["type"] | -1;
    const int code = sampleEvent["code"] | -1;
    const char *name = sampleEvent["name"] | "";
    if (type != CMD_TYPE_EVENT || code != CMD_OK || strcmp(name, "SAMPLE") != 0) {
      consumeRawSscmaMessage(message);
      continue;
    }

    const char *image = sampleEvent["data"]["image"] | "";
    const size_t imageLength = strlen(image);
    if (imageLength == 0) {
      consumeRawSscmaMessage(message);
      printUsbOnlyStatus("error", "calibration_capture_empty_image");
      return;
    }

    char beginLine[STATUS_BUFFER_SIZE];
    snprintf(
        beginLine,
        sizeof(beginLine),
        "%s,%lu",
        CALIBRATION_IMAGE_BEGIN_PREFIX,
        static_cast<unsigned long>(imageLength));
    Serial.println(beginLine);
    Serial.println(image);
    Serial.println(CALIBRATION_IMAGE_END);
    consumeRawSscmaMessage(message);
    return;
  }

  printUsbOnlyStatus("error", "calibration_capture_timeout");
}

void printModuleInfoUsbOnly()
{
  printUsbOnlyStatusValue("status", "firmware_version", PEDFLOW_FIRMWARE_VERSION);
  printUsbOnlyStatus("status", aiReady ? "ai_ready" : "ai_not_ready");
  char line[STATUS_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "#status,ai_init_attempts,%lu",
      static_cast<unsigned long>(aiInitAttemptCount));
  Serial.println(line);
  snprintf(
      line,
      sizeof(line),
      "#status,ai_i2c_errors,%u,%u,%u,%u",
      static_cast<unsigned int>(lastSscmaProbeError),
      static_cast<unsigned int>(lastSscmaWriteError),
      static_cast<unsigned int>(lastSscmaAvailableError),
      static_cast<unsigned int>(lastSscmaReadError));
  Serial.println(line);
  if (aiId[0] != '\0') {
    printUsbOnlyStatusValue("status", "ai_id", aiId);
  }
  if (aiName[0] != '\0') {
    printUsbOnlyStatusValue("status", "ai_name", aiName);
  }
  if (aiInfo[0] != '\0') {
    printUsbOnlyStatusValue("status", "ai_info", aiInfo);
  }
}

void printWifiStatusUsbOnly()
{
  printUsbOnlyStatus("status", udpReady ? "wifi_ap_ready" : "wifi_ap_not_ready");
  if (wifiApDetails[0] != '\0') {
    Serial.println(wifiApDetails);
  }
  char line[STATUS_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "#status,wifi_ap_runtime,%s,%s,%d,%u",
      WiFi.softAPSSID().c_str(),
      WiFi.softAPIP().toString().c_str(),
      static_cast<int>(WiFi.channel()),
      static_cast<unsigned int>(WiFi.softAPgetStationNum()));
  Serial.println(line);

  snprintf(
      line,
      sizeof(line),
      "#status,udp_runtime,%lu,%lu,%lu,%lu,%lu,%lu,%lu",
      static_cast<unsigned long>(udpPacketAttemptCount),
      static_cast<unsigned long>(udpPacketSuccessCount),
      static_cast<unsigned long>(udpPacketFailureCount),
      static_cast<unsigned long>(udpBeginFailureCount),
      static_cast<unsigned long>(udpWriteFailureCount),
      static_cast<unsigned long>(udpEndFailureCount),
      static_cast<unsigned long>(udpHeartbeatCount));
  Serial.println(line);
}

void sendUdpTestUsbOnly()
{
  printUsbOnlyStatus("status", "udp_test_started");
  const uint32_t startAttempts = udpPacketAttemptCount;
  const uint32_t startSuccesses = udpPacketSuccessCount;
  const uint32_t startFailures = udpPacketFailureCount;
  for (uint8_t i = 0; i < 10; ++i) {
    char line[STATUS_BUFFER_SIZE];
    snprintf(
        line,
        sizeof(line),
        "#status,udp_test,%u,%u",
        static_cast<unsigned int>(i),
        static_cast<unsigned int>(WiFi.softAPgetStationNum()));
    sendUdpLine(line);
    delay(100);
  }
  char line[STATUS_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "#status,udp_test_result,%lu,%lu,%lu",
      static_cast<unsigned long>(udpPacketAttemptCount - startAttempts),
      static_cast<unsigned long>(udpPacketSuccessCount - startSuccesses),
      static_cast<unsigned long>(udpPacketFailureCount - startFailures));
  Serial.println(line);
  printUsbOnlyStatus("status", "udp_test_done");
}

void sendUdpHeartbeat()
{
  if (!udpReady) {
    return;
  }

  const unsigned long now = millis();
  if (now - lastUdpHeartbeatMs < UDP_HEARTBEAT_INTERVAL_MS) {
    return;
  }

  lastUdpHeartbeatMs = now;
  char line[STATUS_BUFFER_SIZE];
  snprintf(
      line,
      sizeof(line),
      "#status,udp_heartbeat,%lu,%u,%lu,%lu,%lu,%s",
      static_cast<unsigned long>(udpHeartbeatCount++),
      static_cast<unsigned int>(WiFi.softAPgetStationNum()),
      static_cast<unsigned long>(udpPacketAttemptCount),
      static_cast<unsigned long>(udpPacketSuccessCount),
      static_cast<unsigned long>(udpPacketFailureCount),
      PEDFLOW_FIRMWARE_VERSION);
  sendUdpLine(line);
}

void printRawAtResponseUsbOnly(const char *commandBody)
{
  if (!usbDebugActive) {
    return;
  }

  char command[USB_COMMAND_BUFFER_SIZE + 6];
  if (strncmp(commandBody, "AT+", 3) == 0) {
    snprintf(command, sizeof(command), "%s\r\n", commandBody);
  } else {
    snprintf(command, sizeof(command), "AT+%s\r\n", commandBody);
  }

  char error[STATUS_BUFFER_SIZE] = "";
  flushSscmaInput();
  if (!sendSscmaCommand(command, error, sizeof(error))) {
    printUsbOnlyStatusValue("error", "raw_at_failed", error);
    return;
  }

  for (uint8_t i = 0; i < 4; ++i) {
    RawSscmaMessage message;
    if (!waitForRawSscmaMessage(message, 3000, error, sizeof(error))) {
      if (i == 0) {
        printUsbOnlyStatusValue("error", "raw_at_failed", error);
      }
      return;
    }
    Serial.print("#raw_at,");
    Serial.println(message.json);
    consumeRawSscmaMessage(message);
  }
}

void handleUsbCommand(const char *command)
{
  if (strcmp(command, USB_CALIBRATION_CAPTURE_COMMAND) == 0) {
    captureCalibrationImageUsbOnly();
    return;
  }

  if (strcmp(command, USB_MODULE_INFO_COMMAND) == 0) {
    printModuleInfoUsbOnly();
    return;
  }

  if (strcmp(command, USB_WIFI_STATUS_COMMAND) == 0) {
    printWifiStatusUsbOnly();
    return;
  }

  if (strcmp(command, USB_UDP_TEST_COMMAND) == 0) {
    sendUdpTestUsbOnly();
    return;
  }

  if (strncmp(command, USB_RAW_AT_COMMAND_PREFIX, strlen(USB_RAW_AT_COMMAND_PREFIX)) == 0) {
    printRawAtResponseUsbOnly(command + strlen(USB_RAW_AT_COMMAND_PREFIX));
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

  if (!initializeAiModule()) {
    aiReady = false;
  } else {
    aiReady = true;
  }
}

void loop()
{
  updateUsbDebugMode();
  handleUsbSerialCommands();
  sendUdpHeartbeat();
  retryAiInitialization();
  logDetections();
}
