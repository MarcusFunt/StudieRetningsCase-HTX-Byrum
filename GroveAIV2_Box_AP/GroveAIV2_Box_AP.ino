#include <ArduinoJson.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

#include <stdio.h>
#include <string.h>

#include "pedflow_secrets.h"

namespace {
constexpr unsigned long SERIAL_BAUD = 115200;
constexpr unsigned long INFERENCE_INTERVAL_MS = 100;
constexpr unsigned long ERROR_LOG_INTERVAL_MS = 5000;
constexpr unsigned long SSCMA_RESPONSE_TIMEOUT_MS = 1000;
constexpr unsigned long SSCMA_INVOKE_EVENT_TIMEOUT_MS = 5000;
constexpr unsigned long SSCMA_IMAGE_EVENT_TIMEOUT_MS = 15000;
constexpr uint16_t WIFI_UDP_PORT = 4210;
constexpr uint8_t SSCMA_I2C_ADDRESS = 0x62;
constexpr uint32_t SSCMA_I2C_CLOCK = 400000;
constexpr uint8_t SSCMA_I2C_WAIT_DELAY_MS = 2;
constexpr uint8_t SSCMA_MAX_PAYLOAD_LEN = 250;
constexpr size_t STATUS_BUFFER_SIZE = 120;
constexpr size_t CSV_ROW_BUFFER_SIZE = 160;
constexpr size_t USB_COMMAND_BUFFER_SIZE = 64;
constexpr size_t MODULE_TEXT_BUFFER_SIZE = 128;
constexpr size_t SSCMA_RX_BUFFER_SIZE = 64 * 1024;
constexpr const char CSV_HEADER[] =
    "timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target";
constexpr const char USB_CALIBRATION_CAPTURE_COMMAND[] = "CALIB_CAPTURE";
constexpr const char USB_MODULE_INFO_COMMAND[] = "MODULE_INFO";
constexpr const char CALIBRATION_IMAGE_BEGIN_PREFIX[] = "#calibration_image_begin";
constexpr const char CALIBRATION_IMAGE_END[] = "#calibration_image_end";
constexpr const char SSCMA_INVOKE_DETECTIONS_COMMAND[] = "AT+INVOKE=1,0,1\r\n";
constexpr const char SSCMA_INVOKE_IMAGE_COMMAND[] = "AT+INVOKE=1,0,0\r\n";
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
uint32_t frameId = 0;
uint32_t invokeFailureCount = 0;
bool aiReady = false;
char aiId[MODULE_TEXT_BUFFER_SIZE] = "";
char aiName[MODULE_TEXT_BUFFER_SIZE] = "";
char aiInfo[MODULE_TEXT_BUFFER_SIZE] = "";
char sscmaRxBuffer[SSCMA_RX_BUFFER_SIZE];
size_t sscmaRxLength = 0;

struct RawSscmaMessage {
  char *json = nullptr;
  size_t jsonLength = 0;
  size_t consumeLength = 0;
};

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

void printUsbLine(const char *line)
{
  if (usbDebugActive) {
    Serial.println(line);
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

void printBootSummaryUsbOnly()
{
  if (!usbDebugActive) {
    return;
  }

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
  return Wire.endTransmission() == 0;
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
  if (Wire.endTransmission() != 0) {
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
    if (Wire.endTransmission() != 0) {
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
    snprintf(error, errorSize, "write_failed");
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

void resetSscmaParser()
{
  sscmaRxLength = 0;
  sscmaRxBuffer[0] = '\0';
}

bool initializeAiModule()
{
  Wire.begin();
  Wire.setClock(SSCMA_I2C_CLOCK);
  resetSscmaParser();

  if (!queryModuleText(SSCMA_ID_COMMAND, "ID?", nullptr, aiId, sizeof(aiId))) {
    printStatus("error", "ai_begin_failed");
    return false;
  }

  if (!queryModuleText(SSCMA_NAME_COMMAND, "NAME?", nullptr, aiName, sizeof(aiName))) {
    printStatus("error", "ai_begin_failed");
    return false;
  }

  queryModuleText(SSCMA_INFO_COMMAND, "INFO", "info", aiInfo, sizeof(aiInfo));
  printStatus("status", "ai_ready");
  printStatusValue("status", "ai_id", aiId);
  printStatusValue("status", "ai_name", aiName);
  if (aiInfo[0] != '\0') {
    printStatusValue("status", "ai_info", aiInfo);
  }
  return true;
}

void printInvokeError(unsigned long now, const char *stage, const char *reason)
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
  char detail[STATUS_BUFFER_SIZE] = "";

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

bool rawMessageLooksLikeInvokeEvent(const char *json)
{
  return strstr(json, "\"type\":1") != nullptr &&
         strstr(json, "\"name\":\"INVOKE\"") != nullptr &&
         strstr(json, "\"code\":0") != nullptr;
}

const char *findJsonStringValue(const char *json, const char *key)
{
  const char *cursor = strstr(json, key);
  if (cursor == nullptr) {
    return nullptr;
  }
  return cursor + strlen(key);
}

size_t jsonStringDecodedLength(const char *start, const char **endOut)
{
  size_t length = 0;
  const char *cursor = start;
  while (*cursor != '\0') {
    if (*cursor == '"') {
      *endOut = cursor;
      return length;
    }
    if (*cursor == '\\' && cursor[1] != '\0') {
      cursor += 2;
      ++length;
      continue;
    }
    ++cursor;
    ++length;
  }

  *endOut = nullptr;
  return 0;
}

void printJsonStringPayload(const char *start, const char *end)
{
  const char *cursor = start;
  while (cursor < end) {
    if (*cursor == '\\' && cursor + 1 < end) {
      ++cursor;
      Serial.write(*cursor++);
      continue;
    }
    Serial.write(*cursor++);
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

  char error[STATUS_BUFFER_SIZE] = "";
  JsonDocument response;
  if (!sendSscmaCommand(SSCMA_INVOKE_IMAGE_COMMAND, error, sizeof(error)) ||
      !waitForExpectedJson(
          CMD_TYPE_RESPONSE,
          "INVOKE",
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

    if (!rawMessageLooksLikeInvokeEvent(message.json)) {
      consumeRawSscmaMessage(message);
      continue;
    }

    const char *imageStart = findJsonStringValue(message.json, "\"image\":\"");
    if (imageStart == nullptr) {
      consumeRawSscmaMessage(message);
      printUsbOnlyStatus("error", "calibration_capture_empty_image");
      return;
    }

    const char *imageEnd = nullptr;
    const size_t imageLength = jsonStringDecodedLength(imageStart, &imageEnd);
    if (imageEnd == nullptr || imageLength == 0) {
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
    printJsonStringPayload(imageStart, imageEnd);
    Serial.println();
    Serial.println(CALIBRATION_IMAGE_END);
    consumeRawSscmaMessage(message);
    return;
  }

  printUsbOnlyStatus("error", "calibration_capture_timeout");
}

void printModuleInfoUsbOnly()
{
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
    return;
  }

  aiReady = true;
}

void loop()
{
  updateUsbDebugMode();
  handleUsbSerialCommands();
  logDetections();
}
