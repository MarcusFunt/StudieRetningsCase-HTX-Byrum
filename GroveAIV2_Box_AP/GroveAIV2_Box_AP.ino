#include <Seeed_Arduino_SSCMA.h>

namespace {
constexpr unsigned long SERIAL_BAUD = 115200;
constexpr unsigned long INFERENCE_INTERVAL_MS = 100;

SSCMA AI;

unsigned long lastInferenceMs = 0;
uint32_t frameId = 0;

void printCsvHeader()
{
  Serial.println("timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target");
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

  Serial.print(timestampMs);
  Serial.print(",");
  Serial.print(currentFrameId);
  Serial.print(",");
  Serial.print(detectionId);
  Serial.print(",");
  Serial.print(bboxX, 2);
  Serial.print(",");
  Serial.print(bboxY, 2);
  Serial.print(",");
  Serial.print(box.w);
  Serial.print(",");
  Serial.print(box.h);
  Serial.print(",");
  Serial.print(confidence, 3);
  Serial.print(",");
  Serial.println(box.target);
}

void logDetections()
{
  const unsigned long now = millis();

  if (now - lastInferenceMs < INFERENCE_INTERVAL_MS) {
    return;
  }

  lastInferenceMs = now;

  if (AI.invoke(1, false, false) != 0) {
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
  AI.begin();
  printCsvHeader();
}

void loop()
{
  logDetections();
}
