# Privacy-Preserving Pedestrian Flow Logger

School project for measuring pedestrian flow and movement patterns with a Grove Vision AI Module V2 and Seeed Studio XIAO ESP32-C6 without storing or transmitting normal surveillance footage.

## Privacy Rules

During normal data collection the firmware:

- does not start Wi-Fi
- does not serve a web page or live box endpoint
- does not stream camera frames
- does not call image/JPEG capture APIs
- logs only anonymous detection rows over USB serial

Calibration images are separate manual inputs for OpenCV geometry calibration. They should be captured manually, avoid pedestrians, and be deleted after the calibration JSON is verified. The physical button must not be used to trigger automatic calibration capture.

## Firmware

The sketch in `GroveAIV2_Box_AP/GroveAIV2_Box_AP.ino` prints CSV rows over USB serial at 115200 baud:

```text
timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target
```

The Seeed SSCMA library reports boxes as center `x,y,w,h`. The sketch converts them to top-left `bbox_x,bbox_y,bbox_w,bbox_h`, and converts `score` from `0-100` to confidence `0.0-1.0`.

Capture a session from a PC:

```powershell
python scripts/capture_serial.py --port COM5 --output data/detections/session.csv
```

Change `COM5` to the XIAO serial port.

## Setup

Install and prepare the project:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The setup script creates `.venv`, installs Python dependencies, prepares local data/output folders, copies `data/ground_markers_template.csv` to `data/ground_markers.csv` if needed, generates the printable ChArUco board, and runs the tests.

Manual dependency install:

```powershell
python -m pip install -r requirements.txt
```

Start the local Panel dashboard:

```powershell
.\.venv\Scripts\python.exe -m panel serve pedflow/gui.py --show --autoreload
```

## Panel Dashboard

The dashboard is the supported workflow. It has two top-level tabs:

- `Analysis`: choose or upload the anonymous detections CSV and calibration JSON, tune tracking and PedPy metric settings, view HoloViews/hvPlot paths and heatmaps, inspect Tabulator tables, and optionally write processed CSV outputs to `outputs/analysis/`.
- `Calibration`: generate the printable ChArUco board, calibrate camera intrinsics from manually captured ChArUco images, and combine intrinsics with `data/ground_markers.csv` into `outputs/calibration.json`.

Generate the printable ChArUco board from the command line if needed:

```powershell
python scripts/generate_charuco_board.py --output-dir outputs/charuco_board
```

Print `outputs/charuco_board/charuco_board.pdf` at 100% scale. Do not use fit-to-page, because the metadata stores the exact board dimensions used by OpenCV calibration.

Geometry order is always:

```text
distorted bbox foot point -> undistort point -> homography -> ground x/y meters
```

## Metrics

The analysis exports interpretable sub-metrics instead of one fake quality score:

- pedestrian count and people per minute/hour
- PedPy-backed position heatmap and path/desire-line plot
- PedPy-backed median speed heatmap
- stop/dwell map
- bottleneck index
- detour ratio and direction-change summaries

Direction changes are not treated as proof that a street is good or bad. They can also indicate obstacles, confusion, crowding, or tracking noise.

## Limitations

Bounding boxes are noisy, bbox bottom-center is only an estimated foot point, homography assumes flat ground, lens distortion correction depends on calibration quality, occlusion can break tracks, and manual marker clicking introduces measurement error.
