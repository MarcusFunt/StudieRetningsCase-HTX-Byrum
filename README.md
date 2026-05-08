# Privacy-Preserving Pedestrian Flow Logger

School project for measuring pedestrian flow and movement patterns with a Grove Vision AI Module V2 and Seeed Studio XIAO ESP32-C6 without storing or transmitting normal surveillance footage.

## Privacy Rules

During normal data collection the firmware:

- starts a local WPA2 Wi-Fi access point only for anonymous CSV telemetry
- does not serve a web page or live box endpoint
- does not stream camera frames or detection previews
- does not call image/JPEG capture APIs during normal data collection
- sends only anonymous detection rows over Wi-Fi UDP and USB serial
- enables local debug/testing output only when a USB serial connection is open
- allows one-shot calibration photos only from a local USB serial GUI command

Calibration images are separate inputs for OpenCV geometry calibration. The Panel dashboard can request actual JPEG calibration photos over USB serial only; images are never served over Wi-Fi or included in UDP telemetry. Calibration photos should avoid pedestrians and be deleted after the calibration JSON is verified. The physical button must not be used to trigger automatic calibration capture.

Sensor Wi-Fi credentials are generated locally and ignored by git. Run setup, or run `python scripts/generate_secrets.py`, before flashing firmware so the device header and laptop credential files are created from the same local password.

## Firmware

The sketch in `GroveAIV2_Box_AP/GroveAIV2_Box_AP.ino` starts a direct sensor Wi-Fi network and sends CSV rows to the connected computer over UDP:

```text
SSID: see secrets/pedflow_wifi.txt
Password: see secrets/pedflow_wifi.txt
Sensor IP: 192.168.4.1
UDP port: 4210
```

The generated secret files share one local source of truth:

- `secrets/pedflow_wifi.json`: local source credentials
- `GroveAIV2_Box_AP/pedflow_secrets.h`: Arduino header included by the sketch
- `secrets/pedflow_wifi.txt`: laptop-readable SSID and password
- `secrets/PedFlowSensor-wifi-profile.xml`: Windows Wi-Fi profile for the laptop when using the default SSID

All of these files are untracked. Rotate the local password with:

```powershell
python scripts/generate_secrets.py --rotate
```

After rotating, flash the firmware again so the device uses the new generated header. On Windows, you can install the generated laptop Wi-Fi profile with the path printed by the generator. For the default SSID:

```powershell
netsh wlan add profile filename="secrets\PedFlowSensor-wifi-profile.xml" user=current
```

It also prints the same CSV rows over USB serial at 115200 baud:

```text
timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target
```

The Grove Vision AI V2 AT protocol reports boxes as center `x,y,w,h`. The sketch converts them to top-left `bbox_x,bbox_y,bbox_w,bbox_h`, and converts `score` from `0-100` to confidence `0.0-1.0`.

For UDP debugging, the firmware sends a one-second `#status,udp_heartbeat,...` datagram. With no connected station it uses `192.168.4.255`; once a laptop is connected to the sensor AP it also attempts unicast delivery to the expected client addresses on `192.168.4.0/24`. Over USB serial, `WIFI_STATUS` prints AP and UDP counters, and `UDP_TEST` sends a short burst of status datagrams.

Capture a session from a PC over Wi-Fi:

1. Flash `GroveAIV2_Box_AP/GroveAIV2_Box_AP.ino` to the XIAO ESP32-C6.
2. Connect the computer Wi-Fi using the generated SSID and password in `secrets/pedflow_wifi.txt`.
3. Start the UDP capture script:

```powershell
python scripts/capture_wifi.py --output data/detections/session.csv
```

Windows may ask whether Python can receive network traffic. Allow it on the private network for the sensor feed.

USB serial capture is still available for debugging:

```powershell
python scripts/capture_serial.py --port COM5 --output data/detections/session.csv
```

Change `COM5` to the XIAO serial port.

Calibration photos are taken from the GUI only over USB serial:

1. Flash `GroveAIV2_Box_AP/GroveAIV2_Box_AP.ino` to the XIAO ESP32-C6.
2. Start the Panel dashboard and open `Calibration` -> `Camera Intrinsics`.
3. Select the ChArUco image folder and USB serial port.
4. Click `Take USB calibration photo`.

Stop `USB Debug` or any serial capture script before taking calibration photos, because only one process can own the serial port at a time.

## USB Debug / Testing Mode

Debug mode is USB-only. It turns on automatically when a computer opens the XIAO USB serial port and turns off when that serial connection closes. There is no Wi-Fi command, UDP receiver, web route, or remote toggle that can enable it.

The `USB Debug` tab in the Panel dashboard reads the USB serial rows live and shows:

- latest bounding boxes in image coordinates
- bbox bottom-center ground contact points
- calibrated ground contact points and live tracks when `outputs/calibration.json` exists
- PedPy-backed live summary, speed, dwell, and grid-density outputs as soon as enough calibrated points exist

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m panel serve pedflow/gui.py --show --autoreload
```

Open the `USB Debug` tab, enter the XIAO serial port such as `COM5`, and click `Start USB debug`.

You can also run only the debug dashboard:

```powershell
.\.venv\Scripts\python.exe -m panel serve pedflow/debug_gui.py --show
```

## Setup

Install and prepare the project:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The setup script creates `.venv`, installs Python dependencies, generates local untracked Wi-Fi secrets, prepares local data/output folders, copies `data/ground_markers_template.csv` to `data/ground_markers.csv` if needed, generates the printable ChArUco board, and runs the tests.

Manual dependency install:

```powershell
python -m pip install -r requirements.txt
```

Start the local Panel dashboard:

```powershell
.\.venv\Scripts\python.exe -m panel serve pedflow/gui.py --show --autoreload
```

## Panel Dashboard

The dashboard is the supported workflow. It has four top-level tabs:

- `Analysis`: pick discovered detection sessions and calibration files from dropdowns, upload files only when needed, tune advanced settings, view paths and heatmaps, inspect tables, and optionally write processed CSV outputs.
- `Calibration`: generate the printable ChArUco board, take USB-only calibration photos, calibrate camera intrinsics from discovered image folders, and combine intrinsics with `data/ground_markers.csv` into `outputs/calibration.json`.
- `USB Debug`: pick a detected USB serial port, start local testing, and view live boxes, contact points, tracks, and PedPy outputs.
- `Operations`: run the repo scripts from the GUI, including Wi-Fi/USB CSV capture, USB-only calibration image capture, Wi-Fi secret generation, ChArUco board generation, and logged PedPy/OpenCV analysis runs.

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
