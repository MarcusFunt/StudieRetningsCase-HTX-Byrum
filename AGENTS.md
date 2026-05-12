# AGENTS.md

Working guide for AI agents (Claude Code, Codex, Copilot, etc.) contributing to this repo. Human contributors should also read this — it captures invariants, conventions, and gotchas that aren't obvious from a glance at the code.

For end-user / operator documentation, read `README.md` first. This file assumes you've read that.

---

## 1. What this project is

**Pedflow** is a school project: a **privacy-preserving pedestrian flow logger**. A Seeed Studio **Grove Vision AI Module V2** (Himax HX6538 + OV5647) runs an on-device person detector and emits anonymous bounding-box CSV rows. A **Seeed XIAO ESP32-C6** bridges the HX6538 to a laptop over Wi-Fi (UDP) and USB serial. A Python (Panel) dashboard on the laptop calibrates the camera, captures detection rows, links tracks, and produces flow analytics via [PedPy](https://pedpy.readthedocs.io/) + OpenCV.

The defining constraint is **privacy**:

- No surveillance footage is ever stored or transmitted in normal operation.
- No web server, no HTTP route, no Wi-Fi image streaming, no `camera.*` API.
- Only one path emits JPEG bytes: a one-shot `CALIB_CAPTURE` over **USB serial** (physical cable), triggered from the local GUI. Calibration photos are deleted after the calibration JSON is verified.
- These rules are **enforced by tests** (`tests/test_privacy_firmware.py`), not just docs. See §6.

If you propose a change that touches Wi-Fi/UDP, the firmware sketch, or image capture, this is the property you must not regress.

---

## 2. Architecture at a glance

```
┌────────────────────┐    UART 921600    ┌────────────────────┐    USB CDC      ┌────────────────────┐
│  HX6538 (Himax)    │ ─── JSON events ──▶│  XIAO ESP32-C6     │ ──── lines ────▶│  Laptop (Python)   │
│  sscma-micro +     │                    │  Arduino sketch    │                 │  pedflow package   │
│  CALIBSAMPLE patch │ ◀── AT commands ───│  (translator)      │ ◀── commands ───│  Panel dashboard   │
│  + OV5647 driver   │                    │                    │                 │                    │
│                    │                    │                    │   UDP 4210      │                    │
│                    │                    │   Wi-Fi AP ────────┼─── CSV rows ───▶│  capture_wifi.py   │
└────────────────────┘                    └────────────────────┘                 └────────────────────┘
```

Three layers, three languages:

| Layer | Code | Language |
|---|---|---|
| **HX6538 firmware** | `firmware/hx6538/patches/sscma-micro-calibsample-uart.patch` (applied to external `sscma-micro` clone) | C++ |
| **ESP32-C6 sketch** (the "bridge") | `GroveAIV2_Box_AP/GroveAIV2_Box_AP.ino` | Arduino C++ |
| **Host tools** (capture, calibration, analysis, GUI) | `pedflow/`, `scripts/`, `tests/` | Python 3.11+ |

The HX6538 source is **not** in this repo. It lives under `external/SSCMA-Micro` after running the setup scripts (see §4). Only the patch file is tracked.

---

## 3. Repo layout

```
.
├── AGENTS.md                   ← you are here
├── README.md                   ← operator/setup docs
├── pyproject.toml              ← Python project + ruff + pytest config
├── requirements.txt            ← deps for pip install -r
├── setup.ps1                   ← one-shot Windows setup (venv, deps, secrets, board, tests)
├── pytest config               in pyproject.toml: testpaths=["tests"], pythonpath=["."]
│
├── GroveAIV2_Box_AP/
│   └── GroveAIV2_Box_AP.ino    ← XIAO ESP32-C6 sketch (~1400 lines)
│       (also: pedflow_secrets.h, generated, gitignored)
│
├── firmware/hx6538/
│   ├── setup_sdks.ps1          ← clone Himax SDK + sscma-micro into external/
│   ├── apply_sscma_patch.ps1   ← apply our patch on top of sscma-micro
│   └── patches/
│       └── sscma-micro-calibsample-uart.patch
│
├── pedflow/                    ← the Python package (see §7 for module map)
│   ├── __init__.py             ← public API re-exports
│   ├── analysis.py             ← FlowAnalysisSettings / run_flow_analysis / write_analysis_outputs
│   ├── calibration.py          ← ChArUco board, camera intrinsics, ground homography
│   ├── capture.py              ← UDP/serial CSV capture worker (background thread)
│   ├── debug_gui.py            ← thin Panel entry point that mounts UsbDebugPanel
│   ├── debug_panel.py          ← USB Debug tab (live boxes/tracks/metrics)
│   ├── geometry.py             ← detection schema + bbox→ground pipeline (OpenCV)
│   ├── gui.py                  ← main Panel app (Analysis / Calibration / USB Debug / Operations)
│   ├── live_debug.py           ← serial reader + per-snapshot live analysis
│   ├── metrics.py              ← PedPy-backed speed/density/dwell/grid + summaries
│   ├── operations_panel.py     ← Operations tab (run scripts from GUI)
│   ├── serial_protocol.py      ← CSV header, status-line parsing, row validation
│   ├── tracking.py             ← linear-assignment track linker (scipy)
│   ├── ui_helpers.py           ← Panel widget helpers (paths, serial ports, options)
│   └── usb_calibration_capture.py  ← USB JPEG capture protocol (#calibration_image_*)
│
├── scripts/                    ← standalone CLI tools (all callable from the GUI's Operations tab too)
│   ├── capture_serial.py       ← USB serial → CSV
│   ├── capture_wifi.py         ← UDP → CSV
│   ├── generate_charuco_board.py
│   └── generate_secrets.py     ← Wi-Fi SSID/password generator (secrets/, pedflow_secrets.h)
│
├── tests/                      ← pytest suite (see §8)
├── data/                       ← local working data (detections, calibration images) — mostly gitignored
├── outputs/                    ← generated outputs (calibration JSON, analysis CSV/PNG/HTML) — gitignored
├── secrets/                    ← Wi-Fi credentials, gitignored, never commit
├── external/                   ← cloned SDKs for HX6538 firmware build, gitignored
└── .github/workflows/pytest.yml ← CI on windows-latest, Python 3.11, ruff + pytest
```

**Gitignored, do not commit:** `.venv/`, `secrets/`, `external/`, `data/detections/*.csv`, `data/calibration_images/**/*.jpg`, `GroveAIV2_Box_AP/pedflow_secrets.h`, `outputs/**`.

---

## 4. Setup (Windows-first)

The whole tooling stack assumes Windows + PowerShell because that's the deploy target. CI also runs on `windows-latest`.

```powershell
# One-shot setup: venv, deps, Wi-Fi secrets, folders, ChArUco board, run tests
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Useful flags: `-SkipTests`, `-SkipBoard`, `-RotateSecrets`, `-ForceGroundMarkers`, `-Python <path>`.

After setup, the venv is at `.venv/`. Always run Python tools via `.\.venv\Scripts\python.exe`.

```powershell
# Tests
.\.venv\Scripts\python.exe -m pytest

# Lint (must pass in CI)
.\.venv\Scripts\python.exe -m ruff check .

# Main GUI
.\.venv\Scripts\python.exe -m panel serve pedflow/gui.py --show --autoreload

# Live USB-only debug GUI (smaller, just the live boxes tab)
.\.venv\Scripts\python.exe -m panel serve pedflow/debug_gui.py --show
```

For HX6538 firmware work (only needed if you're modifying the patch):

```powershell
powershell -ExecutionPolicy Bypass -File .\firmware\hx6538\setup_sdks.ps1     # clones into external/
powershell -ExecutionPolicy Bypass -File .\firmware\hx6538\apply_sscma_patch.ps1
```

A teammate who clones fresh and skips these will build a firmware **without** `AT+CALIBSAMPLE`; the ESP32 will time out with `calibration_capture_timeout`. README §"HX6538 Calibration Firmware" documents the full build/flash flow.

---

## 5. Build, test, lint — what to run before you commit

| Action | Command |
|---|---|
| Lint everything | `python -m ruff check .` |
| Run the full test suite | `python -m pytest` |
| Run one test file | `python -m pytest tests/test_geometry.py -q` |
| Run one test | `python -m pytest tests/test_geometry.py::test_bbox_foot_points_use_bottom_center` |
| Regenerate the ChArUco PDF | `python scripts/generate_charuco_board.py --output-dir outputs/charuco_board` |
| Rotate Wi-Fi secrets | `python scripts/generate_secrets.py --rotate` |

CI (`.github/workflows/pytest.yml`) runs **ruff then pytest** on `push` and `pull_request`. Both must pass. There is no separate type-check job (no mypy/pyright config), but the package uses modern Python (`from __future__ import annotations`, PEP 604 unions, builtin generics) — keep it that way.

**Ruff config** (in `pyproject.toml`): line length 100, target `py311`, selects `B, E4, E7, E9, F, I, RUF, UP`. Notable: `I` enforces import sorting, `UP` upgrades to modern syntax, `RUF` is Ruff-specific lints. Three `scripts/*` files have `E402` ignored because they `sys.path.insert(...)` before importing local modules — keep that pattern only there.

---

## 6. Invariants the code is built around

If you break one of these, you're not refactoring, you're breaking the project.

### 6.1 Privacy rules (enforced by `tests/test_privacy_firmware.py`)

Searches the .ino source for forbidden tokens. **The sketch must not contain (case-insensitive):**

- `#include <WebServer.h>`, `WiFiServer`, `WebServer`, `server.on` → no HTTP server
- `stream` → no streaming
- `save_jpeg` → no on-device JPEG persistence
- `camera` (any case) → no camera API surface (`AI.invoke(image, true)` etc.)

It also **must contain** all the markers that prove the legitimate paths exist:

- The detection invocation: `AT+INVOKE=1,0,1\r\n` (image arg `0` — boxes only, no JPEG)
- The CSV header (exact bytes)
- The HX6538 reset path (`SSCMA_RESET_PIN = D3`, `resetSscmaModule`)
- The calibration capture path (`CALIB_CAPTURE`, `AT+CALIBSAMPLE=1\r\n`, opt id 5, `#calibration_image_begin/_end`, sensor restore)
- The Wi-Fi telemetry path (`udp.beginPacket(...)`, broadcast on `192.168.4.255`)

When you edit the .ino, **run `pytest tests/test_privacy_firmware.py` first** — these are the cheapest tests in the repo and catch most regressions instantly.

### 6.2 Calibration capture is USB-only

The XIAO sketch has exactly one code path that triggers JPEG bytes: a `CALIB_CAPTURE` line received on the USB `Serial` port. There is no UDP listener for it, no Wi-Fi command, no auto-trigger, no button binding. Don't add one.

### 6.3 Detection row schema is frozen

The CSV header is defined in two places that **must stay in sync**:

- `GroveAIV2_Box_AP.ino`: `constexpr const char CSV_HEADER[] = "..."`
- `pedflow/serial_protocol.py`: `CSV_HEADER = "..."`

Schema: `timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target`.

Bbox is **top-left + size**, in pixels. The sketch converts from the Grove Vision AI V2 AT protocol's center-form `x,y,w,h`. Confidence is `[0.0, 1.0]` (sketch divides the AT `score` from `0..100`).

`pedflow/geometry.py::DETECTION_COLUMNS` is the single source of truth on the Python side. `validate_detection_input` enforces types, finiteness, non-negativity, and the confidence range.

### 6.4 Status / error line format

Anything the firmware emits over USB serial that isn't a CSV row starts with `#`. These are parsed by `pedflow/serial_protocol.py::firmware_version_from_status_line` and (for calibration) by `pedflow/usb_calibration_capture.py::_capture_settings_from_status_line`.

Convention: `#status,<code>[,<value>[,<detail>]]` or `#error,<code>[,...]`. When you add a new status line, add a parser case in the appropriate module **and** a test, otherwise it'll be silently dropped from sidecar metadata.

### 6.5 Calibration JSON keys

Read/written by `pedflow/geometry.py::load_calibration` / `save_calibration` / `calibration_matrix`. Downstream code expects:

- `K` — 3×3 camera matrix
- `dist` — distortion coefficients
- `H_image_to_ground` — 3×3 homography from undistorted image pixels to ground meters

Geometry pipeline order (in `detections_to_ground`):

```
distorted bbox foot point  →  cv2.undistortPoints  →  cv2.perspectiveTransform (H)  →  ground x/y meters
```

Don't reorder this. Don't apply the homography before undistortion.

### 6.6 Secrets file flow

`scripts/generate_secrets.py` is the **only** source of truth for Wi-Fi credentials. It writes four files in one pass from a single password:

- `secrets/pedflow_wifi.json` (canonical)
- `secrets/pedflow_wifi.txt` (human-readable)
- `secrets/PedFlowSensor-wifi-profile.xml` (Windows Wi-Fi profile)
- `GroveAIV2_Box_AP/pedflow_secrets.h` (Arduino header)

All four are gitignored. Never check any of them in. The firmware build pulls credentials from `pedflow_secrets.h`; the laptop reads `pedflow_wifi.txt` or installs the XML profile.

---

## 7. Module-by-module Python reference

Keep imports inside `pedflow/` as **relative** (`from .geometry import ...`); scripts under `scripts/` use absolute imports after the `sys.path.insert` shim. `pedflow/gui.py` and `operations_panel.py` use a `try: from .X / except ImportError: from pedflow.X` dual-import pattern so they work both when Panel serves them from a source checkout and when the package is installed.

| Module | What it does | Public surface |
|---|---|---|
| `geometry.py` | Detection schema, bbox→ground transform, calibration I/O | `DETECTION_COLUMNS`, `validate_detection_input`, `bbox_foot_points`, `undistort_points`, `apply_homography`, `detections_to_ground`, `load_calibration`, `save_calibration` |
| `serial_protocol.py` | CSV header + row validation, status-line parsing | `CSV_HEADER`, `CSV_COLUMNS`, `UNKNOWN_FIRMWARE_VERSION`, `validate_csv_row`, `parse_serial_csv_line`, `firmware_version_from_status_line`, `metadata_path_for` |
| `capture.py` | Background-thread UDP/serial CSV writer with metadata sidecar | `_CsvCaptureWorker`, `CaptureSnapshot`, `parse_udp_payload_lines` |
| `tracking.py` | Hungarian-assignment track linker with α-smoothed predictions | `link_detections`, `filter_short_tracks` |
| `metrics.py` | PedPy speeds/density/dwell + grid stats + summary table | `estimate_speeds`, `add_dwell_flags`, `summarize_flow`, `track_summaries`, `grid_statistics` |
| `analysis.py` | One-call flow analysis (`run_flow_analysis`) and output writer with QA PNG/HTML | `FlowAnalysisSettings` (frozen dataclass), `FlowAnalysisResult`, `run_flow_analysis`, `write_analysis_outputs` |
| `calibration.py` | ChArUco board PDF/PNG, intrinsics, ground homography (from CSV markers or ChArUco-on-floor), pass/warn/fail quality | `DEFAULT_CHARUCO_*`, `generate_charuco_board`, `calibrate_camera_from_charuco`, `compute_ground_homography`, `compute_ground_homography_from_charuco`, `merge_and_save_calibration`, `read_marker_csv` |
| `usb_calibration_capture.py` | USB serial `CALIB_CAPTURE` request, chunked base64 reassembly, JPEG validation, metadata sidecar | `capture_usb_calibration_photos`, `request_calibration_image`, `decode_calibration_image_payload`, `UsbCalibrationCapture` |
| `live_debug.py` | Background-thread USB serial reader + per-snapshot analysis for the live tab | `SerialDebugReader`, `analyze_live_debug_rows`, `DEFAULT_DEBUG_SETTINGS` |
| `debug_panel.py` | The `UsbDebugPanel` widget tree | `UsbDebugPanel` |
| `operations_panel.py` | The `OperationsPanel` widget tree (runs scripts as subprocesses) | `OperationsPanel` |
| `gui.py` | Top-level Panel app: 4 tabs (Analysis / Calibration / USB Debug / Operations) | top-level `template.servable()` |
| `ui_helpers.py` | Path / serial-port / file-listing widget helpers | `directory_options`, `file_options`, `serial_port_options`, `resolve_path`, `display_path`, `keep_or_first` |

**Conventions inside `pedflow/`:**

- `from __future__ import annotations` at the top of every module.
- Dataclasses for value types, mostly **frozen** (`@dataclass(frozen=True)`).
- All public functions take/return `pd.DataFrame` or `np.ndarray` with explicit column / shape contracts; validate at the boundary (`validate_detection_input`, `_parse_int`).
- Long-running work (USB / UDP capture, calibration capture, live debug) uses a small background-thread worker class with `threading.Event` for stop signaling and a deque-bounded log buffer.
- Metadata sidecars (`*.metadata.json`) accompany every output file produced by a capture or analysis run. The schema is intentionally permissive: capture-side firmware fields land in `settings`, analysis fields land in the manifest. When you add new firmware status codes, plumb them into the relevant `_*_from_status_line` parser so they show up in sidecars.

---

## 8. Testing strategy

Tests live in `tests/` (configured in `pyproject.toml`). They're fast — the whole suite runs in well under a minute on CI hardware. There's no fixture framework; tests use plain pytest functions and construct in-memory DataFrames / fake serial objects.

| Test file | Covers |
|---|---|
| `test_privacy_firmware.py` | **Static analysis of the .ino** — the privacy invariants in §6.1 |
| `test_geometry.py` | bbox foot points, undistort, homography, detection validation |
| `test_calibration.py` | ChArUco board generation, intrinsics math, ground homography from CSV/board |
| `test_charuco.py` | Default board params, output file shape |
| `test_tracking_metrics.py` | Track linker assignment, speed/density/dwell |
| `test_analysis.py` | End-to-end `run_flow_analysis` on synthetic data |
| `test_synthetic_pipeline.py` | End-to-end with a fully synthetic detection stream |
| `test_capture_serial.py`, `test_capture_wifi.py` | CSV writer worker with fake serial/UDP sources |
| `test_live_debug.py` | Live debug parser + analyzer |
| `test_usb_calibration_capture.py` | USB calibration capture: chunk reassembly, error lines, status mapping, metadata sidecar |
| `test_secrets_generation.py` | Wi-Fi credential generator + Windows XML profile escaping |
| `test_panel_gui.py` | Smoke test that the Panel widgets construct without raising |

**Patterns to follow when adding tests:**

- **Don't** spin up real Panel / browser. Test the underlying functions instead, and rely on `test_panel_gui.py` for "it instantiates" coverage.
- **Don't** open real serial ports. Construct a small `SerialLike` stub class with a `readline()` that returns pre-baked bytes (see how `test_usb_calibration_capture.py` does it).
- **Don't** require GPU / Wi-Fi / hardware. Tests must pass on `windows-latest` GitHub runners with no peripherals.
- **Do** add a `test_privacy_firmware.py` assertion if you add a new sketch invariant.

---

## 9. Wire protocols cheat sheet

### 9.1 HX6538 → ESP32 (UART, 921600 8N1)

Two protocols share the wire:

1. **Detections** (`AT+INVOKE=1,0,1`): chunked JSON events `\r{"type":1,"name":"INVOKE","code":0,"data":{"boxes":[[x,y,w,h,score,target], ...], ...}}\n`. The sketch's `findRawSscmaMessage` framer scans for `\r{...}\n` boundaries.
2. **Calibration capture** (`AT+CALIBSAMPLE=1`, added by our patch): event sequence `begin → N × chunk → end`. Each `chunk` event carries 3072 raw bytes / 4100 base64 chars of JPEG in `data.image_chunk`. Sensor preset opt_id 5 ("640x480 Calibration HQ", `JPEG_ENC_QTABLE_4X`) is the only one that emits this format; opt 0/1/2 use `QTABLE_10X` for the regular streaming detection mode.

ESP32-side parser uses ArduinoJson v7 `JsonDocument` (heap-backed, elastic — no fixed capacity tuning needed). It validates `chunk_index` order and accumulated `base64_length` before forwarding.

### 9.2 ESP32 → laptop (USB CDC `Serial`)

Mostly line-oriented. Three categories of line:

| Prefix | Meaning |
|---|---|
| `<csv row>` | A detection: 9 comma-separated fields per §6.3 |
| `#status,<code>[,...]` | Informational firmware status |
| `#error,<code>[,...]` | Recoverable error |
| `#calibration_image_begin,<base64_length>` | Start of a calibration capture |
| `<base64 line>` | One chunk of base64 JPEG (multiple lines per image) |
| `#calibration_image_end` | End-of-image marker |

USB commands the laptop can send (one per line):

- `CALIB_CAPTURE` — capture one calibration JPEG
- `MODULE_INFO` — print AT+ID/NAME/INFO + UART/Wi-Fi diagnostics
- `WIFI_STATUS` — print AP + UDP counters
- `UDP_TEST` — emit a burst of status datagrams
- `AT:<raw>` — pass-through to the HX6538

### 9.3 ESP32 → laptop (UDP `192.168.4.0/24` port 4210)

Same line format as USB, but UDP datagrams. The sketch broadcasts to `192.168.4.255` and, once a station is connected, also unicasts to expected client addresses. Heartbeat datagrams (`#status,udp_heartbeat,...`) go out at 1 Hz.

---

## 10. Common gotchas

- **One process owns a serial port.** If `capture_serial.py` or the USB Debug tab is running, the Calibration tab's USB capture will fail. Stop the other first.
- **Calibration capture and detection streaming use different sensor opts.** Capture switches to opt 5 (HQ JPEG), captures one frame, then **restores** the previous opt. Look for `sensor_restore_status: "restored"` in the calibration metadata sidecar. If it's `"failed"`, detection capture afterward will be wrong-resolution until you reset the board.
- **The HX6538 SDK is huge and lives in `external/`.** Don't try to commit it. Only the patch and the helper PowerShell scripts are tracked.
- **`data/`, `outputs/`, `secrets/`, and `external/` are mostly gitignored.** Don't fight the .gitignore; if you genuinely need to track a file under those paths, add an explicit `!` rule in `.gitignore`.
- **Panel + Matplotlib backend.** `pedflow/analysis.py` calls `matplotlib.use("Agg")` at import time, before any pyplot import, because the analysis runs in non-GUI contexts. If you add a module that imports pyplot, do the same, or import after `analysis.py` has loaded.
- **Windows paths everywhere.** All scripts assume PowerShell + backslashes; CI runs on `windows-latest`. If you add a script that has to handle paths, use `pathlib.Path` and don't hand-roll string concatenation.
- **PedPy column conventions.** `pedflow/metrics.py` translates between pedflow's `(timestamp_ms, track_id, ground_x_m, ground_y_m)` and PedPy's `(FRAME_COL, ID_COL, X_COL, Y_COL)`. Don't bypass that translation — PedPy is strict about column names and frame indices.
- **Detection bbox is top-left, but the AT protocol is centered.** Conversion happens in the sketch. Python code only ever sees top-left.

---

## 11. Where to add things

| Task | Where |
|---|---|
| New analysis metric (e.g., a new heatmap) | `pedflow/metrics.py` for the math, `pedflow/analysis.py` for the `FlowAnalysisResult` plumbing + `write_analysis_outputs` for the QA file, and a test in `test_tracking_metrics.py` or `test_analysis.py` |
| New firmware status line | Sketch: emit via `printUsbOnlyStatusValue` (or UDP equivalent). Python: add a case in `pedflow/serial_protocol.py` *or* `pedflow/usb_calibration_capture.py::_capture_settings_from_status_line`. Test in the matching `tests/` file. |
| New USB command | Sketch: add a `USB_<NAME>_COMMAND` constant and a handler in the USB-line dispatcher. Add an assertion in `test_privacy_firmware.py` to lock in the command string. |
| New Panel tab | Build a panel class in its own module (mirror `debug_panel.py` / `operations_panel.py`), mount it from `pedflow/gui.py` with the dual-import shim, smoke-test in `tests/test_panel_gui.py`. |
| New CLI script | Drop in `scripts/`, add the `sys.path.insert` shim, add `E402` to `[tool.ruff.lint.per-file-ignores]` if you need imports below the shim, and wire it into the Operations tab if it should be GUI-runnable. |
| New calibration default | Add a `DEFAULT_CHARUCO_*` (or equivalent) constant in `pedflow/calibration.py`; re-export from `pedflow/gui.py` / `operations_panel.py` / `scripts/generate_charuco_board.py` so the GUI default and CLI default stay in sync. There's a `test_charuco.py` check for the defaults. |
| New firmware behaviour (sketch) | Edit `GroveAIV2_Box_AP.ino`. **Always** run `pytest tests/test_privacy_firmware.py -q` before/after. |
| Modifying the HX6538 firmware | Edit `firmware/hx6538/patches/sscma-micro-calibsample-uart.patch` (or regenerate it from a working `external/SSCMA-Micro` tree). Then run `apply_sscma_patch.ps1` clean to verify the patch still applies. Document any new AT command in README §"HX6538 Calibration Firmware". |

---

## 12. Style nudges for AI agents

- Match the existing typing style: `from __future__ import annotations`, PEP 604 unions (`int | None`), builtin generics (`list[str]`). Don't import from `typing` unless you actually need `Protocol`, `Callable`, etc.
- Don't reach for new dependencies. The pin list in `requirements.txt` is intentional. If you think something needs adding, justify it in the PR.
- Keep functions pure where reasonable. The heavy I/O code paths (`capture.py`, `live_debug.py`, `usb_calibration_capture.py`) all separate parsing from I/O so the parsers are unit-testable without serial ports.
- When changing the .ino, prefer extending the existing translator-and-line-protocol architecture (HX6538 emits JSON event → ESP32 validates → ESP32 emits `#`-prefixed line) over inventing a new wire format.
- When adding error states, emit a `#error,<code>` line on the sketch side and add a parser case + test on the Python side. Silent failures are the hardest bugs in this stack.
- Don't `git add -A`. The `.gitignore` catches most things, but adding files under `secrets/`, `external/`, `data/`, or `outputs/` by hand is almost never what you want.
- Don't add a CLAUDE.md. This file is the agent guide.

---

## 13. Hardware reference (quick)

- **Compute board:** Seeed XIAO ESP32-C6 (USB Serial CDC at 115200, Wi-Fi station only as AP, UDP 4210)
- **AI module:** Seeed Grove Vision AI V2 (Himax HX6538, OV5647 camera)
- **Wiring (UART bridge):** HX `PB6/UART1_RX` ↔ XIAO `D6/TX`; HX `PB7/UART1_TX` ↔ XIAO `D7/RX`; common GND; 3.3 V logic only.
- **Reset pin:** XIAO `D3` → HX6538 reset (used by `resetSscmaModule`).
- **Default Wi-Fi AP:** SSID `PedFlowSensor` (override via `generate_secrets.py --ssid`), channel 6, max 2 clients, hidden=false. Sensor IP `192.168.4.1`, UDP broadcast `192.168.4.255`, port `4210`.

External references (Grove Vision AI V2 docs, OV5647 / HX6538 datasheets, Himax + sscma-micro repos, XIAO ESP32-C6 getting-started) are in the project intake notes; pull them up via `README.md` and the Seeed Wiki if you need to touch firmware.

---

## 14. When in doubt

- Read `README.md` for the operator's view of the workflow.
- Read `tests/test_privacy_firmware.py` for the non-negotiable firmware invariants.
- Read `pedflow/geometry.py` for the detection schema and the bbox→ground pipeline.
- Read `pedflow/serial_protocol.py` for the wire format.
- Read `pedflow/analysis.py::FlowAnalysisSettings` for the tunable analysis parameters.

If a change you're considering wouldn't fit in one of those files or their direct neighbours, you're probably introducing a new concept — flag it in the PR description rather than scattering it.
