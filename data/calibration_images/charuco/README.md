# ChArUco Calibration Images

Use the Panel `Calibration` -> `Camera Intrinsics` tab to take USB-only ChArUco calibration photos into this folder, or place manually captured ChArUco board images here.

Generate the printable board and metadata with:

```powershell
python scripts/generate_charuco_board.py --output-dir outputs/charuco_board
```

Delete calibration images after `outputs/calibration_intrinsics.json` has been verified. These images are not part of normal field data collection and must not be captured over Wi-Fi.
