# ChArUco Calibration Images

Place manually captured ChArUco board images here before using the Panel calibration tab.

Generate the printable board and metadata with:

```powershell
python scripts/generate_charuco_board.py --output-dir outputs/charuco_board
```

Delete calibration images after `outputs/calibration_intrinsics.json` has been verified. These images are not part of normal field data collection.
