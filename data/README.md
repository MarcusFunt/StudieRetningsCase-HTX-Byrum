# Data Folder

Use this folder for local working data. Do not commit real field data unless it has been reviewed for privacy.

- `calibration_images/charuco/`: ChArUco board photos for camera intrinsics only. Delete after `outputs/calibration_intrinsics.json` is verified.
- `calibration_images/ground/`: flat ChArUco ground photos or marker reference photos for homography. Keep separate from ChArUco intrinsics photos.
- `ground_markers.csv`: measured marker correspondences for homography calibration.
- `detections/session.csv`: anonymous USB-serial bbox CSV from normal data collection.

Detection captures write a matching `*.metadata.json` sidecar with firmware version, settings, start/end time, calibration path when known, row counts, and output files.

Calibration photo captures also write `*.metadata.json` sidecars beside each JPEG. Normal data collection must contain only detection rows, never images or video. Calibration photos are a separate USB-only GUI action.
