# Data Folder

Use this folder for local working data. Do not commit real field data unless it has been reviewed for privacy.

- `calibration_images/charuco/`: manually captured ChArUco calibration images. Delete after `outputs/calibration_intrinsics.json` is verified.
- `calibration_images/checkerboard/`: optional fallback checkerboard images if you choose the older calibration path.
- `ground_markers.csv`: measured marker correspondences for homography calibration.
- `detections/session.csv`: anonymous USB-serial bbox CSV from normal data collection.

Normal data collection must contain only detection rows, never images or video.
