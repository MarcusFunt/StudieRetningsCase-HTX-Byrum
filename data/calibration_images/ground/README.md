# Ground Marker Reference Images

Use the Panel `Calibration` -> `Ground Homography` tab to take USB-only photos of the ground markers into this folder.

These photos are reference images for filling `data/ground_markers.csv` with marker pixel positions and measured ground coordinates. They are separate from ChArUco intrinsics photos in `data/calibration_images/charuco/`.

Delete the images after `outputs/calibration.json` has been verified. These images are not part of normal field data collection and must not be captured over Wi-Fi.
