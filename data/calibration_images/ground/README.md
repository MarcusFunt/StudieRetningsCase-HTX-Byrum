# Ground Marker Reference Images

Use the Panel `Calibration` -> `Ground Homography` tab to take USB-only photos of the ground markers or a flat ChArUco board into this folder.

For automatic ChArUco ground homography, place the board flat on the walking plane, enter the board origin and rotation in ground coordinates, and build calibration directly from the detected ChArUco corners.

These photos can also be reference images for filling `data/ground_markers.csv` with marker pixel positions and measured ground coordinates. They are separate from ChArUco intrinsics photos in `data/calibration_images/charuco/`.

Delete the images after `outputs/calibration.json` has been verified. These images are not part of normal field data collection and must not be captured over Wi-Fi.
