import cv2
import numpy as np
import pytest

from pedflow.calibration import calibrate_camera_from_checkerboard


def test_checkerboard_calibration_rejects_mixed_image_resolutions(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    cv2.imwrite(str(first), np.zeros((20, 20, 3), dtype=np.uint8))
    cv2.imwrite(str(second), np.zeros((30, 30, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="same resolution"):
        calibrate_camera_from_checkerboard(
            [first, second],
            pattern_size=(3, 3),
            square_size_m=0.1,
        )
