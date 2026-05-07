from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .geometry import apply_homography, save_calibration, undistort_points


def checkerboard_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    columns, rows = pattern_size
    points = np.zeros((rows * columns, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points *= square_size_m
    return points


def calibrate_camera_from_checkerboard(
    image_paths: Iterable[str | Path],
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> dict:
    object_template = checkerboard_object_points(pattern_size, square_size_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    image_size: tuple[int, int] | None = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path in image_paths:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        used_images.append(str(path))

    if image_size is None:
        raise ValueError("No calibration images could be read")
    if len(object_points) < 3:
        raise ValueError("At least 3 usable checkerboard images are required")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    per_image_errors = []
    for obj_points, img_points, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(img_points, projected, cv2.NORM_L2) / len(projected)
        per_image_errors.append(float(error))

    return {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "K": camera_matrix.tolist(),
        "dist": dist_coeffs.reshape(-1).tolist(),
        "checkerboard_pattern_size": list(pattern_size),
        "checkerboard_square_size_m": float(square_size_m),
        "rms_reprojection_error_px": float(rms),
        "per_image_reprojection_error_px": per_image_errors,
        "used_images": used_images,
        "calibration_notes": (
            "Images were manually captured for geometry calibration only. "
            "Delete calibration images after this file has been verified."
        ),
    }


def read_marker_csv(path: str | Path) -> pd.DataFrame:
    markers = pd.read_csv(path)
    required = {"image_x", "image_y", "ground_x_m", "ground_y_m"}
    missing = required.difference(markers.columns)
    if missing:
        raise ValueError(f"Missing marker columns: {sorted(missing)}")
    if len(markers) < 4:
        raise ValueError("At least 4 ground markers are required")
    return markers


def compute_ground_homography(
    marker_points: pd.DataFrame,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    ransac_threshold_m: float = 0.10,
) -> dict:
    image_points = marker_points[["image_x", "image_y"]].to_numpy(dtype=np.float64)
    ground_points = marker_points[["ground_x_m", "ground_y_m"]].to_numpy(dtype=np.float64)
    undistorted_image_points = undistort_points(image_points, camera_matrix, dist_coeffs)

    method = cv2.RANSAC if len(marker_points) > 4 else 0
    homography, inlier_mask = cv2.findHomography(
        undistorted_image_points,
        ground_points,
        method,
        ransac_threshold_m,
    )
    if homography is None:
        raise ValueError("OpenCV could not compute a ground homography")

    projected = apply_homography(undistorted_image_points, homography)
    residuals = np.linalg.norm(projected - ground_points, axis=1)

    mask = (
        inlier_mask.reshape(-1).astype(bool).tolist()
        if inlier_mask is not None
        else [True] * len(marker_points)
    )

    return {
        "H_image_to_ground": homography.tolist(),
        "ground_marker_count": int(len(marker_points)),
        "ground_marker_inliers": mask,
        "ground_marker_residuals_m": residuals.astype(float).tolist(),
        "ground_marker_mean_residual_m": float(np.mean(residuals)),
        "ground_marker_max_residual_m": float(np.max(residuals)),
    }


def merge_and_save_calibration(
    intrinsic_calibration: dict,
    homography_calibration: dict,
    output_path: str | Path,
) -> dict:
    calibration = {**intrinsic_calibration, **homography_calibration}
    save_calibration(calibration, output_path)
    return calibration
