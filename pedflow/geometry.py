from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


Calibration = dict[str, Any]

DETECTION_COLUMNS = (
    "timestamp_ms",
    "frame_id",
    "detection_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "confidence",
    "target",
)
_DETECTION_NUMERIC_COLUMNS = DETECTION_COLUMNS
_DETECTION_ID_COLUMNS = ("frame_id", "detection_id", "target")


def load_calibration(path: str | Path) -> Calibration:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_calibration(calibration: Calibration, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(calibration, file, indent=2)
        file.write("\n")


def calibration_matrix(calibration: Calibration, key: str) -> np.ndarray:
    if key not in calibration:
        raise KeyError(f"Missing '{key}' in calibration data")
    return np.asarray(calibration[key], dtype=np.float64)


def validate_detection_input(detections: pd.DataFrame) -> pd.DataFrame:
    missing = set(DETECTION_COLUMNS).difference(detections.columns)
    if missing:
        raise ValueError(f"Missing required detection columns: {sorted(missing)}")

    output = detections.copy()
    for column in _DETECTION_NUMERIC_COLUMNS:
        converted = pd.to_numeric(output[column], errors="coerce")
        values = converted.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"Detection column '{column}' must contain only finite numeric values")
        output[column] = converted

    for column in _DETECTION_ID_COLUMNS:
        values = output[column].to_numpy(dtype=np.float64)
        if not np.equal(np.mod(values, 1.0), 0.0).all():
            raise ValueError(f"Detection column '{column}' must contain integer values")
        if (output[column] < 0).any():
            raise ValueError(f"Detection column '{column}' must be non-negative")
        output[column] = output[column].astype("int64")

    if (output["timestamp_ms"] < 0).any():
        raise ValueError("Detection column 'timestamp_ms' must be non-negative")
    if (output[["bbox_w", "bbox_h"]] < 0).any().any():
        raise ValueError("Detection bbox_w and bbox_h must be non-negative")
    if ((output["confidence"] < 0.0) | (output["confidence"] > 1.0)).any():
        raise ValueError("Detection confidence values must be between 0.0 and 1.0")

    return output


def bbox_foot_points(detections: pd.DataFrame) -> np.ndarray:
    required = {"bbox_x", "bbox_y", "bbox_w", "bbox_h"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Missing required bbox columns: {sorted(missing)}")

    foot_x = detections["bbox_x"].to_numpy(dtype=np.float64) + (
        detections["bbox_w"].to_numpy(dtype=np.float64) / 2.0
    )
    foot_y = detections["bbox_y"].to_numpy(dtype=np.float64) + detections["bbox_h"].to_numpy(
        dtype=np.float64
    )
    return np.column_stack([foot_x, foot_y])


def undistort_points(points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.size == 0:
        return point_array.reshape(0, 2)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")

    undistorted = cv2.undistortPoints(
        point_array.reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        P=np.asarray(camera_matrix, dtype=np.float64),
    )
    return undistorted.reshape(-1, 2)


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.size == 0:
        return point_array.reshape(0, 2)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")

    transformed = cv2.perspectiveTransform(
        point_array.reshape(-1, 1, 2),
        np.asarray(homography, dtype=np.float64),
    )
    return transformed.reshape(-1, 2)


def detections_to_ground(
    detections: pd.DataFrame,
    calibration: Calibration,
    confidence_threshold: float = 0.6,
) -> pd.DataFrame:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    validated = validate_detection_input(detections)
    filtered = validated.loc[validated["confidence"] >= confidence_threshold].copy()
    filtered = filtered.sort_values(["timestamp_ms", "frame_id", "detection_id"], kind="stable")

    foot_points = bbox_foot_points(filtered)
    camera_matrix = calibration_matrix(calibration, "K")
    dist_coeffs = calibration_matrix(calibration, "dist")
    homography = calibration_matrix(calibration, "H_image_to_ground")

    undistorted = undistort_points(foot_points, camera_matrix, dist_coeffs)
    ground_points = apply_homography(undistorted, homography)

    filtered["foot_x"] = foot_points[:, 0]
    filtered["foot_y"] = foot_points[:, 1]
    filtered["undistorted_foot_x"] = undistorted[:, 0]
    filtered["undistorted_foot_y"] = undistorted[:, 1]
    filtered["ground_x_m"] = ground_points[:, 0]
    filtered["ground_y_m"] = ground_points[:, 1]
    return filtered.reset_index(drop=True)
