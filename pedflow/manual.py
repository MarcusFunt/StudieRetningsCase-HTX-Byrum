from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .analysis import FlowAnalysisResult, FlowAnalysisSettings, run_flow_analysis
from .calibration import homography_quality_report
from .geometry import apply_homography

SYNTHETIC_DETECTION_COLUMNS = (
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
IDENTITY_CAMERA_MATRIX = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
ZERO_DISTORTION = [0.0, 0.0, 0.0, 0.0, 0.0]


@dataclass(frozen=True)
class ManualPathPoint:
    image_x: float
    image_y: float
    elapsed_s: float = 0.0


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _xy_point(value: object, label: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        x_value = value.get("image_x", value.get("x"))
        y_value = value.get("image_y", value.get("y"))
    else:
        try:
            x_value = value[0]  # type: ignore[index]
            y_value = value[1]  # type: ignore[index]
        except (TypeError, IndexError) as exc:
            raise ValueError(f"{label} must contain x/y coordinates") from exc

    return (
        _finite_float(x_value, f"{label} x"),
        _finite_float(y_value, f"{label} y"),
    )


def _path_point(value: object, label: str) -> ManualPathPoint:
    x, y = _xy_point(value, label)
    if isinstance(value, Mapping):
        elapsed_value = value.get("elapsed_s", value.get("time_s", value.get("t", 0.0)))
    else:
        try:
            elapsed_value = value[2]  # type: ignore[index]
        except (TypeError, IndexError):
            elapsed_value = 0.0

    elapsed_s = _finite_float(elapsed_value, f"{label} elapsed_s")
    if elapsed_s < 0:
        raise ValueError(f"{label} elapsed_s must be non-negative")
    return ManualPathPoint(x, y, elapsed_s)


def _image_size_tuple(image_size: Sequence[object]) -> tuple[int, int]:
    try:
        width = int(image_size[0])
        height = int(image_size[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("image_size must contain width and height") from exc
    if width <= 0 or height <= 0:
        raise ValueError("image_size width and height must be positive")
    return width, height


def _polygon_area(points: np.ndarray) -> float:
    xs = points[:, 0]
    ys = points[:, 1]
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2.0)


def _validate_point_set(points: np.ndarray, label: str) -> None:
    if len(np.unique(points, axis=0)) < 4:
        raise ValueError(f"{label} points must contain four unique points")
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2 or _polygon_area(points) <= 1e-6:
        raise ValueError(f"{label} points must form a non-degenerate quadrilateral")


def _coerce_corners(corners_px: Sequence[object]) -> np.ndarray:
    if len(corners_px) != 4:
        raise ValueError("Manual calibration requires exactly four road corner points")
    points = np.asarray(
        [_xy_point(corner, f"corner {index}") for index, corner in enumerate(corners_px, start=1)],
        dtype=np.float64,
    )
    _validate_point_set(points, "Manual calibration image")
    return points


def _coerce_paths(paths: Sequence[Sequence[object]]) -> list[list[ManualPathPoint]]:
    coerced: list[list[ManualPathPoint]] = []
    for path_index, path in enumerate(paths, start=1):
        points = [
            _path_point(point, f"path {path_index} point {point_index}")
            for point_index, point in enumerate(path, start=1)
        ]
        if len(points) < 2:
            raise ValueError(f"path {path_index} must contain at least two points")
        coerced.append(points)
    return coerced


def build_manual_calibration(
    corners_px: Sequence[object],
    length_m: float,
    width_m: float,
    image_size: Sequence[object],
) -> dict:
    """Build a test-only identity-intrinsics calibration from four road corners."""

    image_width, image_height = _image_size_tuple(image_size)
    length = _finite_float(length_m, "length_m")
    width = _finite_float(width_m, "width_m")
    if length <= 0 or width <= 0:
        raise ValueError("length_m and width_m must be greater than zero")

    image_points = _coerce_corners(corners_px)
    ground_points = np.asarray(
        [
            [0.0, 0.0],
            [length, 0.0],
            [length, width],
            [0.0, width],
        ],
        dtype=np.float64,
    )

    homography, inlier_mask = cv2.findHomography(image_points, ground_points, 0)
    if homography is None:
        raise ValueError("OpenCV could not compute a manual homography")

    projected = apply_homography(image_points, homography)
    residuals = np.linalg.norm(projected - ground_points, axis=1)
    inliers = (
        inlier_mask.reshape(-1).astype(bool).tolist()
        if inlier_mask is not None
        else [True] * len(image_points)
    )

    calibration = {
        "schema_version": 1,
        "calibration_source": "manual_mode",
        "created_utc": datetime.now(UTC).isoformat(),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "K": IDENTITY_CAMERA_MATRIX,
        "dist": ZERO_DISTORTION,
        "H_image_to_ground": homography.astype(float).tolist(),
        "ground_marker_count": 4,
        "ground_marker_inliers": inliers,
        "ground_marker_residuals_m": residuals.astype(float).tolist(),
        "ground_marker_mean_residual_m": float(np.mean(residuals)),
        "ground_marker_max_residual_m": float(np.max(residuals)),
        "manual_road_length_m": float(length),
        "manual_road_width_m": float(width),
        "manual_corners_px": image_points.astype(float).tolist(),
        "manual_ground_corners_m": ground_points.astype(float).tolist(),
        "calibration_notes": (
            "Manual mode uses identity camera intrinsics and a user-clicked homography. "
            "Use this only for synthetic testing, not measurement-quality field analysis."
        ),
    }
    homography_report = homography_quality_report(calibration)
    calibration["homography_quality"] = homography_report
    calibration["calibration_quality"] = {
        "status": "warn",
        "intrinsics": {
            "status": "warn",
            "issues": ["Manual mode uses identity intrinsics and zero distortion."],
        },
        "homography": homography_report,
    }
    return calibration


def manual_paths_to_detections(
    paths: Sequence[Sequence[object]],
    bbox_width_px: float = 24.0,
    bbox_height_px: float = 64.0,
    path_gap_s: float = 2.0,
    min_sample_interval_s: float = 0.1,
) -> pd.DataFrame:
    bbox_width = _finite_float(bbox_width_px, "bbox_width_px")
    bbox_height = _finite_float(bbox_height_px, "bbox_height_px")
    gap = _finite_float(path_gap_s, "path_gap_s")
    min_interval = _finite_float(min_sample_interval_s, "min_sample_interval_s")
    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError("bbox_width_px and bbox_height_px must be greater than zero")
    if gap < 0:
        raise ValueError("path_gap_s must be non-negative")
    if min_interval <= 0:
        raise ValueError("min_sample_interval_s must be greater than zero")

    rows: list[dict[str, float | int]] = []
    base_s = 0.0
    frame_id = 0
    for path_index, path in enumerate(_coerce_paths(paths), start=1):
        first_elapsed_s = path[0].elapsed_s
        previous_relative_s: float | None = None

        for point in path:
            relative_s = point.elapsed_s - first_elapsed_s
            if relative_s < 0:
                raise ValueError(f"path {path_index} timestamps must be non-decreasing")
            if previous_relative_s is not None and relative_s <= previous_relative_s:
                relative_s = previous_relative_s + min_interval

            rows.append(
                {
                    "timestamp_ms": round((base_s + relative_s) * 1000.0),
                    "frame_id": frame_id,
                    "detection_id": 0,
                    "bbox_x": float(point.image_x - bbox_width / 2.0),
                    "bbox_y": float(point.image_y - bbox_height),
                    "bbox_w": float(bbox_width),
                    "bbox_h": float(bbox_height),
                    "confidence": 1.0,
                    "target": 0,
                }
            )
            frame_id += 1
            previous_relative_s = relative_s

        base_s += float(previous_relative_s or 0.0) + gap

    return pd.DataFrame(rows, columns=SYNTHETIC_DETECTION_COLUMNS)


def manual_path_summaries(paths: Sequence[Sequence[object]], calibration: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    homography = np.asarray(calibration["H_image_to_ground"], dtype=np.float64)
    for path_id, path in enumerate(_coerce_paths(paths), start=1):
        image_points = np.asarray(
            [[point.image_x, point.image_y] for point in path],
            dtype=np.float64,
        )
        ground_points = apply_homography(image_points, homography)
        segment_lengths = np.hypot(np.diff(ground_points[:, 0]), np.diff(ground_points[:, 1]))
        path_length_m = float(np.sum(segment_lengths))
        duration_s = max(float(path[-1].elapsed_s - path[0].elapsed_s), 0.0)
        rows.append(
            {
                "path_id": path_id,
                "points": len(path),
                "duration_s": duration_s,
                "path_length_m": path_length_m,
                "mean_speed_m_s": path_length_m / duration_s if duration_s > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_manual_analysis(
    image_metadata: Mapping[str, Any],
    corners_px: Sequence[object],
    paths: Sequence[Sequence[object]],
    length_m: float,
    width_m: float,
    settings: FlowAnalysisSettings | None = None,
) -> FlowAnalysisResult:
    image_size = (
        image_metadata.get("image_width", image_metadata.get("width")),
        image_metadata.get("image_height", image_metadata.get("height")),
    )
    calibration = build_manual_calibration(corners_px, length_m, width_m, image_size)
    detections = manual_paths_to_detections(paths)
    return run_flow_analysis(detections, calibration, settings)
