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


@dataclass(frozen=True)
class ManualCornerOrder:
    image_points: np.ndarray
    order_name: str
    source_indices: tuple[int, int, int, int]


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


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return (ab_c * ab_d < 0.0) and (cd_a * cd_b < 0.0)


def _simple_quadrilateral_score(points: np.ndarray) -> float | None:
    area = _polygon_area(points)
    if area <= 1e-6:
        return None
    if _segments_intersect(points[0], points[1], points[2], points[3]):
        return None
    if _segments_intersect(points[1], points[2], points[3], points[0]):
        return None
    side_lengths = np.hypot(
        np.diff(np.r_[points[:, 0], points[0, 0]]),
        np.diff(np.r_[points[:, 1], points[0, 1]]),
    )
    if np.min(side_lengths) <= 1e-6:
        return None
    return area


def _validate_point_set(points: np.ndarray, label: str) -> None:
    if len(np.unique(points, axis=0)) < 4:
        raise ValueError(f"{label} points must contain four unique points")
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError(f"{label} points must form a non-degenerate quadrilateral")


def _corner_order_score(
    points: np.ndarray,
    length_m: float | None = None,
    width_m: float | None = None,
) -> float | None:
    area = _simple_quadrilateral_score(points)
    if area is None:
        return None
    if length_m is None or width_m is None or length_m <= 0 or width_m <= 0:
        return area

    length_side_px = (
        float(np.linalg.norm(points[1] - points[0])) + float(np.linalg.norm(points[2] - points[3]))
    ) / 2.0
    width_side_px = (
        float(np.linalg.norm(points[2] - points[1])) + float(np.linalg.norm(points[3] - points[0]))
    ) / 2.0
    if length_side_px <= 1e-9 or width_side_px <= 1e-9:
        return None

    image_aspect = length_side_px / width_side_px
    ground_aspect = length_m / width_m
    aspect_penalty = abs(math.log(image_aspect / ground_aspect))
    return area / (1.0 + aspect_penalty)


def _coerce_corners(
    corners_px: Sequence[object],
    length_m: float | None = None,
    width_m: float | None = None,
) -> ManualCornerOrder:
    if len(corners_px) != 4:
        raise ValueError("Manual calibration requires exactly four road corner points")
    points = np.asarray(
        [_xy_point(corner, f"corner {index}") for index, corner in enumerate(corners_px, start=1)],
        dtype=np.float64,
    )
    _validate_point_set(points, "Manual calibration image")

    candidates = (
        ("perimeter", (0, 1, 2, 3)),
        ("paired_width_edges", (0, 2, 3, 1)),
        ("paired_length_edges", (0, 1, 3, 2)),
    )
    scored: list[tuple[float, str, tuple[int, int, int, int], np.ndarray]] = []
    for order_name, order in candidates:
        ordered = points[list(order)]
        score = _corner_order_score(ordered, length_m=length_m, width_m=width_m)
        if score is not None:
            scored.append((score, order_name, order, ordered))
    if not scored:
        raise ValueError("Manual calibration image points must form a non-degenerate quadrilateral")

    _, order_name, order, ordered = max(scored, key=lambda item: item[0])
    return ManualCornerOrder(
        image_points=ordered,
        order_name=order_name,
        source_indices=(order[0] + 1, order[1] + 1, order[2] + 1, order[3] + 1),
    )


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

    corner_order = _coerce_corners(corners_px, length_m=length, width_m=width)
    image_points = corner_order.image_points
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
        "manual_corner_order": corner_order.order_name,
        "manual_corner_order_source_indices": list(corner_order.source_indices),
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


def _manual_measurement_bounds(calibration: Mapping[str, Any]) -> tuple[float, float]:
    length = _finite_float(calibration.get("manual_road_length_m"), "manual_road_length_m")
    width = _finite_float(calibration.get("manual_road_width_m"), "manual_road_width_m")
    if length <= 0 or width <= 0:
        raise ValueError("manual road length and width must be greater than zero")
    return length, width


def _inverse_homography(calibration: Mapping[str, Any]) -> np.ndarray:
    homography = np.asarray(calibration["H_image_to_ground"], dtype=np.float64)
    try:
        return np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:
        raise ValueError("manual homography is not invertible") from exc


def _axis_values(max_value: float, spacing: float) -> list[float]:
    values = list(np.arange(0.0, max_value + spacing * 0.5, spacing, dtype=np.float64))
    if not values or not math.isclose(values[-1], max_value, rel_tol=0.0, abs_tol=1e-9):
        values.append(max_value)
    return sorted(set(round(float(value), 9) for value in values if 0.0 <= value <= max_value))


def manual_measurement_grid_lines(
    calibration: Mapping[str, Any],
    grid_size_m: float = 1.0,
) -> pd.DataFrame:
    """Return image-space floor-grid preview lines for a manual calibration."""

    spacing = _finite_float(grid_size_m, "grid_size_m")
    if spacing <= 0:
        raise ValueError("grid_size_m must be greater than zero")

    length, width = _manual_measurement_bounds(calibration)
    ground_to_image = _inverse_homography(calibration)
    rows: list[dict[str, object]] = []

    boundary_ground = np.asarray(
        [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width], [0.0, 0.0]],
        dtype=np.float64,
    )
    boundary_image = apply_homography(boundary_ground, ground_to_image)
    rows.append(
        {
            "xs": boundary_image[:, 0].astype(float).tolist(),
            "ys": boundary_image[:, 1].astype(float).tolist(),
            "kind": "boundary",
            "color": "#e0a21a",
            "line_width": 3.0,
            "alpha": 0.95,
        }
    )

    for x_value in _axis_values(length, spacing):
        ground_line = np.asarray([[x_value, 0.0], [x_value, width]], dtype=np.float64)
        image_line = apply_homography(ground_line, ground_to_image)
        rows.append(
            {
                "xs": image_line[:, 0].astype(float).tolist(),
                "ys": image_line[:, 1].astype(float).tolist(),
                "kind": "grid",
                "color": "#f1bf4f",
                "line_width": 1.4,
                "alpha": 0.7,
            }
        )

    for y_value in _axis_values(width, spacing):
        ground_line = np.asarray([[0.0, y_value], [length, y_value]], dtype=np.float64)
        image_line = apply_homography(ground_line, ground_to_image)
        rows.append(
            {
                "xs": image_line[:, 0].astype(float).tolist(),
                "ys": image_line[:, 1].astype(float).tolist(),
                "kind": "grid",
                "color": "#f1bf4f",
                "line_width": 1.4,
                "alpha": 0.7,
            }
        )

    return pd.DataFrame(rows, columns=["xs", "ys", "kind", "color", "line_width", "alpha"])


def _clip_segment_to_measurement(
    start_ground: np.ndarray,
    end_ground: np.ndarray,
    length_m: float,
    width_m: float,
) -> tuple[float, float] | None:
    x0, y0 = float(start_ground[0]), float(start_ground[1])
    x1, y1 = float(end_ground[0]), float(end_ground[1])
    dx = x1 - x0
    dy = y1 - y0
    u_enter = 0.0
    u_exit = 1.0

    for p_value, q_value in (
        (-dx, x0),
        (dx, length_m - x0),
        (-dy, y0),
        (dy, width_m - y0),
    ):
        if math.isclose(p_value, 0.0, rel_tol=0.0, abs_tol=1e-12):
            if q_value < 0.0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            u_enter = max(u_enter, ratio)
        else:
            u_exit = min(u_exit, ratio)
        if u_enter > u_exit:
            return None

    return max(0.0, u_enter), min(1.0, u_exit)


def _point_from_ground(
    ground_point: np.ndarray,
    ground_to_image: np.ndarray,
    elapsed_s: float,
) -> dict[str, float]:
    image_point = apply_homography(np.asarray([ground_point], dtype=np.float64), ground_to_image)[0]
    return {
        "x": float(image_point[0]),
        "y": float(image_point[1]),
        "elapsed_s": float(elapsed_s),
    }


def _append_clipped_point(
    path: list[dict[str, float]],
    point: dict[str, float],
    min_interval_s: float,
) -> None:
    if path:
        previous = path[-1]
        if math.hypot(previous["x"] - point["x"], previous["y"] - point["y"]) <= 1e-7:
            return
        if point["elapsed_s"] <= previous["elapsed_s"]:
            point = {**point, "elapsed_s": previous["elapsed_s"] + min_interval_s}
    path.append(point)


def clip_manual_paths_to_measurement(
    paths: Sequence[Sequence[object]],
    calibration: Mapping[str, Any],
    min_sample_interval_s: float = 0.1,
) -> list[list[dict[str, float]]]:
    """Trim manual paths to the measured floor rectangle and split re-entry segments."""

    min_interval = _finite_float(min_sample_interval_s, "min_sample_interval_s")
    if min_interval <= 0:
        raise ValueError("min_sample_interval_s must be greater than zero")

    length, width = _manual_measurement_bounds(calibration)
    image_to_ground = np.asarray(calibration["H_image_to_ground"], dtype=np.float64)
    ground_to_image = _inverse_homography(calibration)
    clipped_paths: list[list[dict[str, float]]] = []

    for raw_path in _coerce_paths(paths):
        image_points = np.asarray(
            [[point.image_x, point.image_y] for point in raw_path],
            dtype=np.float64,
        )
        ground_points = apply_homography(image_points, image_to_ground)
        current: list[dict[str, float]] = []

        for index in range(len(raw_path) - 1):
            start = raw_path[index]
            end = raw_path[index + 1]
            if end.elapsed_s < start.elapsed_s:
                raise ValueError("manual path timestamps must be non-decreasing")

            interval = _clip_segment_to_measurement(
                ground_points[index],
                ground_points[index + 1],
                length,
                width,
            )
            if interval is None:
                if len(current) >= 2:
                    clipped_paths.append(current)
                current = []
                continue

            u_start, u_end = interval
            ground_delta = ground_points[index + 1] - ground_points[index]
            time_delta = max(end.elapsed_s - start.elapsed_s, min_interval)
            segment_start_ground = ground_points[index] + ground_delta * u_start
            segment_end_ground = ground_points[index] + ground_delta * u_end
            segment_start = _point_from_ground(
                segment_start_ground,
                ground_to_image,
                start.elapsed_s + time_delta * u_start,
            )
            segment_end = _point_from_ground(
                segment_end_ground,
                ground_to_image,
                start.elapsed_s + time_delta * u_end,
            )

            if current:
                last = current[-1]
                if math.hypot(last["x"] - segment_start["x"], last["y"] - segment_start["y"]) > 1e-7:
                    if len(current) >= 2:
                        clipped_paths.append(current)
                    current = []

            _append_clipped_point(current, segment_start, min_interval)
            _append_clipped_point(current, segment_end, min_interval)

        if len(current) >= 2:
            clipped_paths.append(current)

    return clipped_paths


def manual_path_length_m(
    path: Sequence[object],
    calibration: Mapping[str, Any],
    clip_to_measurement: bool = False,
) -> float:
    paths = [path]
    if clip_to_measurement:
        paths = clip_manual_paths_to_measurement(paths, calibration)
    coerced = _coerce_paths(paths)
    homography = np.asarray(calibration["H_image_to_ground"], dtype=np.float64)
    length = 0.0
    for coerced_path in coerced:
        image_points = np.asarray(
            [[point.image_x, point.image_y] for point in coerced_path],
            dtype=np.float64,
        )
        ground_points = apply_homography(image_points, homography)
        length += float(np.sum(np.hypot(np.diff(ground_points[:, 0]), np.diff(ground_points[:, 1]))))
    return length


def manual_paths_to_detections(
    paths: Sequence[Sequence[object]],
    bbox_width_px: float = 24.0,
    bbox_height_px: float = 64.0,
    path_gap_s: float = 2.0,
    min_sample_interval_s: float = 0.1,
    preserve_timestamps: bool = False,
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

            timestamp_s = first_elapsed_s + relative_s if preserve_timestamps else base_s + relative_s
            rows.append(
                {
                    "timestamp_ms": round(timestamp_s * 1000.0),
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

    output = pd.DataFrame(rows, columns=SYNTHETIC_DETECTION_COLUMNS)
    if preserve_timestamps and not output.empty:
        output = output.sort_values(["timestamp_ms", "frame_id"], kind="stable").reset_index(drop=True)
        output["frame_id"] = np.arange(len(output), dtype=np.int64)
    return output


def manual_path_summaries(
    paths: Sequence[Sequence[object]],
    calibration: Mapping[str, Any],
    clip_to_measurement: bool = False,
) -> pd.DataFrame:
    if clip_to_measurement:
        paths = clip_manual_paths_to_measurement(paths, calibration)
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
                "start_s": float(path[0].elapsed_s),
                "end_s": float(path[-1].elapsed_s),
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
    clip_to_measurement: bool = True,
) -> FlowAnalysisResult:
    image_size = (
        image_metadata.get("image_width", image_metadata.get("width")),
        image_metadata.get("image_height", image_metadata.get("height")),
    )
    calibration = build_manual_calibration(corners_px, length_m, width_m, image_size)
    analysis_paths = (
        clip_manual_paths_to_measurement(paths, calibration) if clip_to_measurement else list(paths)
    )
    detections = manual_paths_to_detections(analysis_paths, preserve_timestamps=clip_to_measurement)
    return run_flow_analysis(detections, calibration, settings)
