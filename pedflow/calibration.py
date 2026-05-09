from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .geometry import apply_homography, save_calibration, undistort_points

DEFAULT_CHARUCO_DICTIONARY = "DICT_5X5_100"
GROUND_MARKER_COLUMNS = ("image_x", "image_y", "ground_x_m", "ground_y_m")
_SAFE_BASENAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAX_SAFE_BASENAME_LENGTH = 120
_TRUNCATION_DIGEST_LENGTH = 10


def get_aruco_dictionary(dictionary_name: str = DEFAULT_CHARUCO_DICTIONARY) -> cv2.aruco.Dictionary:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support is unavailable. Install opencv-contrib-python.")
    if not dictionary_name.startswith("DICT_") or not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str = DEFAULT_CHARUCO_DICTIONARY,
) -> cv2.aruco.CharucoBoard:
    if squares_x < 3 or squares_y < 3:
        raise ValueError("A ChArUco board needs at least 3 squares in each direction")
    if marker_length_m <= 0 or square_length_m <= 0:
        raise ValueError("square and marker lengths must be positive")
    if marker_length_m >= square_length_m:
        raise ValueError("marker length must be smaller than square length")

    dictionary = get_aruco_dictionary(dictionary_name)
    return cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_length_m),
        float(marker_length_m),
        dictionary,
    )


def _safe_output_basename(basename: str) -> str:
    if not basename:
        raise ValueError("basename must not be empty")
    path = Path(basename)
    if path.is_absolute() or path.name != basename or basename in {".", ".."}:
        raise ValueError("basename must be a filename stem, not a path")
    if not _SAFE_BASENAME.fullmatch(basename):
        raise ValueError("basename may only contain letters, numbers, dots, dashes, and underscores")
    return _truncate_with_digest(basename, _MAX_SAFE_BASENAME_LENGTH)


def _truncate_with_digest(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:_TRUNCATION_DIGEST_LENGTH]
    prefix_length = max_length - len(digest) - 1
    if prefix_length < 1:
        raise ValueError("max_length is too short for deterministic truncation")
    return f"{value[:prefix_length]}-{digest}"


def charuco_board_from_metadata(metadata: dict) -> cv2.aruco.CharucoBoard:
    return create_charuco_board(
        squares_x=int(metadata["squares_x"]),
        squares_y=int(metadata["squares_y"]),
        square_length_m=float(metadata["square_length_m"]),
        marker_length_m=float(metadata["marker_length_m"]),
        dictionary_name=str(metadata["dictionary"]),
    )


def generate_charuco_board(
    output_dir: str | Path,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length_mm: float = 35.0,
    marker_length_mm: float = 25.0,
    dictionary_name: str = DEFAULT_CHARUCO_DICTIONARY,
    dpi: int = 300,
    margin_mm: float = 10.0,
    basename: str = "charuco_board",
) -> dict:
    """Generate printable ChArUco board assets and metadata.

    Length inputs are millimeters for user-facing print dimensions and stored as meters in metadata.
    Print the PDF at 100% scale so board geometry matches calibration measurements.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe_basename = _safe_output_basename(basename)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    square_length_m = square_length_mm / 1000.0
    marker_length_m = marker_length_mm / 1000.0
    board = create_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_length_m,
        marker_length_m=marker_length_m,
        dictionary_name=dictionary_name,
    )

    board_width_mm = squares_x * square_length_mm
    board_height_mm = squares_y * square_length_mm
    total_width_mm = board_width_mm + margin_mm * 2.0
    total_height_mm = board_height_mm + margin_mm * 2.0
    width_px = round(total_width_mm / 25.4 * dpi)
    height_px = round(total_height_mm / 25.4 * dpi)
    margin_px = round(margin_mm / 25.4 * dpi)

    image = board.generateImage((width_px, height_px), marginSize=margin_px, borderBits=1)

    png_path = output_path / f"{safe_basename}.png"
    pdf_path = output_path / f"{safe_basename}.pdf"
    metadata_path = output_path / f"{safe_basename}.json"

    if not cv2.imwrite(str(png_path), image):
        raise OSError(f"Could not write {png_path}")

    fig = plt.figure(figsize=(total_width_mm / 25.4, total_height_mm / 25.4), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    fig.savefig(pdf_path, dpi=dpi)
    plt.close(fig)

    metadata = {
        "schema_version": 1,
        "board_type": "charuco",
        "created_utc": datetime.now(UTC).isoformat(),
        "opencv_version": cv2.__version__,
        "dictionary": dictionary_name,
        "squares_x": int(squares_x),
        "squares_y": int(squares_y),
        "charuco_corners_x": int(squares_x - 1),
        "charuco_corners_y": int(squares_y - 1),
        "square_length_m": float(square_length_m),
        "marker_length_m": float(marker_length_m),
        "square_length_mm": float(square_length_mm),
        "marker_length_mm": float(marker_length_mm),
        "board_width_m": float(board_width_mm / 1000.0),
        "board_height_m": float(board_height_mm / 1000.0),
        "margin_mm": float(margin_mm),
        "margin_px": int(margin_px),
        "dpi": int(dpi),
        "image_width_px": int(width_px),
        "image_height_px": int(height_px),
        "marker_count": len(board.getIds()),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "metadata_path": str(metadata_path),
        "png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
        "print_instructions": "Print the PDF at 100% scale. Do not fit-to-page or shrink-to-margins.",
        "privacy_note": (
            "This board is for manual geometry calibration only. Calibration images should avoid "
            "pedestrians and should be deleted after calibration JSON has been verified."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    return metadata


def detect_charuco_image_points(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    min_corners: int = 8,
) -> tuple[np.ndarray, np.ndarray, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, marker_ids = detector.detectBoard(gray)
    marker_count = 0 if marker_ids is None else len(marker_ids)

    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_corners:
        raise ValueError(f"Detected fewer than {min_corners} ChArUco corners")

    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return object_points.astype(np.float32), image_points.astype(np.float32), marker_count


def _reprojection_rms_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    measured_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(projected_points) == 0:
        raise ValueError("Cannot calculate reprojection error without image points")
    if len(projected_points) != len(measured_points):
        raise ValueError("Projected and measured image point counts differ")
    residuals = projected_points - measured_points
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def _status_from_fail_warn(failed: bool, warned: bool) -> str:
    if failed:
        return "fail"
    if warned:
        return "warn"
    return "pass"


def intrinsics_quality_report(calibration: dict) -> dict:
    """Return a compact pass/warn/fail report for camera intrinsics calibration."""

    rms = _optional_float(calibration.get("rms_reprojection_error_px"))
    per_image_errors = [
        float(value)
        for value in calibration.get("per_image_reprojection_error_px", [])
        if np.isfinite(float(value))
    ]
    skipped_count = int(calibration.get("skipped_image_count", 0))
    used_count = len(calibration.get("used_images", []))
    max_per_image = max(per_image_errors) if per_image_errors else None
    mean_per_image = float(np.mean(per_image_errors)) if per_image_errors else None

    issues: list[str] = []
    failed = False
    warned = False
    if used_count < 3:
        failed = True
        issues.append("At least 3 usable calibration images are required.")
    if rms is None:
        failed = True
        issues.append("RMS reprojection error is missing.")
    elif rms > 2.0:
        failed = True
        issues.append("RMS reprojection error is above 2.0 px.")
    elif rms > 1.0:
        warned = True
        issues.append("RMS reprojection error is above 1.0 px.")
    if max_per_image is not None and max_per_image > 4.0:
        failed = True
        issues.append("At least one image has reprojection error above 4.0 px.")
    elif max_per_image is not None and max_per_image > 2.0:
        warned = True
        issues.append("At least one image has reprojection error above 2.0 px.")
    if skipped_count > 0:
        warned = True
        issues.append(f"{skipped_count} image(s) were skipped.")

    return {
        "status": _status_from_fail_warn(failed, warned),
        "rms_reprojection_error_px": rms,
        "max_per_image_reprojection_error_px": max_per_image,
        "mean_per_image_reprojection_error_px": mean_per_image,
        "used_image_count": used_count,
        "skipped_image_count": skipped_count,
        "issues": issues,
    }


def homography_quality_report(calibration: dict) -> dict:
    """Return a compact pass/warn/fail report for ground homography calibration."""

    mean_residual = _optional_float(calibration.get("ground_marker_mean_residual_m"))
    max_residual = _optional_float(calibration.get("ground_marker_max_residual_m"))
    marker_count = int(calibration.get("ground_marker_count", 0))
    inliers = calibration.get("ground_marker_inliers", [])
    inlier_ratio = (
        float(np.mean(np.asarray(inliers, dtype=bool))) if len(inliers) > 0 else None
    )

    issues: list[str] = []
    failed = False
    warned = False
    if marker_count < 4:
        failed = True
        issues.append("At least 4 ground markers are required.")
    if mean_residual is None or max_residual is None:
        failed = True
        issues.append("Ground marker residuals are missing.")
    else:
        if mean_residual > 0.10:
            failed = True
            issues.append("Mean homography residual is above 0.10 m.")
        elif mean_residual > 0.03:
            warned = True
            issues.append("Mean homography residual is above 0.03 m.")
        if max_residual > 0.25:
            failed = True
            issues.append("Max homography residual is above 0.25 m.")
        elif max_residual > 0.10:
            warned = True
            issues.append("Max homography residual is above 0.10 m.")
    if inlier_ratio is not None:
        if inlier_ratio < 0.60:
            failed = True
            issues.append("Less than 60% of ground markers are RANSAC inliers.")
        elif inlier_ratio < 0.80:
            warned = True
            issues.append("Less than 80% of ground markers are RANSAC inliers.")

    return {
        "status": _status_from_fail_warn(failed, warned),
        "ground_marker_count": marker_count,
        "ground_marker_inlier_ratio": inlier_ratio,
        "ground_marker_mean_residual_m": mean_residual,
        "ground_marker_max_residual_m": max_residual,
        "issues": issues,
    }


def calibration_quality_report(calibration: dict) -> dict:
    intrinsics = intrinsics_quality_report(calibration) if "K" in calibration else None
    homography = (
        homography_quality_report(calibration)
        if "H_image_to_ground" in calibration
        else None
    )
    statuses = [
        report["status"]
        for report in (intrinsics, homography)
        if report is not None
    ]
    overall = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return {
        "status": overall,
        "intrinsics": intrinsics,
        "homography": homography,
    }


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def calibrate_camera_from_charuco(
    image_paths: Iterable[str | Path],
    board_metadata_path: str | Path,
    min_corners: int = 8,
) -> dict:
    """Estimate camera intrinsics and distortion from ChArUco board images.

    All usable images must share one resolution. Board dimensions come from generated metadata and
    are interpreted in meters; OpenCV estimates the camera matrix and lens distortion coefficients.
    """

    with Path(board_metadata_path).open("r", encoding="utf-8") as file:
        board_metadata = json.load(file)

    board = charuco_board_from_metadata(board_metadata)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    skipped_images: list[dict[str, str]] = []
    detected_corner_counts: list[int] = []
    detected_marker_counts: list[int] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            skipped_images.append({"path": str(path), "reason": "unreadable"})
            continue

        try:
            obj_points, img_points, marker_count = detect_charuco_image_points(
                image,
                board,
                min_corners=min_corners,
            )
        except ValueError as exc:
            skipped_images.append({"path": str(path), "reason": str(exc)})
            continue

        current_image_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_image_size
        elif image_size != current_image_size:
            raise ValueError("All usable ChArUco calibration images must have the same resolution")

        object_points.append(obj_points)
        image_points.append(img_points)
        used_images.append(str(path))
        detected_corner_counts.append(len(img_points))
        detected_marker_counts.append(marker_count)

    if image_size is None:
        raise ValueError(
            f"No usable ChArUco calibration images found; skipped {len(skipped_images)} images"
        )
    if len(object_points) < 3:
        raise ValueError(
            "At least 3 usable ChArUco calibration images are required "
            f"({len(object_points)} usable, {len(skipped_images)} skipped)"
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    per_image_errors = []
    for obj_points, img_points, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs, strict=True
    ):
        per_image_errors.append(
            _reprojection_rms_error(obj_points, img_points, rvec, tvec, camera_matrix, dist_coeffs)
        )

    result = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "K": camera_matrix.tolist(),
        "dist": dist_coeffs.reshape(-1).tolist(),
        "calibration_board_type": "charuco",
        "charuco_board_metadata_path": str(board_metadata_path),
        "charuco_board": {
            key: board_metadata[key]
            for key in [
                "dictionary",
                "squares_x",
                "squares_y",
                "square_length_m",
                "marker_length_m",
                "board_width_m",
                "board_height_m",
                "marker_count",
            ]
            if key in board_metadata
        },
        "min_charuco_corners_per_image": int(min_corners),
        "detected_charuco_corners_per_image": detected_corner_counts,
        "detected_aruco_markers_per_image": detected_marker_counts,
        "skipped_images": skipped_images,
        "skipped_image_count": len(skipped_images),
        "rms_reprojection_error_px": float(rms),
        "per_image_reprojection_error_px": per_image_errors,
        "used_images": used_images,
        "calibration_notes": (
            "ChArUco images were manually captured for geometry calibration only. "
            "Delete calibration images after this file has been verified."
        ),
    }
    result["intrinsics_quality"] = intrinsics_quality_report(result)
    return result


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
    """Estimate camera intrinsics and distortion from checkerboard images.

    ``pattern_size`` is the inner-corner grid as ``(columns, rows)`` and ``square_size_m`` is in
    meters. All usable images must have the same resolution for OpenCV calibration.
    """

    object_template = checkerboard_object_points(pattern_size, square_size_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    skipped_images: list[dict[str, str]] = []
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
            skipped_images.append({"path": str(path), "reason": "unreadable"})
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            skipped_images.append({"path": str(path), "reason": "checkerboard_not_found"})
            continue

        current_image_size = (int(gray.shape[1]), int(gray.shape[0]))
        if image_size is None:
            image_size = current_image_size
        elif image_size != current_image_size:
            raise ValueError("All usable checkerboard calibration images must have the same resolution")

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        used_images.append(str(path))

    if image_size is None:
        raise ValueError(
            f"No usable checkerboard calibration images found; skipped {len(skipped_images)} images"
        )
    if len(object_points) < 3:
        raise ValueError(
            "At least 3 usable checkerboard images are required "
            f"({len(object_points)} usable, {len(skipped_images)} skipped)"
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    per_image_errors = []
    for obj_points, img_points, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs, strict=True
    ):
        per_image_errors.append(
            _reprojection_rms_error(obj_points, img_points, rvec, tvec, camera_matrix, dist_coeffs)
        )

    result = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "K": camera_matrix.tolist(),
        "dist": dist_coeffs.reshape(-1).tolist(),
        "checkerboard_pattern_size": list(pattern_size),
        "checkerboard_square_size_m": float(square_size_m),
        "skipped_images": skipped_images,
        "skipped_image_count": len(skipped_images),
        "rms_reprojection_error_px": float(rms),
        "per_image_reprojection_error_px": per_image_errors,
        "used_images": used_images,
        "calibration_notes": (
            "Images were manually captured for geometry calibration only. "
            "Delete calibration images after this file has been verified."
        ),
    }
    result["intrinsics_quality"] = intrinsics_quality_report(result)
    return result


def read_marker_csv(path: str | Path) -> pd.DataFrame:
    return _validate_marker_points(pd.read_csv(path))


def _validate_marker_points(markers: pd.DataFrame) -> pd.DataFrame:
    missing = set(GROUND_MARKER_COLUMNS).difference(markers.columns)
    if missing:
        raise ValueError(f"Missing marker columns: {sorted(missing)}")

    output = markers.copy()
    for column in GROUND_MARKER_COLUMNS:
        converted = pd.to_numeric(output[column], errors="coerce")
        values = converted.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"Marker column '{column}' must contain only finite numeric values")
        output[column] = converted

    if len(output) < 4:
        raise ValueError("At least 4 ground markers are required")

    _validate_marker_point_set(output[["image_x", "image_y"]], "image")
    _validate_marker_point_set(output[["ground_x_m", "ground_y_m"]], "ground")
    return output


def _validate_marker_point_set(points_frame: pd.DataFrame, label: str) -> None:
    points = points_frame.to_numpy(dtype=np.float64)
    unique_points = np.unique(points, axis=0)
    if len(unique_points) < 4:
        raise ValueError(f"At least 4 unique {label} marker points are required")
    centered = unique_points - unique_points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError(f"{label.capitalize()} marker points must not be collinear")


def _charuco_object_points_to_ground(
    object_points: np.ndarray,
    board_origin_x_m: float = 0.0,
    board_origin_y_m: float = 0.0,
    board_rotation_deg: float = 0.0,
) -> np.ndarray:
    object_array = np.asarray(object_points, dtype=np.float64)
    if object_array.ndim == 0 or object_array.size == 0 or object_array.shape[-1] < 2:
        raise ValueError("ChArUco object points must contain at least x/y coordinates")

    board_xy = object_array.reshape(-1, object_array.shape[-1])[:, :2]
    angle_rad = math.radians(float(board_rotation_deg))
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    rotation = np.array(
        [
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle],
        ],
        dtype=np.float64,
    )
    origin = np.array([float(board_origin_x_m), float(board_origin_y_m)], dtype=np.float64)
    return board_xy @ rotation.T + origin


def detect_charuco_ground_markers(
    image_path: str | Path,
    board_metadata_path: str | Path,
    board_origin_x_m: float = 0.0,
    board_origin_y_m: float = 0.0,
    board_rotation_deg: float = 0.0,
    min_corners: int = 8,
) -> tuple[pd.DataFrame, int]:
    """Detect ChArUco corners in a ground-plane photo and convert them to marker rows.

    The board must lie flat on the walking plane. ``board_origin_*`` is the ground coordinate
    of the board's local origin, and ``board_rotation_deg`` rotates board-local x/y coordinates
    counter-clockwise into the ground coordinate system.
    """

    with Path(board_metadata_path).open("r", encoding="utf-8") as file:
        board_metadata = json.load(file)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read ChArUco ground image: {image_path}")

    object_points, image_points, marker_count = detect_charuco_image_points(
        image,
        charuco_board_from_metadata(board_metadata),
        min_corners=min_corners,
    )
    image_xy = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    ground_xy = _charuco_object_points_to_ground(
        object_points,
        board_origin_x_m=board_origin_x_m,
        board_origin_y_m=board_origin_y_m,
        board_rotation_deg=board_rotation_deg,
    )
    marker_points = pd.DataFrame(
        {
            "image_x": image_xy[:, 0],
            "image_y": image_xy[:, 1],
            "ground_x_m": ground_xy[:, 0],
            "ground_y_m": ground_xy[:, 1],
        }
    )
    return _validate_marker_points(marker_points), marker_count


def compute_ground_homography(
    marker_points: pd.DataFrame,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    ransac_threshold_m: float = 0.10,
) -> dict:
    """Compute an image-to-ground homography from measured marker correspondences.

    Image marker points are undistorted with the camera intrinsics before fitting. Ground marker
    coordinates are meters on the walking plane, and the RANSAC threshold is also in meters.
    """

    marker_points = _validate_marker_points(marker_points)
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

    result = {
        "H_image_to_ground": homography.tolist(),
        "ground_marker_count": len(marker_points),
        "ground_marker_inliers": mask,
        "ground_marker_residuals_m": residuals.astype(float).tolist(),
        "ground_marker_mean_residual_m": float(np.mean(residuals)),
        "ground_marker_max_residual_m": float(np.max(residuals)),
    }
    result["homography_quality"] = homography_quality_report(result)
    return result


def compute_ground_homography_from_charuco(
    image_path: str | Path,
    board_metadata_path: str | Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board_origin_x_m: float = 0.0,
    board_origin_y_m: float = 0.0,
    board_rotation_deg: float = 0.0,
    min_corners: int = 8,
    ransac_threshold_m: float = 0.10,
) -> dict:
    """Compute image-to-ground homography from a flat ChArUco board ground photo."""

    marker_points, marker_count = detect_charuco_ground_markers(
        image_path=image_path,
        board_metadata_path=board_metadata_path,
        board_origin_x_m=board_origin_x_m,
        board_origin_y_m=board_origin_y_m,
        board_rotation_deg=board_rotation_deg,
        min_corners=min_corners,
    )
    homography = compute_ground_homography(
        marker_points=marker_points,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        ransac_threshold_m=ransac_threshold_m,
    )
    homography.update(
        {
            "ground_homography_source": "charuco_board",
            "ground_charuco_image_path": str(image_path),
            "ground_charuco_board_metadata_path": str(board_metadata_path),
            "ground_charuco_detected_corner_count": len(marker_points),
            "ground_charuco_detected_marker_count": int(marker_count),
            "ground_charuco_board_origin_m": [
                float(board_origin_x_m),
                float(board_origin_y_m),
            ],
            "ground_charuco_board_rotation_deg": float(board_rotation_deg),
        }
    )
    return homography


def merge_and_save_calibration(
    intrinsic_calibration: dict,
    homography_calibration: dict,
    output_path: str | Path,
) -> dict:
    calibration = {**intrinsic_calibration, **homography_calibration}
    calibration["calibration_quality"] = calibration_quality_report(calibration)
    save_calibration(calibration, output_path)
    return calibration
