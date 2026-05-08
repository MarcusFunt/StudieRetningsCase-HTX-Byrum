from __future__ import annotations

import hashlib
import json
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
    return basename


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


def calibrate_camera_from_charuco(
    image_paths: Iterable[str | Path],
    board_metadata_path: str | Path,
    min_corners: int = 8,
) -> dict:
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

        current_image_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_image_size
        elif image_size != current_image_size:
            raise ValueError("All ChArUco calibration images must have the same resolution")

        try:
            obj_points, img_points, marker_count = detect_charuco_image_points(
                image,
                board,
                min_corners=min_corners,
            )
        except ValueError as exc:
            skipped_images.append({"path": str(path), "reason": str(exc)})
            continue

        object_points.append(obj_points)
        image_points.append(img_points)
        used_images.append(str(path))
        detected_corner_counts.append(len(img_points))
        detected_marker_counts.append(marker_count)

    if image_size is None:
        raise ValueError(
            f"No ChArUco calibration images could be read; skipped {len(skipped_images)} images"
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

    return {
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
        current_image_size = (int(gray.shape[1]), int(gray.shape[0]))
        if image_size is None:
            image_size = current_image_size
        elif image_size != current_image_size:
            raise ValueError("All checkerboard calibration images must have the same resolution")

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            skipped_images.append({"path": str(path), "reason": "checkerboard_not_found"})
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        used_images.append(str(path))

    if image_size is None:
        raise ValueError(
            f"No checkerboard calibration images could be read; skipped {len(skipped_images)} images"
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

    return {
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


def compute_ground_homography(
    marker_points: pd.DataFrame,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    ransac_threshold_m: float = 0.10,
) -> dict:
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

    return {
        "H_image_to_ground": homography.tolist(),
        "ground_marker_count": len(marker_points),
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
