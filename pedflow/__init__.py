"""Offline tools for privacy-preserving pedestrian flow analysis."""

from .calibration import calibrate_camera_from_charuco, generate_charuco_board
from .geometry import apply_homography, bbox_foot_points, detections_to_ground, undistort_points
from .metrics import add_dwell_flags, estimate_speeds, summarize_flow
from .tracking import filter_short_tracks, link_detections

__all__ = [
    "add_dwell_flags",
    "apply_homography",
    "bbox_foot_points",
    "calibrate_camera_from_charuco",
    "detections_to_ground",
    "estimate_speeds",
    "filter_short_tracks",
    "generate_charuco_board",
    "link_detections",
    "summarize_flow",
    "undistort_points",
]
