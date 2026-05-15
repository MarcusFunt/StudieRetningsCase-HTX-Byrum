"""Offline tools for privacy-preserving pedestrian flow analysis."""

from .analysis import (
    FlowAnalysisResult,
    FlowAnalysisSettings,
    run_flow_analysis,
    write_analysis_outputs,
)
from .calibration import calibrate_camera_from_charuco, generate_charuco_board
from .geometry import (
    apply_homography,
    bbox_foot_points,
    detections_to_ground,
    undistort_points,
    validate_detection_input,
)
from .manual import build_manual_calibration, manual_paths_to_detections, run_manual_analysis
from .metrics import add_dwell_flags, estimate_speeds, summarize_flow
from .tracking import filter_short_tracks, link_detections

__all__ = [
    "FlowAnalysisResult",
    "FlowAnalysisSettings",
    "add_dwell_flags",
    "apply_homography",
    "bbox_foot_points",
    "build_manual_calibration",
    "calibrate_camera_from_charuco",
    "detections_to_ground",
    "estimate_speeds",
    "filter_short_tracks",
    "generate_charuco_board",
    "link_detections",
    "manual_paths_to_detections",
    "run_flow_analysis",
    "run_manual_analysis",
    "summarize_flow",
    "undistort_points",
    "validate_detection_input",
    "write_analysis_outputs",
]
