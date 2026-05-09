# Outputs

The Panel dashboard writes calibration JSON files, processed analysis CSV files, visual QA exports, and run manifests here.

Generated outputs can be deleted and recreated from the anonymous detection CSV plus calibration JSON.

Analysis runs write:

- `detections_ground.csv`, `tracks.csv`, `track_summaries.csv`, `summary_metrics.csv`, and `grid_metrics.csv`
- `paths_qa`, `density_qa`, `speed_qa`, `dwell_qa`, and `bottleneck_qa` PNG/HTML files
- `analysis_manifest.json` with settings, start/end time, input paths, row counts, and output file paths

Calibration JSONs include pass/warn/fail quality sections for camera intrinsics and ground homography when those calibrations are present.

The ChArUco board generator writes the printable board PDF/PNG and metadata JSON to `outputs/charuco_board/` by default.
