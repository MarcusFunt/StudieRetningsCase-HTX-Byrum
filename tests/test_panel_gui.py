from pathlib import Path

import pytest


def test_panel_gui_builds_expected_analysis_and_calibration_tabs():
    pn = pytest.importorskip("panel")
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")

    from pedflow.gui import build_app

    app = build_app(Path.cwd())
    top_tabs = app.main[0]
    calibration_tabs = top_tabs.objects[1]

    assert app.title == "Pedestrian Flow Logger"
    assert isinstance(top_tabs, pn.Tabs)
    assert top_tabs._names == ["Analysis", "Calibration", "USB Debug", "Operations", "Manual Mode"]
    assert isinstance(calibration_tabs, pn.Tabs)
    assert calibration_tabs._names == [
        "ChArUco Board",
        "Camera Intrinsics",
        "Ground Homography",
    ]
    assert "".join(("Jup", "yter")) not in top_tabs._names


def test_primary_workflow_controls_are_selectors_instead_of_path_text_inputs():
    pn = pytest.importorskip("panel")

    from pedflow.debug_panel import UsbDebugPanel
    from pedflow.gui import AnalysisPanel, CalibrationPanel, ManualPanel
    from pedflow.operations_panel import OperationsPanel

    analysis = AnalysisPanel(Path.cwd())
    calibration = CalibrationPanel(Path.cwd())
    debug = UsbDebugPanel(Path.cwd())
    operations = OperationsPanel(Path.cwd())
    manual = ManualPanel(Path.cwd())

    assert isinstance(analysis.detections_path, pn.widgets.Select)
    assert isinstance(analysis.calibration_path, pn.widgets.Select)
    assert isinstance(calibration.intrinsics_image_dir, pn.widgets.Select)
    assert isinstance(calibration.image_dir, pn.widgets.Select)
    assert calibration.image_dir is calibration.intrinsics_image_dir
    assert isinstance(calibration.ground_image_dir, pn.widgets.Select)
    assert calibration.intrinsics_image_dir.value != calibration.ground_image_dir.value
    assert "data/calibration_images/ground" not in calibration.intrinsics_image_dir.options
    assert "data/calibration_images/charuco" not in calibration.ground_image_dir.options
    assert isinstance(calibration.capture_port, pn.widgets.Select)
    assert isinstance(calibration.capture_usb_photo_button, pn.widgets.Button)
    assert isinstance(calibration.ground_capture_port, pn.widgets.Select)
    assert isinstance(calibration.ground_capture_usb_photo_button, pn.widgets.Button)
    assert isinstance(calibration.ground_charuco_image_path, pn.widgets.Select)
    assert isinstance(calibration.ground_board_metadata_path, pn.widgets.Select)
    assert isinstance(calibration.compute_charuco_homography_button, pn.widgets.Button)
    assert isinstance(calibration.intrinsics_quality_table, pn.widgets.Tabulator)
    assert isinstance(calibration.intrinsics_image_errors_table, pn.widgets.Tabulator)
    assert isinstance(calibration.intrinsics_skipped_images_table, pn.widgets.Tabulator)
    assert isinstance(calibration.homography_quality_table, pn.widgets.Tabulator)
    assert isinstance(calibration.homography_residuals_table, pn.widgets.Tabulator)
    assert isinstance(calibration.intrinsics_path, pn.widgets.Select)
    assert isinstance(debug.port, pn.widgets.Select)
    assert isinstance(operations.capture_serial_port, pn.widgets.Select)
    assert isinstance(operations.image_port, pn.widgets.Select)
    assert isinstance(operations.analysis_detections_path, pn.widgets.Select)
    assert isinstance(manual.image_upload, pn.widgets.FileInput)
    assert isinstance(manual.road_length_m, pn.widgets.FloatInput)
    assert isinstance(manual.road_width_m, pn.widgets.FloatInput)
    assert manual.road_length_m.value == 6.0
    assert manual.road_width_m.value == 4.0
    assert isinstance(manual.output_dir, pn.widgets.Select)
    assert isinstance(manual.run_button, pn.widgets.Button)
    assert isinstance(manual.path_summary_table, pn.widgets.Tabulator)


def test_manual_panel_freehand_paths_sync_to_python_state():
    pytest.importorskip("panel")

    from pedflow.gui import ManualPanel

    manual = ManualPanel(Path.cwd())
    manual._image_size = (640, 360)
    manual._corners = [
        {"x": 0.0, "y": 0.0},
        {"x": 640.0, "y": 0.0},
        {"x": 640.0, "y": 360.0},
        {"x": 0.0, "y": 360.0},
    ]
    manual.canvas_mode.value = "Draw walking paths"
    manual._set_draw_tool_state()

    assert manual._freehand_tool.__class__.__name__ == "FreehandDrawTool"
    assert manual.figure.toolbar.active_drag is manual._freehand_tool

    manual._path_source.data = {
        "xs": [[100.0, 140.0, 220.0]],
        "ys": [[200.0, 220.0, 220.0]],
        "label": ["1"],
        "duration_s": [2.4],
    }

    assert len(manual._paths) == 1
    assert manual._paths[0][-1]["elapsed_s"] == pytest.approx(2.4)
    assert len(manual.path_summary_table.value) == 1
