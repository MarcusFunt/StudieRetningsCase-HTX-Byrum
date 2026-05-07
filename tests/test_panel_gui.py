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
    assert top_tabs._names == ["Analysis", "Calibration"]
    assert isinstance(calibration_tabs, pn.Tabs)
    assert calibration_tabs._names == [
        "ChArUco Board",
        "Camera Intrinsics",
        "Ground Homography",
    ]
    assert "".join(("Jup", "yter")) not in top_tabs._names
