from pathlib import Path

import pytest


def test_panel_gui_builds_template_without_running_analysis():
    pytest.importorskip("panel")
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")

    from pedflow.gui import build_app

    app = build_app(Path.cwd())

    assert app.title == "Pedestrian Flow Logger"
