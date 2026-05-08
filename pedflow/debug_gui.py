from __future__ import annotations

try:
    from .debug_panel import build_debug_app
except ImportError:
    from pedflow.debug_panel import build_debug_app


app = build_debug_app()
app.servable()
