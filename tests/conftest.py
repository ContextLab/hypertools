"""Session-wide test settings."""
import os

# plot(..., backend='plotly', show=True) outside a notebook calls fig.show(),
# and plotly's default renderer outside a notebook is the desktop browser:
# a local test run opened browser tabs on the maintainer's screen
# (2026-09-11). Keep every run headless unless the caller chose a renderer.
# Set before any test module imports plotly, which reads it at import.
os.environ.setdefault('PLOTLY_RENDERER', 'json')
