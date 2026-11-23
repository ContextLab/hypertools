"""Session-wide test settings."""
import os
import sys

# plot(..., backend='plotly', show=True) outside a notebook calls fig.show(),
# and plotly's default renderer outside a notebook is the desktop browser:
# a local test run opened browser tabs on the maintainer's screen
# (2026-09-11). Keep every run headless unless the caller chose a renderer.
# Set before any test module imports plotly, which reads it at import.
os.environ.setdefault('PLOTLY_RENDERER', 'json')

# LSL tests: on macOS, liblsl also sends its discovery queries straight to
# every outlet port on this machine (tests/lsl_loopback.cfg; the reason and
# the measurements are under "Discovery" in the docstring of
# tests/test_lsl_streaming.py). liblsl reads LSLAPICFG once, at its first use
# in the process, so this is set before any test module imports pylsl; a
# caller's own LSLAPICFG is kept.
if sys.platform == 'darwin':
    os.environ.setdefault('LSLAPICFG', os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'lsl_loopback.cfg'))
