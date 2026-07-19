# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pytest

from hypertools.plot.interactive import (
    detect_environment,
    resolve_backend,
    plotly_draw,
    _parse_fmt,
    _camera_eye,
)
from hypertools.plot.plot import plot


walk = np.cumsum(np.random.default_rng(0).standard_normal((80, 5)), axis=0)


def test_detect_environment_local():
    # tests run neither on Colab nor Kaggle
    assert detect_environment() == 'other'


def test_resolve_backend_auto_is_matplotlib_locally():
    assert resolve_backend('auto') == 'matplotlib'


def test_resolve_backend_explicit():
    assert resolve_backend('matplotlib') == 'matplotlib'
    assert resolve_backend('plotly') == 'plotly'  # plotly installed via [dev]


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_backend('bokeh')


def test_resolve_backend_auto_on_colab(monkeypatch):
    # Colab is detected via the google.colab module marker; registering the
    # name in sys.modules reproduces exactly what the Colab runtime does
    monkeypatch.setitem(sys.modules, 'google.colab', sys)
    assert detect_environment() == 'colab'
    assert resolve_backend('auto') == 'plotly'


def test_resolve_backend_auto_on_kaggle(monkeypatch):
    monkeypatch.setenv('KAGGLE_KERNEL_RUN_TYPE', 'Interactive')
    assert detect_environment() == 'kaggle'
    assert resolve_backend('auto') == 'plotly'


def test_parse_fmt():
    assert _parse_fmt('-', {}) == ('lines', 'circle', 'solid')
    assert _parse_fmt('o', {}) == ('markers', 'circle', 'solid')
    assert _parse_fmt('.-', {}) == ('lines+markers', 'circle', 'solid')
    assert _parse_fmt('--', {}) == ('lines', 'circle', 'dash')
    assert _parse_fmt(':', {}) == ('lines', 'circle', 'dot')
    assert _parse_fmt('-.', {}) == ('lines', 'circle', 'dashdot')
    assert _parse_fmt('s', {}) == ('markers', 'square', 'solid')
    assert _parse_fmt(None, {}) == ('lines', 'circle', 'solid')
    # explicit kwargs win over the format string
    assert _parse_fmt('-', {'marker': 'D'})[1] == 'diamond'
    assert _parse_fmt('-', {'linestyle': '--'})[2] == 'dash'


def test_camera_eye_matches_matplotlib_convention():
    eye = _camera_eye(90, 0)  # looking straight down
    assert abs(eye['x']) < 1e-9 and abs(eye['y']) < 1e-9 and eye['z'] > 0


# every printable matplotlib marker character (matplotlib.lines.Line2D.markers
# minus the line/None sentinels and non-character marker keys)
ALL_MPL_MARKERS = list('.,ov^<>1234spP*hH+xXDd|_')


def test_pixel_marker_is_markers_not_lines():
    # regression: the ',' (pixel) marker was omitted from the plotly marker
    # table, so it fell through to mode='lines' (GH plotly-marker bug). Every
    # marker-only fmt must yield a markers mode with no line.
    mode, _symbol, _dash = _parse_fmt(',', {})
    assert mode == 'markers'


def test_all_matplotlib_markers_recognized_as_markers_plotly():
    # EVERY matplotlib marker char must map to a plotly 'markers' mode
    for m in ALL_MPL_MARKERS:
        mode, symbol, _dash = _parse_fmt(m, {})
        assert mode == 'markers', f"marker {m!r} parsed as {mode!r}, not markers"
        assert symbol, f"marker {m!r} produced empty plotly symbol"


def test_plotly_marker_plot_end_to_end_pixel():
    # full pipeline: hyp.plot with the pixel marker must produce marker traces
    d = [np.random.default_rng(0).random((20, 3)) for _ in range(2)]
    fig = plot(d, ',', backend='plotly', show=False)
    assert fig.data[0].mode == 'markers'


def test_marker_vs_line_parity_both_backends():
    # matplotlib and plotly must AGREE: a marker-only fmt draws markers (no
    # line) on both backends. matplotlib is the reference (Line2D handles all
    # markers); plotly must match.
    import matplotlib.pyplot as plt
    d = [np.random.default_rng(1).random((20, 3)) for _ in range(2)]
    for m in ALL_MPL_MARKERS:
        mpl_fig = plot(d, m, backend='matplotlib', show=False)
        line = mpl_fig.axes[0].lines[0]
        assert str(line.get_marker()) == m
        assert line.get_linestyle() in ('None', 'none', '')
        plt.close('all')

        plotly_fig = plot(d, m, backend='plotly', show=False)
        assert plotly_fig.data[0].mode == 'markers', (
            f"plotly rendered marker {m!r} as {plotly_fig.data[0].mode!r}")


def test_plotly_draw_3d():
    fig = plotly_draw([walk[:, :3], walk[:, :3] + 2], show=False)
    # 2 data traces + 1 wireframe-cube trace (matches matplotlib's frame)
    assert len(fig.data) == 3
    assert fig.data[0].type == 'scatter3d'
    assert fig.data[-1].mode == 'lines'
    assert fig.data[-1].line.color == 'black'
    # hypertools aesthetic: axes fully hidden, unit cube range
    assert fig.layout.scene.xaxis.visible is False
    assert tuple(fig.layout.scene.xaxis.range) == (-1, 1)
    # matches matplotlib's default 4:4:3 Axes3D box aspect
    assert fig.layout.scene.aspectmode == 'manual'
    assert fig.layout.scene.aspectratio.z == 0.75


def test_plotly_draw_2d():
    fig = plotly_draw([walk[:, :2]], fmt=['o'], show=False)
    assert fig.data[0].type == 'scatter'
    assert fig.data[0].mode == 'markers'


def test_plot_backend_plotly_end_to_end():
    fig = plot(walk, backend='plotly', show=False)
    assert type(fig).__module__.startswith('plotly')
    # transformed data is still available via return_model
    result = plot(walk, backend='plotly', show=False, return_model=True)
    assert result['xform_data'] is not None


def test_plot_backend_plotly_animate_frames():
    fig = plot(walk, backend='plotly', animate=True, show=False)
    assert len(fig.frames) > 0
    fig = plot(walk, backend='plotly', animate='spin', show=False)
    assert len(fig.frames) > 0


def test_plot_backend_matplotlib_unchanged():
    fig = plot(walk, backend='matplotlib', show=False)
    assert type(fig).__module__.startswith('matplotlib')
    import matplotlib.pyplot as plt
    plt.close('all')
