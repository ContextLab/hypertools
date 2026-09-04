# -*- coding: utf-8 -*-
"""GH #141 (follow-up): marker+line combo styles (e.g. 'o-') must get the
SAME connecting-line smoothing/interpolation as pure line styles (e.g.
'-') -- previously the interpolation step in `hypertools.plot.plot` was
gated on `is_line(fmt)`, which is False for ANY format string containing a
marker character, so 'o-' silently skipped interpolation entirely and drew
straight (unsmoothed) segments between raw points where '-' alone drew a
smoothed curve on identical data.

The fix draws marker+line combo styles as TWO matplotlib artists: a
smoothed line (from the interpolated data, no marker) plus markers at the
ORIGINAL (pre-interpolation) sample points (no connecting line) -- mirroring
how matplotlib conceptually treats 'o-' with dense interpolated lines while
keeping markers anchored to the true data. Real `hyp.plot()` calls only
(no mocks): every assertion below inspects actual matplotlib `Line3D`/
`Line2D` artists returned from a real (Agg-backed) render.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools._shared.helpers import has_line_component
# the data-faithful STATIC line interpolator (release-1.0 audit, F01-001):
# static smoothing now keeps every original sample as a drawn vertex and no
# longer depends on the animation frame_rate/duration kwargs, so the ground
# truth below mirrors plot.py by calling the same function.
from hypertools.plot.plot import _interp_static_line
from hypertools.tools.format_data import format_data
from hypertools.tools.analyze import analyze
from hypertools.reduce.reduce import reduce as reducer

def _lines_for(fmt, data, **kwargs):
    fig = hyp.plot(data, fmt, animate=False, show=False, **kwargs)
    lines = list(fig.axes[0].get_lines())
    plt.close('all')
    return lines


def _expected_line_and_raw(fmt, data, ndims=3):
    """Independently reproduce hyp.plot's own analyze -> reduce ->
    interpolate -> center/scale pipeline (GH #141), using the same public
    building blocks `hypertools.plot.plot` itself calls, so this is a
    faithful "ground truth" for BOTH the smoothed line's data AND the raw
    (pre-interpolation) marker data -- WITHOUT just re-reading plot.py's
    own internal variables. Comparing raw markers against the raw NUMPY
    INPUT directly (skipping analyze/reduce) is wrong -- the default
    reduce='IncrementalPCA' can rotate/reflect axes -- and comparing
    against center()/scale() of the PRE-interpolation data alone is ALSO
    wrong: the scale statistics (min/max) are computed from the
    INTERPOLATED (denser) trajectory, which can have slightly different
    extrema than the raw points (cubic PCHIP interpolation can overshoot
    past the original data's own min/max), and both the line and the raw
    markers must share that SAME scale so they land in one consistent
    coordinate frame.

    Returns (expected_line[0], expected_raw[0]) for the single dataset
    passed in `data` (a single 2D array).
    """
    raw = format_data([data])
    xform = analyze(raw, ndims=ndims, normalize=None, reduce='IncrementalPCA',
                    align=None, internal=True, impute=None)
    xform = reducer(xform, ndims=ndims, reduce='IncrementalPCA',
                    internal=True, format_data=False)
    raw_xform = [xi.copy() for xi in xform]

    n = xform[0].shape[0]
    if has_line_component(fmt) and n > 1:
        xform = [_interp_static_line(xi) for xi in xform]

    stacked = np.vstack(xform)
    mean = np.mean(stacked, 0)
    xform = [xi - mean for xi in xform]
    raw_xform = [xi - mean for xi in raw_xform]

    stacked = np.vstack(xform)
    m1 = np.min(stacked)
    m2 = np.max(stacked - m1)

    def rescale(a):
        return 2 * (np.divide(a - m1, m2)) - 1

    xform = [rescale(xi) for xi in xform]
    raw_xform = [rescale(xi) for xi in raw_xform]

    return xform[0], raw_xform[0]


def _line_artist(lines):
    """The artist with an actual connecting line (linestyle != 'None')."""
    matches = [ln for ln in lines if ln.get_linestyle() not in ('None', '')]
    assert len(matches) == 1
    return matches[0]


def _marker_artist(lines):
    """The artist with no connecting line (markers only)."""
    matches = [ln for ln in lines if ln.get_linestyle() in ('None', '')]
    assert len(matches) == 1
    return matches[0]


class TestComboStyleGetsLineSmoothing:
    """'o-' and '-' must draw the identical smoothed line."""

    def test_3d_line_vertex_count_matches_pure_line(self):
        rng = np.random.default_rng(0)
        data = np.cumsum(rng.standard_normal((20, 3)), axis=0)

        dash_lines = _lines_for('-', [data.copy()])
        combo_lines = _lines_for('o-', [data.copy()])

        assert len(dash_lines) == 1
        # combo style draws 2 artists: the smoothed line + raw markers
        assert len(combo_lines) == 2

        combo_line = _line_artist(combo_lines)
        dash_xdata = np.column_stack(dash_lines[0].get_data_3d())
        combo_xdata = np.column_stack(combo_line.get_data_3d())

        # same interpolated vertex count AND same coordinates -- 'o-' now
        # gets exactly the same line smoothing '-' alone gets.
        assert dash_xdata.shape == combo_xdata.shape
        assert dash_xdata.shape[0] > data.shape[0]  # actually interpolated
        assert np.allclose(dash_xdata, combo_xdata)

    def test_2d_line_vertex_count_matches_pure_line(self):
        rng = np.random.default_rng(1)
        data = np.cumsum(rng.standard_normal((15, 2)), axis=0)

        dash_lines = _lines_for('-', [data.copy()], ndims=2)
        combo_lines = _lines_for('o-', [data.copy()], ndims=2)

        combo_line = _line_artist(combo_lines)
        dash_xdata = np.column_stack(dash_lines[0].get_data())
        combo_xdata = np.column_stack(combo_line.get_data())

        assert dash_xdata.shape == combo_xdata.shape
        assert np.allclose(dash_xdata, combo_xdata)

    def test_dash_dot_combo_also_smoothed(self):
        rng = np.random.default_rng(2)
        data = np.cumsum(rng.standard_normal((12, 3)), axis=0)

        dashdot_lines = _lines_for('-.', [data.copy()])
        combo_lines = _lines_for('o-.', [data.copy()])

        combo_line = _line_artist(combo_lines)
        dashdot_xdata = np.column_stack(dashdot_lines[0].get_data_3d())
        combo_xdata = np.column_stack(combo_line.get_data_3d())
        assert dashdot_xdata.shape == combo_xdata.shape
        assert np.allclose(dashdot_xdata, combo_xdata)


class TestComboStyleMarkersAtRawPoints:
    """Markers in a combo style must sit at the TRUE (pre-interpolation)
    sample points, not the smoothed line's dense interpolated points."""

    def test_3d_marker_positions_equal_raw_data(self):
        rng = np.random.default_rng(3)
        data = np.cumsum(rng.standard_normal((15, 3)), axis=0)
        n = data.shape[0]

        combo_lines = _lines_for('o-', [data.copy()])
        _, expected_raw = _expected_line_and_raw('o-', data)
        marker = _marker_artist(combo_lines)
        marker_xdata = np.column_stack(marker.get_data_3d())

        assert marker_xdata.shape[0] == n
        assert np.allclose(marker_xdata, expected_raw)

    def test_2d_marker_positions_equal_raw_data(self):
        rng = np.random.default_rng(4)
        data = np.cumsum(rng.standard_normal((10, 2)), axis=0)
        n = data.shape[0]

        combo_lines = _lines_for('o-', [data.copy()], ndims=2)
        _, expected_raw = _expected_line_and_raw('o-', data, ndims=2)
        marker = _marker_artist(combo_lines)
        marker_xdata = np.column_stack(marker.get_data())

        assert marker_xdata.shape[0] == n
        assert np.allclose(marker_xdata, expected_raw)

    def test_marker_only_style_unaffected(self):
        """Plain 'o' (no line component) must still render exactly ONE
        artist at the raw sample points -- no smoothing, no split."""
        rng = np.random.default_rng(5)
        data = np.cumsum(rng.standard_normal((10, 3)), axis=0)
        n = data.shape[0]

        lines = _lines_for('o', [data.copy()])
        _, expected_raw = _expected_line_and_raw('o', data)
        assert len(lines) == 1
        xdata = np.column_stack(lines[0].get_data_3d())
        assert xdata.shape[0] == n
        assert np.allclose(xdata, expected_raw)

    def test_pure_line_style_unaffected(self):
        """Plain '-' must still render exactly ONE (smoothed) artist."""
        rng = np.random.default_rng(6)
        data = np.cumsum(rng.standard_normal((10, 3)), axis=0)

        lines = _lines_for('-', [data.copy()])
        assert len(lines) == 1
        xdata = np.column_stack(lines[0].get_data_3d())
        assert xdata.shape[0] > data.shape[0]


class TestComboStyleMultipleDatasets:
    def test_two_datasets_each_split_independently(self):
        rng = np.random.default_rng(7)
        data = [np.cumsum(rng.standard_normal((12, 3)), axis=0)
               for _ in range(2)]

        lines = _lines_for('o-', [d.copy() for d in data])
        # 2 datasets x 2 artists each (line + markers)
        assert len(lines) == 4
