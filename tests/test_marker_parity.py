# -*- coding: utf-8 -*-
"""Marker-size parity between the matplotlib and plotly backends (R2 fix,
maintainer request: "dots should be smaller in plotly (to match
matplotlib)"). No mocks -- the parity tests render BOTH backends for real
(via `scripts/measure_marker_parity.py`, matplotlib `Agg` + plotly/kaleido)
and measure actual rendered pixel diameters.

Background: matplotlib renders at dpi=100 (never overridden by this
codebase), so 1 point = 100/72 px (`PT_TO_PX`). matplotlib's '.'/','
marker glyphs are additionally defined at HALF the path scale of 'o' and
most other marker characters (`matplotlib.markers.MarkerStyle('.')
.get_transform()` gives a `0.25` scale vs. `0.5` for 'o') -- so at the
SAME `markersize`, '.' renders at half the diameter of 'o'. plotly has no
equivalently-tiny "dot" symbol (both map to plotly's 'circle'), so
`_marker_size_px` applies `_DOT_MARKER_SCALE` explicitly for '.'/','.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hypertools.plot.plotly_backend import (
    DEFAULT_MARKERSIZE_PT,
    MORPH_DEFAULT_MARKERSIZE_PT,
    PT_TO_PX,
    _DOT_MARKER_SCALE,
    _SCATTER3D_SIZE_FACTOR,
    _marker_size_px,
    _resolve_fmt,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from measure_marker_parity import (  # noqa: E402
    measure_diameter,
    render_mpl_marker,
    render_mpl_marker_3d,
    render_plotly_marker,
    render_plotly_marker_3d,
)


class TestMarkerSizeFormulaExact:
    """Exact-formula unit tests for `_marker_size_px` -- no rendering."""

    def test_pt_to_px_is_dpi100_over_72(self):
        assert PT_TO_PX == pytest.approx(100.0 / 72.0)

    def test_dot_marker_scale_is_half(self):
        assert _DOT_MARKER_SCALE == pytest.approx(0.5)

    def test_o_marker_no_discount(self):
        assert _marker_size_px(6.0, 'o') == pytest.approx(6.0 * PT_TO_PX)

    def test_dot_marker_gets_half_discount(self):
        assert _marker_size_px(6.0, '.') == pytest.approx(
            6.0 * PT_TO_PX * 0.5)

    def test_comma_marker_gets_half_discount(self):
        assert _marker_size_px(6.0, ',') == pytest.approx(
            6.0 * PT_TO_PX * 0.5)

    def test_other_markers_no_discount(self):
        for ch in ('s', '^', 'v', 'D', '+', 'x', '*'):
            assert _marker_size_px(6.0, ch) == pytest.approx(6.0 * PT_TO_PX)

    def test_none_marker_char_no_discount(self):
        assert _marker_size_px(6.0, None) == pytest.approx(6.0 * PT_TO_PX)

    def test_ndims3_divides_by_scatter3d_factor(self):
        px_2d = _marker_size_px(6.0, 'o', ndims=2)
        px_3d = _marker_size_px(6.0, 'o', ndims=3)
        assert px_3d == pytest.approx(px_2d / _SCATTER3D_SIZE_FACTOR)

    def test_ndims3_and_dot_marker_compose(self):
        px_3d = _marker_size_px(6.0, '.', ndims=3)
        expected = 6.0 * PT_TO_PX * _DOT_MARKER_SCALE / _SCATTER3D_SIZE_FACTOR
        assert px_3d == pytest.approx(expected)

    def test_default_ndims_is_2d_no_correction(self):
        assert _marker_size_px(6.0, 'o') == pytest.approx(
            _marker_size_px(6.0, 'o', ndims=2))

    def test_scales_linearly_with_markersize(self):
        for ms in (2.0, 6.0, 10.0, 20.0):
            assert _marker_size_px(ms, 'o') == pytest.approx(ms * PT_TO_PX)

    def test_resolve_fmt_dot_char_feeds_marker_size(self):
        # '.', the fmt hypertools' own density=/morph examples pass, must
        # resolve to a marker_char of '.' (not None/'o') so the dot
        # discount actually engages end to end.
        _mode, _symbol, _dash, marker_char = _resolve_fmt('.', {})
        assert marker_char == '.'

    def test_morph_default_markersize_is_4_not_6(self):
        # the ONE default both backends read (`morph.
        # MORPH_DEFAULT_MARKERSIZE_PT`) -- NOT the general 6.0pt default.
        # (1.1 visual review L9: this used to pin 1.5pt, which drew
        # sub-pixel dots on plotly -- 30 morph dots covered 24 px.)
        from hypertools.plot import morph
        assert MORPH_DEFAULT_MARKERSIZE_PT == pytest.approx(4.0)
        assert MORPH_DEFAULT_MARKERSIZE_PT == morph.MORPH_DEFAULT_MARKERSIZE_PT
        assert MORPH_DEFAULT_MARKERSIZE_PT != DEFAULT_MARKERSIZE_PT


class TestMarkerSizeEmpiricalParity:
    """Real renders (matplotlib Agg + plotly/kaleido), no mocks: for both
    of hypertools' calibration marker characters ('o' and '.'), the
    rendered plotly dot diameter must match matplotlib's within 20%."""

    @pytest.mark.parametrize('marker,markersize', [
        ('o', 6.0), ('o', 10.0), ('.', 6.0), ('.', 10.0),
    ])
    def test_plotly_diameter_within_20pct_of_matplotlib(
            self, marker, markersize, tmp_path):
        mpl_path = str(tmp_path / f'mpl_{marker}_{markersize}.png')
        plotly_path = str(tmp_path / f'plotly_{marker}_{markersize}.png')

        render_mpl_marker(markersize, marker, mpl_path)
        mpl_w, mpl_h = measure_diameter(mpl_path)
        mpl_diameter = (mpl_w + mpl_h) / 2.0

        plotly_size_px = _marker_size_px(markersize, marker)
        render_plotly_marker(plotly_size_px, plotly_path)
        plotly_w, plotly_h = measure_diameter(plotly_path)
        plotly_diameter = (plotly_w + plotly_h) / 2.0

        assert mpl_diameter > 0 and plotly_diameter > 0
        rel_error = abs(plotly_diameter - mpl_diameter) / mpl_diameter
        assert rel_error <= 0.20, (
            f"marker={marker!r} markersize={markersize}: mpl diameter="
            f"{mpl_diameter}px, plotly diameter={plotly_diameter}px "
            f"(rel_error={rel_error:.2%})"
        )


class TestMarkerSizeEmpiricalParity3D:
    """As `TestMarkerSizeEmpiricalParity`, but through `Axes3D.plot` /
    `go.Scatter3d` -- the code path hypertools' 3-D data/trail/morph
    traces actually use. This is the regression that the flat (2-D-only)
    parity test above CANNOT catch: `go.Scatter3d`'s `marker.size` has a
    different (~1.776x) rendered-pixel-diameter relationship than
    `go.Scatter`'s (see `_SCATTER3D_SIZE_FACTOR`'s docstring) -- without
    dividing by it, plotly's 3-D dots still rendered ~1.8x fatter than
    matplotlib's even after the 2-D-calibrated conversion."""

    @pytest.mark.parametrize('marker,markersize', [
        ('o', 6.0), ('o', 10.0), ('.', 6.0), ('.', 10.0),
    ])
    def test_plotly_scatter3d_diameter_within_20pct_of_matplotlib(
            self, marker, markersize, tmp_path):
        mpl_path = str(tmp_path / f'mpl3d_{marker}_{markersize}.png')
        plotly_path = str(tmp_path / f'plotly3d_{marker}_{markersize}.png')

        render_mpl_marker_3d(markersize, marker, mpl_path)
        mpl_w, mpl_h = measure_diameter(mpl_path)
        mpl_diameter = (mpl_w + mpl_h) / 2.0

        plotly_size_px = _marker_size_px(markersize, marker, ndims=3)
        render_plotly_marker_3d(plotly_size_px, plotly_path)
        plotly_w, plotly_h = measure_diameter(plotly_path)
        plotly_diameter = (plotly_w + plotly_h) / 2.0

        assert mpl_diameter > 0 and plotly_diameter > 0
        rel_error = abs(plotly_diameter - mpl_diameter) / mpl_diameter
        assert rel_error <= 0.20, (
            f"marker={marker!r} markersize={markersize}: mpl3d diameter="
            f"{mpl_diameter}px, plotly3d diameter={plotly_diameter}px "
            f"(rel_error={rel_error:.2%})"
        )


class TestObservationMarkerParity3D:
    """A marker on each OBSERVATION of a smoothed 3-D line ('o-' and every
    other line+marker fmt, `plotly_backend._observation_marker`) is a
    per-vertex size ARRAY -- and `go.Scatter3d` renders an array size at
    HALF the diameter of the same scalar size (measured in kaleido: 8 ->
    7.25 px vs 14.25 px, 16 -> 14.25 vs 28.75, 64 -> 57 vs 114). The
    array was filled with the scalar-calibrated size, so 3-D 'o-' markers
    drew at half matplotlib's diameter: 2 px dots under a 1.4 px line,
    hidden (1.1 release review). 2-D `go.Scatter` arrays render at the
    scalar size, so only 3-D is corrected."""

    @pytest.mark.parametrize('markersize', [3.0, 6.0, 12.0])
    def test_observation_marker_diameter_within_20pct_of_matplotlib(
            self, markersize, tmp_path):
        import numpy as np
        from hypertools.plot.plotly_backend import _observation_marker
        mpl_path = str(tmp_path / f'mpl3d_obs_{markersize}.png')
        plotly_path = str(tmp_path / f'plotly3d_obs_{markersize}.png')

        render_mpl_marker_3d(markersize, 'o', mpl_path)
        mpl_w, mpl_h = measure_diameter(mpl_path)
        mpl_diameter = (mpl_w + mpl_h) / 2.0

        marker = _observation_marker(
            dict(color='red', symbol='circle',
                 size=_marker_size_px(markersize, 'o', ndims=3)),
            n_vertices=5, vertices=[0, 4], ndims=3)
        assert not np.isscalar(marker['size'])     # the bubble form
        render_plotly_marker_3d(None, plotly_path, marker=marker)
        plotly_w, plotly_h = measure_diameter(plotly_path)
        plotly_diameter = (plotly_w + plotly_h) / 2.0

        assert mpl_diameter > 0 and plotly_diameter > 0
        rel_error = abs(plotly_diameter - mpl_diameter) / mpl_diameter
        assert rel_error <= 0.20, (
            f"markersize={markersize}: mpl3d diameter={mpl_diameter}px, "
            f"plotly3d observation-marker diameter={plotly_diameter}px "
            f"(rel_error={rel_error:.2%})")

    @pytest.mark.parametrize('ndims', [2, 3])
    def test_hyp_plot_line_marker_sizes_render_like_marker_only(self, ndims):
        """End to end: the observation markers of a 'o-' trace render at
        the same diameter as the plain markers of an 'o' trace (scalar
        size) with the same markersize -- in 3-D that is twice the scalar
        value, since Scatter3d halves an array size; in 2-D the same."""
        import numpy as np
        import hypertools as hyp
        rng = np.random.default_rng(0)
        a = rng.standard_normal((36, 5)).cumsum(0)
        lines = hyp.plot(a, '-o', markersize=3, ndims=ndims,
                         backend='plotly', show=False)
        dots = hyp.plot(a, 'o', markersize=3, ndims=ndims,
                        backend='plotly', show=False)
        arr = np.asarray(lines.data[0].marker.size, dtype=float)
        scalar = float(dots.data[0].marker.size)
        assert arr.ndim == 1 and (arr > 0).sum() == 36
        expected = scalar * (2.0 if ndims >= 3 else 1.0)
        assert np.allclose(arr[arr > 0], expected)
