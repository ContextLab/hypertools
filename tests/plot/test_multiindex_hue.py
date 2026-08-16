# -*- coding: utf-8 -*-
"""Continuous hue through a COLUMN hierarchy.

Two accepted forms, both input-relative: flat length-T (broadcast), or one
sequence per leaf. A flat array sized to the TOTAL DRAWN observations is
rejected -- it is new API, not "existing behaviour", and it would require
the caller to know how many mean traces expansion creates.
"""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection

import hypertools as hyp


def market_frame(T=120, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Financials', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0) + 100.0, columns=cols)


def sector_prices(df):
    """The scalar the market example colours by: each sector's mean price."""
    return [df['Market'][s].mean(axis=1).to_numpy()
            for s in ('Tech', 'Financials', 'Energy')]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _collections(ax):
    """The per-segment-coloured DATA collections, in trace order.

    Not simply "the LineCollections on the axes": the 3-D bounding cube is
    six `Line3DCollection` wireframe faces (`matplotlib_backend._draw_cube`),
    so an untagged count is 6 too high -- measured, and the reason
    `_apply_multicolor_lines` tags each collection it creates with the trace
    it draws.
    """
    tagged = [c for c in ax.collections
              if isinstance(c, LineCollection)
              and getattr(c, '_hyp_trace_index', None) is not None]
    return sorted(tagged, key=lambda c: c._hyp_trace_index)


def test_flat_hue_is_broadcast_to_every_trace():
    df = market_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)), show=False)
    assert not [w for w in caught if 'ignoring hue' in str(w.message)]
    assert len(_collections(_ax(fig))) == 4


def test_nested_hue_supplies_one_vector_per_leaf():
    df = market_frame()
    fig = hyp.plot(df, '-', hue=sector_prices(df), show=False)
    assert len(_collections(_ax(fig))) == 4


def test_mean_trace_hue_is_the_mean_of_its_leaves():
    """The mean trace's colours are the EXACT colormap RGBA of the
    element-wise mean of its leaves' hue -- not merely "two colours differ",
    which any varying hue would satisfy.

    BOTH available checks are asserted, deliberately:

    1. **Exact RGBA.** The colour chain is pinned in Task 6 Step 3 and
       reproduced here from the same public helpers: `mat2colors` bins the
       CONCATENATION of every trace's aux (`_multicolor_line_colors`:
       "Colors are mapped over the CONCATENATED hue values so the scale is
       shared across datasets") with the default `n_bins=100`, and
       `_apply_multicolor_lines` gives each SEGMENT the midpoint of its two
       endpoints' colours. `antialias=False` keeps the point count at
       `len(df)`, so no interpolation intervenes and the arithmetic is
       exact.
    2. **The bundle's auxiliary array.** `trace_metadata['aux']` exposes the
       per-trace auxiliary values after the same co-truncation the data
       gets (Contract 6), so the mean-of-leaves rule is asserted on the
       numbers themselves and not only through the colormap.
    """
    from hypertools.plot.colors import mat2colors

    df = market_frame()
    hues = sector_prices(df)
    out = hyp.plot(df, '-', hue=hues, palette='viridis', antialias=False,
                   return_model=True, show=False)

    # (2) the numbers
    aux = out['trace_metadata']['aux']
    assert len(aux) == 4
    expected_hue = np.mean(np.stack([np.asarray(h, dtype=float)
                                     for h in hues]), axis=0)
    assert np.allclose(np.asarray(aux[-1], dtype=float), expected_hue)

    # (1) the exact RGBA those numbers must produce
    concatenated = np.concatenate([np.asarray(a, dtype=float) for a in aux])
    point_colors = mat2colors(concatenated, palette='viridis')
    start = sum(len(a) for a in aux[:-1])
    mean_points = point_colors[start:start + len(expected_hue)]
    expected_segments = (mean_points[:-1] + mean_points[1:]) / 2.0

    mean_colours = np.asarray(_collections(_ax(out['fig']))[-1].get_colors())
    assert len(mean_colours) == len(expected_segments)
    assert np.allclose(mean_colours[:, :3], expected_segments, atol=1e-6)


def test_hierarchy_still_sets_exact_widths_under_a_continuous_hue():
    df = market_frame()
    fig = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)), show=False)
    widths = [float(np.atleast_1d(c.get_linewidth())[0])
              for c in _collections(_ax(fig))]
    assert widths == pytest.approx([1.0, 1.0, 1.0, 2.0])


def test_hierarchy_still_sets_exact_alphas_under_a_continuous_hue():
    df = market_frame()
    fig = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)), show=False)
    alphas = [np.asarray(c.get_colors())[:, 3].max()
              for c in _collections(_ax(fig))]
    assert alphas == pytest.approx([0.7, 0.7, 0.7, 1.0])


def test_nested_hue_with_wrong_leaf_count_raises():
    df = market_frame()
    with pytest.raises(ValueError, match='one hue sequence per'):
        hyp.plot(df, '-', hue=sector_prices(df)[:2], show=False)


def test_nested_hue_with_unequal_lengths_raises():
    df = market_frame()
    bad = sector_prices(df)
    bad[1] = bad[1][:-5]
    with pytest.raises(ValueError, match='length'):
        hyp.plot(df, '-', hue=bad, show=False)


def test_flat_hue_of_wrong_length_raises():
    df = market_frame()
    with pytest.raises(ValueError, match='hue'):
        hyp.plot(df, '-', hue=np.linspace(0, 1, 7), show=False)


def test_flat_hue_of_total_drawn_length_is_rejected():
    """F12: 4 traces x 120 rows = 480 is NOT an accepted form."""
    df = market_frame()
    with pytest.raises(ValueError, match='120 row'):
        hyp.plot(df, '-', hue=np.linspace(0, 1, 480), show=False)


def test_categorical_hue_still_defers_to_the_grouping():
    df = market_frame()
    labels = np.array(['a', 'b'] * (len(df) // 2))
    with pytest.warns(UserWarning, match='hue'):
        hyp.plot(df, '-', hue=labels, show=False)


def test_row_hierarchy_hue_is_still_warned_and_ignored():
    """Row plotting semantics are unchanged in 1.1 (Global Constraints)."""
    idx = pd.MultiIndex.from_tuples(
        [('c1', s) for s in range(3)] + [('c2', s) for s in range(3)],
        names=['cond', 'subj'])
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 4)), index=idx)
    with pytest.warns(UserWarning, match='ignoring hue'):
        fig = hyp.plot(df, '-', hue=list(range(6)), show=False)
    assert len(_ax(fig).lines) == 8


def test_no_hue_keeps_the_documented_group_colour():
    fig = hyp.plot(market_frame(), '-', show=False)
    colours = {tuple(np.round(matplotlib.colors.to_rgb(ln.get_color()), 4))
               for ln in _ax(fig).lines}
    assert len(colours) == 1


def test_colorbar_renders_for_a_continuous_hue_over_a_hierarchy():
    """A second axes alone does NOT prove this works.

    Measured before the implementation landed: `colorbar=True` produced 2
    axes even while the hue was being warned about and dropped. So the
    scale itself is asserted -- it must span the concatenation of every
    trace's aux, leaves AND means, which is the one shared scale
    `_multicolor_line_colors` maps over.
    """
    df = market_frame()
    hues = sector_prices(df)
    out = hyp.plot(df, '-', hue=hues, colorbar=True, return_model=True,
                   show=False)
    fig = out['fig']
    assert len(fig.axes) == 2
    every = np.concatenate([np.asarray(a, dtype=float)
                            for a in out['trace_metadata']['aux']])
    bar = [a for a in fig.axes if not hasattr(a, 'zaxis')][0]
    assert bar.get_ylim() == pytest.approx((every.min(), every.max()))


def test_price_hue_maps_monotonically_through_the_palette():
    """Not merely 'two colours differ': a monotone hue under a sequential
    palette must give monotone luminance along the trace."""
    df = market_frame()
    hues = [np.linspace(100.0, 200.0, len(df)) for _ in range(3)]
    fig = hyp.plot(df, '-', hue=hues, palette='viridis', show=False)
    rgb = np.asarray(_collections(_ax(fig))[0].get_colors())[:, :3]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(lum) > -1e-6), 'viridis luminance must rise with hue'
    assert lum[-1] > lum[0] + 0.2


def test_hue_and_data_are_co_truncated():
    """Contract 6: one truncation operation, applied to both.

    `fig.canvas.draw()` first: a `Line3DCollection` keeps its 3-D segments
    privately and `get_segments()` returns the PROJECTED 2-D ones, which is
    an empty list until a draw has projected them (measured).
    """
    df = market_frame()
    hues = sector_prices(df)
    fig = hyp.plot(df, '-', hue=hues, show=False)
    fig.canvas.draw()
    colls = _collections(_ax(fig))
    assert len(colls) == 4
    for coll in colls:
        segs = coll.get_segments()
        assert len(segs) > 0
        assert len(np.asarray(coll.get_colors())) == len(segs)


# --- the adversarial matrix owed against this task ---------------------------
# Raised in review as cases Tasks 1-5 established but Task 6 could regress:
# NA hierarchy labels combined with hue, duplicate innermost feature names
# reaching the drawing code, and the aux co-truncation rule under
# unequal-length members.


@pytest.mark.parametrize('missing', [np.nan, None, pd.NA])
def test_na_group_labels_still_group_once_under_a_continuous_hue(missing):
    """The NA-aware grouping key must survive the hue path.

    Without it every NA-labelled leaf becomes its own group, so the trace
    count changes -- and under a hue that shows up as extra collections
    rather than as an obviously wrong picture.
    """
    measures = ('return', 'volatility', 'momentum')
    tuples = [('Market', missing, m) for m in measures]
    tuples += [('Market', 'Tech', m) for m in measures]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(40, 6)).cumsum(axis=0), columns=cols)
    fig = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)), show=False)
    # 2 leaves + 1 'Market' mean, NOT 3 leaves from a split NA group
    assert len(_collections(_ax(fig))) == 3


def test_duplicate_innermost_feature_names_survive_the_hue_path():
    """Duplicate feature labels are matched by (label, occurrence) (v8).

    They reach `plot()` as arrays, so a continuous hue must not care -- but
    it is exactly the sort of frame a re-detection bug would trip over.
    """
    tuples = [('Rig', well, sensor)
              for well in ('W1', 'W2')
              for sensor in ('temp', 'temp', 'flow')]
    cols = pd.MultiIndex.from_tuples(tuples, names=['Rig', 'Well', 'Sensor'])
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(30, 6)).cumsum(axis=0), columns=cols)
    out = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)),
                   return_model=True, show=False)
    assert len(_collections(_ax(out['fig']))) == 3
    assert [len(a) for a in out['trace_metadata']['aux']] == [30, 30, 30]


def test_aux_is_truncated_with_its_data_when_members_are_unequal():
    """Contract 6 at the unit level: ONE truncation, applied to both.

    A column hierarchy cannot produce unequal-length members (every group
    is a column slice of one frame), so this is asserted directly on
    `build_hierarchy_traces` rather than through a `plot()` call that
    cannot reach the state.
    """
    from hypertools.plot.hierarchy import build_hierarchy_traces

    meta = {'leaf_keys': [('G', 'a'), ('G', 'b')], 'n_levels': 2,
            'level_names': ['G', 'leaf'], 'axis': 'columns'}
    leaves = [np.arange(10, dtype=float).reshape(10, 1),
              np.arange(6, dtype=float).reshape(6, 1)]
    aux = [np.arange(10, dtype=float), np.full(6, 4.0)]
    with pytest.warns(UserWarning, match='unequal-length'):
        ft = build_hierarchy_traces(leaves, meta, aux=aux)
    assert [len(a) for a in ft.arrays] == [10, 6, 6]
    assert [len(a) for a in ft.aux] == [10, 6, 6]
    # the mean aux is the mean of the SAME 6-row slices the data used
    assert np.allclose(ft.aux[-1], (np.arange(6.0) + 4.0) / 2.0)
