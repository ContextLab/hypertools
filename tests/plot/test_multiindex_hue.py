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
import matplotlib.colors as mcolors
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


def test_ANIMATED_continuous_hue_collections_carry_the_trace_tag():
    """`_hyp_trace_index` on the animated path, not just the static one.

    `_apply_multicolor_lines` (static) tags every collection it creates;
    `_apply_multicolor_animation` did not, so an animated continuous-hue
    figure carried ZERO tagged artists and there was no supported way to
    tell its data collections from the 3-D bounding cube -- which is SIX
    `Line3DCollection` wireframe faces sitting in the same `ax.collections`
    list. Measured while auditing a gallery figure: an extent computed over
    `ax.collections` reported the CUBE's extent (a constant 1.00 fill)
    rather than the data's, and nothing in the public surface said so.

    Head and trail collections both belong to a trace, so both are tagged
    and `_hyp_trace_role` says which is which -- the same shape as
    `_hyp_forecast_role` on forecast artists.
    """
    df = market_frame()                       # 3 sector leaves + 1 mean
    hues = [np.linspace(k, k + 1.0, len(df)) for k in (0.0, 5.0, 9.0)]
    anim = hyp.plot(df, '-', hue=hues, palette='viridis', animate=True,
                    chemtrails=True, duration=2, frame_rate=4, show=False)
    ax = anim.figure.axes[0]
    tagged = [c for c in ax.collections
              if getattr(c, '_hyp_trace_index', None) is not None]
    heads = [c for c in tagged if c._hyp_trace_role == 'head']
    trails = [c for c in tagged if c._hyp_trace_role == 'trail']

    assert sorted(c._hyp_trace_index for c in heads) == [0, 1, 2, 3], (
        'one tagged HEAD collection per final trace (3 leaves + 1 mean)')
    assert sorted(c._hyp_trace_index for c in trails) == [0, 1, 2, 3], (
        'chemtrails=True draws one trail per trace, and a trail belongs to '
        'the same trace its head does')
    # ...and the cube is still NOT tagged, which is the whole reason the tag
    # exists. Measured: the six wireframe faces are added by the FIRST frame
    # draw, not at construction (8 collections before, 14 after), so the
    # discriminating half of this test needs a driven frame -- checking it
    # on the freshly-built figure would assert against a cube that does not
    # exist yet and pass for the wrong reason.
    anim.draw_frame(0)
    untagged = [c for c in ax.collections
                if getattr(c, '_hyp_trace_index', None) is None]
    assert len(untagged) == 6, (
        f'expected the six untagged bounding-cube faces, got {len(untagged)}')
    assert len([c for c in ax.collections
                if getattr(c, '_hyp_trace_index', None) is not None]) == 8


def test_a_MATRIX_hue_colours_each_leaf_by_ONE_blend_not_a_rainbow():
    """A 2-D per-leaf hue is mixture weights, one row per observation.

    It used to pass the per-leaf length check -- an (n_rows, k) matrix has
    n_rows as its FIRST dimension -- and then get `ravel()`led, which
    reinterpreted each row's k weights as k consecutive CONTINUOUS values.
    Measured on a 6-leaf hierarchy with 4-column weights: every trace spanned
    ~220 degrees of hue (one spanned 360) where each should have held a
    single hue. Nothing warned; the figure was simply wrong.

    Kept as a matrix, a one-hot leaf holds EXACTLY one hue and varies only
    in lightness -- which is what lets the palette carry identity in the hue
    and a second quantity in the weight on a dark/light entry.
    """
    df = market_frame()                       # 3 leaves + 1 mean
    n = len(df)
    # leaf 0 pure red, leaf 1 pure yellow, leaf 2 pure blue
    weights = [np.zeros((n, 3)) for _ in range(3)]
    for leaf in range(3):
        weights[leaf][:, leaf] = 1.0
    fig = hyp.plot(df, '-', hue=weights,
                   palette=['#d92b2b', '#e8c72a', '#2f5fd0'], show=False)
    colls = [c for c in fig.axes[0].collections
             if getattr(c, '_hyp_trace_index', None) is not None]
    assert len(colls) == 4, '3 leaves + 1 derived mean'

    spreads = {}
    for coll in colls:
        rgb = np.clip(np.asarray(coll.get_colors())[:, :3], 0, 1)
        hues = mcolors.rgb_to_hsv(rgb)[:, 0] * 360.0
        spreads[coll._hyp_trace_index] = float(hues.max() - hues.min())
    for leaf in range(3):
        assert spreads[leaf] < 1.0, (
            f'leaf {leaf} spans {spreads[leaf]:.1f} degrees of hue; a '
            'one-hot weight row must give it exactly one colour')


def test_a_MATRIX_hue_mean_trace_is_the_MEAN_OF_THE_WEIGHTS():
    """The reason matrix hue is worth supporting through a hierarchy.

    A derived mean takes the element-wise mean of its children's aux, and
    the mean of mixture weights IS a mixture weight -- so leaves given one
    primary each make their parent come out the blend of those primaries,
    with nothing computing it. Two leaves at pure red and pure yellow must
    give a mean that is neither: it sits BETWEEN them in hue.
    """
    df = market_frame()
    n = len(df)
    weights = [np.zeros((n, 3)) for _ in range(3)]
    weights[0][:, 0] = 1.0        # red
    weights[1][:, 1] = 1.0        # yellow
    weights[2][:, 1] = 1.0        # yellow
    fig = hyp.plot(df, '-', hue=weights,
                   palette=['#d92b2b', '#e8c72a', '#2f5fd0'], show=False)
    colls = {c._hyp_trace_index: c for c in fig.axes[0].collections
             if getattr(c, '_hyp_trace_index', None) is not None}

    def hue_of(index):
        rgb = np.clip(np.asarray(colls[index].get_colors())[:, :3], 0, 1)
        return float(np.median(mcolors.rgb_to_hsv(rgb)[:, 0]) * 360.0)

    red, yellow, mean_hue = hue_of(0), hue_of(1), hue_of(3)
    assert red < mean_hue < yellow, (
        f'the mean trace ({mean_hue:.1f} deg) must blend its leaves '
        f'({red:.1f} and {yellow:.1f} deg), not copy one of them')


def test_per_leaf_hue_matrices_must_agree_in_width():
    """They are blended through ONE shared palette, so a 3-column leaf and a
    4-column leaf cannot both be right."""
    df = market_frame()
    n = len(df)
    ragged = [np.ones((n, 3)), np.ones((n, 4)), np.ones((n, 3))]
    with pytest.raises(ValueError, match='same number of columns'):
        hyp.plot(df, '-', hue=ragged, palette='viridis', show=False)


# --------------------------------------------------------------------------
# The matrix-hue CONTRACT under a column hierarchy.
#
# A per-leaf 2-D hue is one row of MIXTURE WEIGHTS per observation, blended
# through the palette by `mat2colors`. Everything the contract has to answer
# is pinned below rather than described in prose only, because the same
# argument used to mean different things under a hierarchy than it did on a
# flat plot -- measured, not supposed.
# --------------------------------------------------------------------------

def _leaf_weights(n, k=3, leaves=3):
    """One-hot weights: leaf i is pure palette component i."""
    weights = []
    for leaf in range(leaves):
        row = np.zeros((n, k))
        row[:, leaf % k] = 1.0
        weights.append(row)
    return weights


def _drawn(fig):
    return {c._hyp_trace_index: np.clip(np.asarray(c.get_colors())[:, :3], 0, 1)
            for c in _collections(_ax(fig))}


def test_matrix_hue_WEIGHTS_ARE_NORMALIZED_so_magnitude_carries_nothing():
    """Rows are scaled to sum to 1, so only the RATIO between components is
    visible. This is why a second quantity (the Market example's return)
    needs its own palette entry rather than a smaller total weight: halving
    every weight in a row changes nothing at all.
    """
    df = market_frame()
    n = len(df)
    half = [w * 0.5 for w in _leaf_weights(n)]
    palette = ['#8a2be2', '#ff8c00', '#00ced1']
    full = _drawn(hyp.plot(df, '-', hue=_leaf_weights(n), palette=palette,
                           show=False))
    scaled = _drawn(hyp.plot(df, '-', hue=half, palette=palette, show=False))
    for index in full:
        assert np.allclose(full[index], scaled[index]), (
            f'trace {index} changed colour when every weight was halved; '
            'weights are documented as normalized')


def test_matrix_hue_needs_ONE_PALETTE_COLOUR_PER_COLUMN():
    """Wider than the palette is a real error, not a silent truncation: the
    caller has asked for a component that has no colour."""
    df = market_frame()
    n = len(df)
    with pytest.raises(ValueError, match='at least 3 colors'):
        hyp.plot(df, '-', hue=_leaf_weights(n, k=3),
                 palette=['#f00', '#0f0'], show=False)
    # a palette LONGER than the matrix is fine -- the extra colours are
    # simply unused components
    hyp.plot(df, '-', hue=_leaf_weights(n, k=3),
             palette=['#f00', '#0f0', '#00f', '#ff0'], show=False)


def test_a_NONFINITE_weight_greys_the_leaf_AND_EVERY_ANCESTOR_MEAN():
    """The blast radius of one bad weight, measured rather than assumed.

    A mean trace is the element-wise mean of its children's weights, so a
    single NaN propagates up every level of the hierarchy -- the leaf, its
    sector mean and the market mean all go neutral grey at that row, while
    sibling leaves are untouched. Worth pinning: a caller debugging one grey
    patch on three different traces should find this documented, not have to
    rediscover that mean(NaN) is NaN.
    """
    df = market_frame()
    n = len(df)
    weights = _leaf_weights(n)
    weights[0][n // 2, 1] = np.nan
    with pytest.warns(UserWarning, match='non-finite'):
        fig = hyp.plot(df, '-', hue=weights,
                       palette=['#8a2be2', '#ff8c00', '#00ced1'], show=False)
    drawn = _drawn(fig)
    neutral = np.array([0.75, 0.75, 0.75])

    def greyed(index):
        return int((np.abs(drawn[index] - neutral).max(axis=1) < 1e-6).sum())

    assert greyed(0) > 0, 'the leaf carrying the NaN was not greyed'
    assert greyed(1) == 0 and greyed(2) == 0, (
        'a sibling leaf was greyed by another leaf s NaN')
    ancestors = [i for i in drawn if i >= 3]
    assert ancestors, 'the hierarchy drew no mean traces'
    assert all(greyed(i) > 0 for i in ancestors), (
        'a NaN weight did not reach the derived means, which are computed '
        'as the mean of their children s weights')


def test_color_reduce_MEANS_THE_SAME_THING_with_and_without_a_hierarchy():
    """The inconsistency this pins: `color_reduce=` selected the literal-RGB
    route on a flat plot and was SILENTLY IGNORED under a column hierarchy,
    so one `hue=` array produced two different figures depending only on
    whether the frame had a hierarchy. Measured on a 3-column hue with a
    non-RGB palette: flat changed colour, hierarchy did not.
    """
    df = market_frame()
    n = len(df)
    palette = ['#8a2be2', '#ff8c00', '#00ced1']       # NOT red/green/blue:
    # with an r/g/b palette, "blend the weights" and "use the weights AS
    # rgb" coincide for one-hot rows, and the test would pass vacuously
    weights = _leaf_weights(n)

    blended = _drawn(hyp.plot(df, '-', hue=weights, palette=palette,
                              show=False))
    as_rgb = _drawn(hyp.plot(df, '-', hue=weights, palette=palette,
                             color_reduce='PCA', show=False))
    assert not np.allclose(blended[0], as_rgb[0]), (
        'color_reduce= had no effect under a hierarchy')

    flat = [df['Market'][s].to_numpy() for s in ('Tech', 'Financials', 'Energy')]
    flat_rgb = hyp.plot(flat, '-', hue=np.vstack(weights), palette=palette,
                        color_reduce='PCA', show=False)
    flat_colours = np.clip(
        np.asarray(_collections(_ax(flat_rgb))[0].get_colors())[:, :3], 0, 1)
    assert np.allclose(as_rgb[0][:3], flat_colours[:3], atol=1e-6), (
        'the same hue and color_reduce produced different colours with and '
        'without a hierarchy')


def test_a_WIDE_matrix_hue_takes_the_RGB_ROUTE_under_a_hierarchy_too():
    """More than 3 columns means literal RGB after reduction, on both paths.
    Under a hierarchy this used to stay on the palette-blend path, which
    also meant a >3-column hue silently required a >3-colour palette."""
    df = market_frame()
    n = len(df)
    rng = np.random.default_rng(3)
    # a REAL 5-component mixture: every component varies. One-hot leaves
    # would leave two columns constant, and sklearn rightly emits a divide
    # warning for a matrix with no variance in a column -- see the
    # degenerate case below, which is pinned separately rather than hidden.
    wide = [rng.random((n, 5)) for _ in range(3)]
    # no palette long enough to blend 5 components is supplied, and it is
    # not needed: the RGB route does not consult the palette at all
    fig = hyp.plot(df, '-', hue=wide, palette=['#f00', '#0f0', '#00f'],
                   show=False)
    assert len(_drawn(fig)) >= 4


def test_a_WIDE_hue_with_CONSTANT_COLUMNS_still_draws_usable_colours():
    """Degenerate input goes through the reducer, so sklearn warns that a
    column has no variance to explain. Measured: the warning is about
    `explained_variance_ratio`, which this path never reads -- the colours
    come out finite and distinct. Pinned rather than suppressed, because
    silencing a numerical warning from inside a reducer would also silence
    the cases where it means something.
    """
    df = market_frame()
    n = len(df)
    one_hot = [np.eye(5)[i] * np.ones((n, 1)) for i in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        fig = hyp.plot(df, '-', hue=one_hot, palette=['#f00', '#0f0', '#00f'],
                       show=False)
    drawn = _drawn(fig)
    stacked = np.vstack(list(drawn.values()))
    assert np.isfinite(stacked).all(), 'constant columns produced NaN colours'
    assert len(np.unique(np.round(stacked, 6), axis=0)) >= 4, (
        'every trace collapsed to the same colour')


@pytest.mark.parametrize('fraction', [0.0, 0.5, 1.0])
def test_ANIMATED_matrix_hue_holds_ONE_HUE_PER_LEAF_on_every_frame(fraction):
    """The animated path, which the static matrix tests do not reach.

    Colour for an animation is resolved per frame, so "each leaf holds a
    single hue" has to be true on the early, middle and final frame -- not
    only in the static figure. The failure this guards against is the one
    that started this work: a matrix hue that was ravelled into consecutive
    CONTINUOUS values drew every trace across ~220 degrees of hue.
    """
    df = market_frame(T=60)
    n = len(df)
    anim = hyp.plot(df, '-', hue=_leaf_weights(n),
                    palette=['#8a2be2', '#ff8c00', '#00ced1'],
                    animate=True, duration=2, frame_rate=8, show=False)
    frame = min(int(round(fraction * (anim.n_frames - 1))), anim.n_frames - 1)
    anim.draw_frame(frame)

    drawn = _drawn(anim.figure)
    assert len(drawn) >= 4, f'frame {frame} drew {len(drawn)} traces'
    for index, rgb in drawn.items():
        if not len(rgb):
            continue
        hues = mcolors.rgb_to_hsv(rgb)[:, 0] * 360.0
        spread = float(hues.max() - hues.min())
        assert spread < 1.0, (
            f'frame {frame}: trace {index} spans {spread:.1f} degrees of '
            'hue; a one-hot weight row must hold exactly one colour')


def test_ANIMATED_matrix_hue_keeps_the_MEAN_BETWEEN_ITS_CHILDREN():
    """The property that makes matrix hue worth having, checked while
    animating: leaves at pure component 0 and pure component 1 must give a
    parent that is neither, on a drawn frame rather than only statically."""
    df = market_frame(T=60)
    n = len(df)
    weights = [np.zeros((n, 3)) for _ in range(3)]
    weights[0][:, 0] = 1.0        # Tech   -> component 0
    weights[1][:, 1] = 1.0        # Fin    -> component 1
    weights[2][:, 1] = 1.0        # Energy -> component 1
    anim = hyp.plot(df, '-', hue=weights,
                    palette=['#d92b2b', '#e8c72a', '#2f5fd0'],
                    animate=True, duration=2, frame_rate=8, show=False)
    anim.draw_frame(anim.n_frames - 1)
    drawn = _drawn(anim.figure)

    def hue_of(index):
        return float(np.median(mcolors.rgb_to_hsv(drawn[index])[:, 0]) * 360.0)

    first, second, mean_hue = hue_of(0), hue_of(1), hue_of(3)
    assert first < mean_hue < second, (
        f'the mean trace ({mean_hue:.1f} deg) does not blend its leaves '
        f'({first:.1f} and {second:.1f} deg) on a drawn frame')
