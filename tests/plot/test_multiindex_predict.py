#!/usr/bin/env python
"""One forecast per pre-center/pre-scale plotted trajectory, and a bundle
that proves it.

Contract 5: forecasts[i] == hyp.predict(trace_data[i], model, t) for every
i. `plot()`'s forecast/trace correspondence guard used to null
`raw_forecasts` on any count mismatch, so a missing forecast was invisible;
for a hierarchy `FinalTraces.assert_consistent` now raises instead.

Contract 10: EVERY final trace of EVERY hierarchy -- leaf or derived mean,
row axis or column axis -- needs >= 2 rows. All three sides are tested: a
repeating innermost level forecasts; a unique-per-row one raises a message
about the row-expansion rule; a T=1 column frame raises a message about the
INPUT, since column grouping never shortens a trace and flattening cannot
lengthen one. The precondition also runs before the animation schedule is
built, so an animated one-row hierarchy raises rather than silently drawing
no forecast at every frame.

PLAN DEFECT (2026-07-28-hypertools-1.1-multiindex.md Task 8 Step 1). The
prescribed `_solid`/`_dashed` helpers split `ax.lines` on linestyle. That
cannot work: `_forecast_style_from` (plot.py:222-234) makes a forecast
INHERIT its source line's linestyle, so under this module's `fmt='-'` every
artist is solid and `_dashed(ax)` returns []. The helpers below use the
repo's own idiom instead (tests/plot/test_predict_integration.py:20-31):
forecast artists identify THEMSELVES through `_hyp_forecast_role`.
"""
import itertools
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def market_frame(T=150, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Financials', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0) + 100.0,
                        columns=cols)


def multirow_row_frame(n_time=10, seed=0):
    """A ROW hierarchy whose innermost level REPEATS, so each leaf keeps
    `n_time` rows: 2 conds x 3 subjs x n_time timepoints.

    Verified construction: `expand_multiindex` yields 6 leaves of shape
    (n_time, 3), and the plot draws 8 traces (6 leaves + 2 top-level means),
    every one of them n_time rows -- so every trace clears the >= 2-row
    precondition."""
    rng = np.random.default_rng(seed)
    tuples, rows = [], []
    for ci, cond in enumerate(['cond1', 'cond2']):
        for si in range(3):
            rows.append(rng.standard_normal((n_time, 3)).cumsum(axis=0)
                        + ci * 5.0)
            tuples.extend([(cond, f'S{si}')] * n_time)
    idx = pd.MultiIndex.from_tuples(tuples, names=['cond', 'subj'])
    return pd.DataFrame(np.vstack(rows), index=idx, columns=['x', 'y', 'z'])


def one_row_row_frame(seed=0):
    """A ROW hierarchy whose innermost level is UNIQUE PER ROW, so every
    leaf is 1 row. Verified: 6 leaves of shape (1, 4), 8 traces, all 1 row."""
    idx = pd.MultiIndex.from_tuples(
        [('cond1', s) for s in range(3)] + [('cond2', s) for s in range(3)],
        names=['cond', 'subj'])
    return pd.DataFrame(np.random.default_rng(seed).normal(size=(6, 4)),
                        index=idx)


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _observed(ax):
    """The DATA lines: everything that is not a forecast overlay.

    See the PLAN DEFECT note in the module docstring for why this is not a
    linestyle test."""
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None]


def _forecasts(ax):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is not None]


def _trace_collections(ax):
    """The per-segment collections a continuous hue draws, in trace order.

    `ax.collections` also holds the SIX `Line3DCollection` wireframe faces of
    the 3-D bounding cube (`matplotlib_backend._draw_cube`), which carry no
    `_hyp_trace_index`; the plan's version of this test zipped those against
    the forecasts."""
    tagged = [c for c in ax.collections
              if getattr(c, '_hyp_trace_index', None) is not None]
    return sorted(tagged, key=lambda c: c._hyp_trace_index)


def test_every_plotted_trajectory_gets_its_own_forecast():
    fig = hyp.plot(market_frame(), '-', predict='Kalman', t=1, show=False)
    ax = _ax(fig)
    assert len(_observed(ax)) == 4 and len(_forecasts(ax)) == 4


def test_bundle_forecasts_correspond_to_trace_data():
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                   return_model=True, show=False)
    assert len(out['trace_data']) == 4
    assert len(out['predict']['forecasts']) == 4
    assert len(_forecasts(_ax(out['fig']))) == 4


def test_each_bundled_forecast_equals_hyp_predict_on_its_trace():
    """Contract 5, asserted numerically for EVERY trace including the mean."""
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=2,
                   return_model=True, show=False)
    assert len(out['trace_data']) == len(out['predict']['forecasts']) == 4
    for trace, forecast in zip(out['trace_data'], out['predict']['forecasts']):
        direct = np.asarray(hyp.predict(np.asarray(trace), model='Kalman',
                                        t=2), dtype=float)
        assert np.allclose(np.asarray(forecast, dtype=float), direct,
                           rtol=1e-6, atol=1e-6)


def test_leaf_forecasts_match_hyp_predict_on_xform_data_when_spaces_coincide():
    """The v1.0 promise (plot.py's `predict=` note) still holds for the
    leaves -- but ONLY where the analysed space and the plotted space are the
    same.

    Contract 5 makes that conditional: a `reduce=` spec pinning more than
    three components leaves `xform_data` in the higher-dimensional space
    while `trace_data` is projected for display, and then this comparison is
    meaningless. The guard below is the condition, asserted rather than
    assumed; `tests/plot/test_hierarchy_bundle.py` covers the diverging
    case, where forecasts follow `trace_data`.
    """
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=2,
                   return_model=True, show=False)
    leaves = out['xform_data']
    # counts asserted explicitly: `zip` alone truncates at len(leaves) and so
    # could not tell a 3-trace figure from a 4-trace one
    assert len(leaves) == 3
    assert len(out['trace_data']) == 4
    assert len(out['predict']['forecasts']) == 4
    assert all(np.asarray(x).shape == np.asarray(tr).shape
               for x, tr in zip(leaves, out['trace_data'])), \
        'this assertion is only valid when the two spaces coincide'
    direct = hyp.predict([np.asarray(x) for x in leaves], model='Kalman', t=2)
    assert len(direct) == 3
    for got, want in zip(out['predict']['forecasts'][:len(leaves)], direct):
        assert np.allclose(np.asarray(got, dtype=float),
                           np.asarray(want, dtype=float),
                           rtol=1e-6, atol=1e-6)


def test_forecasts_are_not_silently_dropped():
    fig = hyp.plot(market_frame(), '-', predict='Kalman', t=3, show=False)
    assert len(_forecasts(_ax(fig))) == 4, \
        'forecasts vanished instead of raising'


def test_mean_trace_forecast_comes_from_the_mean_trajectory():
    """A mean trace is forecast from its OWN averaged trajectory, proven by
    exact equality with `hyp.predict(mean_traj)` -- which pins precisely
    which trajectory the bundled forecast came from, so the contract is
    proven completely.

    Comparing against the average of the LEAF forecasts is deliberately NOT
    asserted: forecasting approximately commutes with averaging as the
    leaves co-move, so a correct implementation fails such an assertion on
    exactly the data this plan targets. Measured (Kalman, t=1, T=150, 3
    leaves, scale ~100, 5 seeds per rho), the deleted assertion -- that the
    bundled forecast is NOT close to the average of the leaf forecasts at
    rtol=1e-3, atol=1e-3 -- held 5/5 at rho=0.0 (mean max abs diff 0.557)
    and 5/5 at rho=0.5 (0.524), but only 3/5 at rho=0.8 (0.130) and 0/5 at
    rho=0.9 (0.028), 0.95 (0.007) and 0.99 (0.0003). Real market sectors
    co-move at roughly rho 0.7-0.9. Do not re-add it.
    """
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                   return_model=True, show=False)
    assert out['trace_metadata']['is_mean'][-1] is True
    mean_traj = np.asarray(out['trace_data'][-1])
    from_mean = np.asarray(hyp.predict(mean_traj, model='Kalman', t=1),
                           dtype=float)
    bundled = np.asarray(out['predict']['forecasts'][-1], dtype=float)
    assert np.allclose(bundled, from_mean, rtol=1e-6, atol=1e-6)


def test_forecasts_anchor_on_their_own_trace():
    fig = hyp.plot(market_frame(), '-', predict='Kalman', t=1, show=False)
    ax = _ax(fig)
    observed, forecasts = _observed(ax), _forecasts(ax)
    assert len(observed) == len(forecasts) == 4
    for line, fc in zip(observed, forecasts):
        drawn = np.array(line.get_data_3d())
        assert np.allclose(np.array(fc.get_data_3d())[:, 0], drawn[:, -1],
                           atol=1e-6)


def test_forecasts_stay_inside_the_axes_limits():
    fig = hyp.plot(market_frame(), '-', predict='Kalman', t=5, show=False)
    ax = _ax(fig)
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    forecasts = _forecasts(ax)
    assert len(forecasts) == 4
    for fc in forecasts:
        pts = np.array(fc.get_data_3d())
        assert (pts.min(axis=1) >= lims[:, 0] - 1e-6).all()
        assert (pts.max(axis=1) <= lims[:, 1] + 1e-6).all()


def test_predict_composes_with_a_continuous_hue():
    df = market_frame()
    fig = hyp.plot(df, '-', predict='Kalman', t=1,
                   hue=np.linspace(0, 1, len(df)), show=False)
    assert len(_forecasts(_ax(fig))) == 4


def test_forecast_takes_the_final_observed_hue_colour():
    """F14, matplotlib half. A forecast under a continuous hue continues in
    the colour its source trace ended on -- ITS OWN trace, not a neighbour's.

    The three leaf ramps END at three different hue values (0.0 / 0.25 /
    1.0), so the derived mean ends at their average (0.4167) and all FOUR
    final colours differ. That is what makes the pairing assertion below
    order-SENSITIVE. The plan's version gave every leaf the same
    `np.linspace(0, 1, len(df))`, which made all four traces end on
    [0.9744 0.9036 0.1302]; the loop then compared one colour with itself
    four times and every permutation of the pairing passed. Measured here,
    the closest two of the four end colours differ by 0.164 per channel --
    8x the 0.02 match tolerance -- so a swap cannot slip through. The
    `assert_pairwise_distinct` guard keeps it that way if the ramps are ever
    edited.

    Both sides are keyed by the artist's own DATASET tag rather than by list
    position, matching what `_apply_multicolor_lines` pairs on
    (plot.py: `_hyp_forecast_dataset`).
    """
    df = market_frame()
    T = len(df)
    fig = hyp.plot(df, '-', predict='Kalman', t=1, palette='viridis',
                   hue=[np.linspace(0.6, 0.0, T),
                        np.linspace(0.0, 0.25, T),
                        np.linspace(0.4, 1.0, T)],
                   show=False)
    ax = _ax(fig)
    colls = _trace_collections(ax)
    forecasts = sorted(_forecasts(ax),
                       key=lambda ln: ln._hyp_forecast_dataset)
    assert len(colls) == len(forecasts) == 4
    assert [c._hyp_trace_index for c in colls] == [0, 1, 2, 3]
    assert [ln._hyp_forecast_dataset for ln in forecasts] == [0, 1, 2, 3]

    last_colors = [np.asarray(coll.get_colors())[-1][:3] for coll in colls]
    for a, b in itertools.combinations(last_colors, 2):
        assert not np.allclose(a, b, atol=0.05), \
            'the four traces must END on different colours, or the pairing ' \
            'below is satisfied by any permutation'

    for last, fc in zip(last_colors, forecasts):
        assert np.allclose(matplotlib.colors.to_rgb(fc.get_color()), last,
                           atol=0.02)


@pytest.mark.parametrize('frame_of', [market_frame,
                                      lambda: multirow_row_frame(n_time=20)],
                         ids=['column-axis', 'row-axis'])
def test_predict_with_hierarchy_and_animation_via_on_frame(frame_of):
    """Assertions go through the PUBLIC on_frame hook (animation-core Task
    7), not through ani._args.

    COUNTS ARE NOT ENOUGH. The animated schedule re-forecasts from
    `analyze_histories`, which is per-trace; if that list is rotated by one
    the counts are all still right and every animated forecast simply
    detaches from its own trace. Measured with a one-step rotation inserted
    after the hierarchy branch's `_compute_forecasts(_ft.arrays)`, this
    module's other 16 tests all still passed while the forecast/trace gaps
    went 0.0 -> 0.64 / 0.51 / 0.26 / 0.35. So the anchoring is asserted on
    COORDINATES here, exactly as the static path does in
    `test_forecasts_anchor_on_their_own_trace`, and on both axes -- the row
    axis reaches this code through a different expansion rule.
    """
    df = frame_of()
    n_traces = 4 if df.columns.nlevels >= 2 else 8
    seen = []
    fig, ani = hyp.plot(df, '-', predict='Kalman', t=1,
                        animate=True, duration=2, frame_rate=4,
                        on_frame=seen.append, show=False)
    for f in range(8):
        ani._func(f, *ani._args)          # harness only; never asserted on
    assert len(seen) == 8
    assert len(seen[-1].datasets) == n_traces
    ax = _ax(fig)
    observed, forecasts = _observed(ax), _forecasts(ax)
    assert len(observed) == len(forecasts) == n_traces
    for line, fc in zip(observed, forecasts):
        drawn = np.array(line.get_data_3d())
        assert np.allclose(np.array(fc.get_data_3d())[:, 0], drawn[:, -1],
                           atol=1e-6), \
            'an animated forecast must start where ITS OWN trace ended'


def test_return_model_bundle_has_one_model_and_forecast_per_trace():
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                   return_model=True, show=False)
    assert len(out['trace_data']) == 4
    assert len(out['predict']['forecasts']) == 4
    assert out['predict']['params'] == {'t': 1}
    assert out['predict']['model'] == 'Kalman'
    assert len(out['trace_metadata']['keys']) == 4


# --- Contract 10: the >= 2-row precondition, on BOTH axes -------------------

def test_row_hierarchy_with_multi_row_leaves_forecasts_every_trace():
    """Row hierarchies DO forecast when the shape allows it.

    The construction is verified before the counts are asserted, so a
    silently-degenerate frame fails here rather than passing vacuously."""
    from hypertools.plot.multiindex import expand_multiindex

    df = multirow_row_frame(n_time=10)
    leaves, _ = expand_multiindex(df)
    assert len(leaves) == 6
    assert all(np.asarray(leaf).shape[0] == 10 for leaf in leaves), \
        'the frame must have MULTI-row leaves for this test to mean anything'

    out = hyp.plot(df, '-', predict='Kalman', t=2, return_model=True,
                   show=False)
    assert all(np.asarray(tr).shape[0] >= 2 for tr in out['trace_data'])
    assert len(out['trace_data']) == 8
    assert len(out['predict']['forecasts']) == 8
    ax = _ax(out['fig'])
    assert len(_observed(ax)) == 8 and len(_forecasts(ax)) == 8


def test_row_hierarchy_with_one_row_leaves_raises_naming_the_trace():
    """Contract 10's other side. The message must be about the DATA -- it
    names the trace and its row count and explains the leaf rule -- not a
    bubbled-up `predict` internal error, so it is raised as a precondition
    over `ft.arrays` before any forecasting."""
    from hypertools.plot.multiindex import expand_multiindex

    df = one_row_row_frame()
    leaves, _ = expand_multiindex(df)
    assert all(np.asarray(leaf).shape[0] == 1 for leaf in leaves)

    with pytest.raises(ValueError) as excinfo:
        hyp.plot(df, '-', predict='Kalman', t=1, show=False)
    message = str(excinfo.value)
    assert 'at least 2 rows per trace' in message
    assert '1 row' in message
    assert 'unique FULL index tuple' in message
    assert 'reset_index(drop=True)' in message
    assert 'COLUMN axis' in message
    assert 'cannot forecast from a single observation' not in message, \
        'the bubbled-up predict error must not be what the user sees'


def test_one_row_column_hierarchy_raises_about_the_input_not_the_grouping():
    """The precondition is NOT row-specific.

    A column hierarchy never shortens a trace -- every group keeps all
    `len(df)` rows -- but it cannot lengthen one either. Measured with this
    plan's grouping idiom: T=1 gives leaf shapes {'Tech': (1, 3),
    'Fin': (1, 3)} and a mean of (1, 3), NOT forecastable; T=2 gives
    (2, 3) throughout, forecastable. So the check must run on this axis
    too, and its message must be about the INPUT having one observation --
    flattening the hierarchy cannot add a row, so it is not offered.

    A 1-row frame also warns from `reduce` ('Cannot reduce a single
    observation (row) of data ...', reduce.py:455). That is orthogonal to
    what is asserted here, so it is tolerated rather than allowed to fail
    the test.
    """
    df = market_frame(T=1)
    assert len(df) == 1

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='Cannot reduce a single observation')
        with pytest.raises(ValueError) as excinfo:
            hyp.plot(df, '-', predict='Kalman', t=1, show=False)

    message = str(excinfo.value)
    assert 'at least 2 rows per trace' in message
    assert '1 row' in message
    assert 'one observation' in message
    assert 'reset_index(drop=True)' not in message, \
        'flattening a COLUMN hierarchy cannot add a row -- do not offer it'
    assert 'cannot forecast from a single observation' not in message, \
        'the bubbled-up predict error must not be what the user sees'


def test_animated_one_row_hierarchy_still_raises_the_precondition():
    """The precondition runs BEFORE the forecast schedule is built.

    Plan 3's `min_history` returns None for a frame whose revealed history
    is too short -- correct for the opening frames of a real animation. But
    a one-row hierarchy never reaches 2 rows at ANY frame, so deferring to
    that path would draw no forecast forever, silently. The full-trace
    precondition is a permanent property of the data, so it raises here
    exactly as it does for a static plot.
    """
    with pytest.raises(ValueError, match='at least 2 rows per trace'):
        hyp.plot(one_row_row_frame(), '-', predict='Kalman', t=1,
                 animate=True, duration=2, frame_rate=4, show=False)


@pytest.mark.parametrize(
    'frame_of, forbidden',
    [(lambda: market_frame(T=30), 'pass a frame with more rows'),
     (lambda: multirow_row_frame(n_time=10), 'reset_index(drop=True)')],
    ids=['column-axis', 'row-axis'])
def test_a_row_count_changing_stage_is_blamed_instead_of_the_grouping(
        frame_of, forbidden):
    """Contract 10's check runs on `_ft.arrays`, i.e. AFTER manip/normalize/
    reduce/align. So a short trace is not necessarily the grouping's or the
    input's doing -- `manip='Resample'` with `n_samples=1` shortens a trace
    the input had 30 (column) or 10 (row) rows for.

    Both axis-specific messages used to be emitted unconditionally, and both
    then asserted something FALSE about the user's data and offered a remedy
    that cannot work: the column text said "the input itself has only one
    observation" of a 30-row frame, and the row text blamed an
    innermost level that "is unique per row" when it repeats 10x. The repo's
    own idiom for this is the sibling hue-length check (plot.py, "the
    analysis pipeline changed the row count before plotting"), and that is
    what the trace-length check must say when the INPUT was long enough.
    """
    with pytest.raises(ValueError) as excinfo:
        hyp.plot(frame_of(), '-', predict='Kalman', t=2,
                 manip={'model': 'Resample', 'args': [],
                        'kwargs': {'n_samples': 1}},
                 show=False)
    message = str(excinfo.value)
    assert 'at least 2 rows per trace' in message
    assert 'analysis pipeline changed the row count' in message
    assert 'drop the row-count-changing stage' in message
    assert forbidden not in message, \
        'the pipeline shortened this trace -- the axis remedy cannot help'
    assert 'the input itself has only' not in message
    assert 'unique per row' not in message


def test_a_genuinely_short_input_still_blames_the_input_not_the_pipeline():
    """The other side of the branch: a row-count-PRESERVING manip stage on a
    T=1 column frame must still get the input-is-short message, so adding the
    pipeline branch did not simply relabel every failure."""
    df = market_frame(T=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='Cannot reduce a single observation')
        with pytest.raises(ValueError) as excinfo:
            hyp.plot(df, '-', predict='Kalman', t=1,
                     manip='ZScore', show=False)
    message = str(excinfo.value)
    assert 'the input itself has only one observation' in message
    assert 'analysis pipeline changed the row count' not in message


def test_one_row_row_hierarchy_still_plots_without_predict():
    """The precondition is scoped to `predict=`; plotting is untouched."""
    fig = hyp.plot(one_row_row_frame(), '-', show=False)
    assert len(_ax(fig).lines) == 8
