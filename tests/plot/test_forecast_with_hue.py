# -*- coding: utf-8 -*-
"""`predict=` must survive `hue=` and `cluster=` regrouping.

Maintainer decision: "forecasts should be supported here."

Before this, `plot()` dropped every forecast whenever regrouping changed the
trace count -- `if len(raw_forecasts) != len(xform): raw_forecasts = None`.
That gate is pure cardinality, so whether a forecast survived depended on an
accident of how the categories happened to fall:

    plain                                 2 datasets -> 2 forecasts
    categorical hue, 1 run per dataset    2 traces   -> 2 forecasts (matched)
    categorical hue, alternating runs     8 traces   -> 0 forecasts
    4 categories                          4 traces   -> 0 forecasts
    cluster=                              6 traces   -> 0 forecasts

A forecast belongs to a DATASET and is anchored at that dataset's last
observation, so after regrouping it belongs to whichever run contains that
observation. `segment_by_run` already reports each run's source dataset, so
that run is `max(j for j where seg_dataset[j] == i)` -- and it is both the
trace the forecast continues and the trace whose style it should inherit.
"""

import matplotlib
matplotlib.use('Agg')

import contextlib
import warnings

import numpy as np
import pytest

import hypertools as hyp


def _walks(n=2, rows=30, dims=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _forecasts(fig):
    ax = fig.axes[0]
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) == 'static']


def _observed(fig):
    ax = fig.axes[0]
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None]


CATEGORICAL = {
    'one run per dataset': ['a'] * 30 + ['b'] * 30,
    'alternating runs': (['a'] * 5 + ['b'] * 5) * 6,
    'four categories': (['a'] * 15 + ['b'] * 15 + ['c'] * 15 + ['d'] * 15),
    'singleton runs': ['a', 'b'] * 30,
}

#: The `'singleton runs'` fixture is deliberately extreme -- alternating
#: every observation makes 60 runs of ONE point each -- so `plot()`
#: legitimately warns that a pure line format cannot render them. It is the
#: only fixture here that should.
_SINGLETON_NOTICE = 'only one observation'


@contextlib.contextmanager
def _categorical_warnings(name):
    """Assert exactly which fixtures provoke a warning, instead of letting
    the true ones scroll past as noise.

    CAPTURED, not filtered: a filter would equally hide the notice
    disappearing, and the singleton fixture exists to provoke it. Anything
    else that warns fails the test rather than hiding behind the expected
    two -- which is the whole reason to clean this up.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield
    singleton = [w for w in caught if _SINGLETON_NOTICE in str(w.message)]
    other = [f'{w.category.__name__}: {w.message}'
             for w in caught if _SINGLETON_NOTICE not in str(w.message)]
    assert not other, f'{name}: unexpected warning(s): {other}'
    if name == 'singleton runs':
        assert singleton, (
            f'{name}: 60 one-observation runs drawn with a pure line format '
            f'no longer warn that they will be invisible')
    else:
        assert not singleton, (
            f'{name}: warned about single-observation runs, but this '
            f'fixture has none: {[str(w.message) for w in singleton]}')


@pytest.mark.parametrize('name', sorted(CATEGORICAL))
def test_a_categorical_hue_keeps_one_forecast_per_dataset(name):
    """However the categories fall, there is one forecast per DATASET.

    The count must not depend on how many runs the regrouping produced --
    that dependence was the bug.
    """
    data = _walks(2)
    with _categorical_warnings(name):
        fig = hyp.plot(data, '-', predict='Kalman', t=4,
                       hue=CATEGORICAL[name], show=False)
    assert len(_forecasts(fig)) == len(data), (
        f'{name}: {len(_forecasts(fig))} forecasts for {len(data)} datasets, '
        f'with {len(_observed(fig))} observed traces drawn')


def test_a_forecast_inherits_the_style_of_the_run_it_continues():
    """Not of "dataset i" -- of the run holding that dataset's LAST
    observation, which is the trace the forecast visually continues. Under an
    alternating hue those are different colours, so this discriminates."""
    data = _walks(2)
    hue = (['a'] * 5 + ['b'] * 5) * 6
    fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=hue, show=False)
    obs, fc = _observed(fig), _forecasts(fig)
    assert len(fc) == 2

    # each dataset's final observation falls in a 'b' run (rows 25-29 of 30),
    # so both forecasts must carry the 'b' colour, not the 'a' colour
    for f in fc:
        # the colour must be one actually used by an observed run
        assert any(np.allclose(matplotlib.colors.to_rgba(f.get_color())[:3],
                               matplotlib.colors.to_rgba(o.get_color())[:3])
                   for o in obs), 'forecast colour matches no drawn trace'


def test_the_forecast_starts_where_its_dataset_ends():
    """Geometry, not just count: a forecast that survived regrouping but got
    anchored on the wrong dataset would still pass a count check."""
    data = _walks(2)
    hue = (['a'] * 5 + ['b'] * 5) * 6
    fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=hue, show=False)
    obs, fc = _observed(fig), _forecasts(fig)

    # guard first: `for f in []` runs zero times, so without this the whole
    # check below passes when NO forecast was drawn -- which is precisely
    # the state this test exists to detect
    assert len(fc) == len(data), (
        f'expected {len(data)} forecasts to check the geometry of; got '
        f'{len(fc)}')
    ends = [np.array(o.get_data_3d())[:, -1] for o in obs]
    for f in fc:
        head = np.array(f.get_data_3d())[:, 0]
        assert any(np.allclose(head, e, atol=1e-6) for e in ends), (
            'a forecast does not start at the end of any drawn trace')


def test_cluster_keeps_its_forecasts_too():
    data = _walks(2)
    fig = hyp.plot(data, '-', predict='Kalman', t=4, cluster='KMeans',
                   n_clusters=3, show=False)
    assert len(_forecasts(fig)) == len(data)


def test_the_bundle_says_whether_its_forecasts_reached_the_figure():
    """`return_model=True` hands back MODEL output, so a successful fit is
    reported whether or not the figure could render it -- discarding it to
    make the bundle mirror the picture would lose the result to a drawing
    limitation. What the bundle must never do is leave the two
    indistinguishable, so `drawn` states which happened.

    An earlier version asserted the opposite (bundle empty when nothing is
    drawn) and passed -- because every case it tried DOES draw, so its
    not-drawn branch never ran. The animated case below is here to run it."""
    data = _walks(2)
    for kw in ({}, dict(hue=(['a'] * 5 + ['b'] * 5) * 6),
               dict(cluster='KMeans', n_clusters=3),
               dict(hue=np.arange(60))):
        bundle = hyp.plot(data, '-', predict='Kalman', t=4, show=False,
                          return_model=True, **kw)
        fig = bundle['fig']
        drawn = len([ln for ln in fig.axes[0].lines
                     if getattr(ln, '_hyp_forecast_role', None) == 'static'])
        reported = bundle['predict']
        assert len(reported['forecasts']) == len(data), (
            f'{kw}: the fit succeeded, so the bundle must report it')
        assert reported['drawn'] is (drawn > 0), (
            f"{kw}: bundle says drawn={reported['drawn']} but the figure has "
            f"{drawn} forecast artist(s)")
        assert (reported['draw_reason'] is None) is reported['drawn'], (
            f'{kw}: a refusal must come with a reason, and a drawn forecast '
            f'without one')

    # The not-drawn branch, actually exercised. It used to be an ANIMATED
    # line plot under `hue=`, which now DRAWS its forecasts (the regrouped
    # reveal gave the schedule the per-dataset mapping it lacked). What is
    # still refused is MARKER-only categorical regrouping: `reshape_data`
    # groups globally by category, so 3 datasets under 2 categories become 2
    # traces that are not datasets and have no per-dataset trace to anchor
    # to. Named rather than blanket-ignored -- an unexpected second warning
    # here must fail, not vanish.
    marker_data = _walks(3, rows=20)
    with pytest.warns(UserWarning, match='no per-dataset trace'):
        refused = hyp.plot(marker_data, 'o', predict='Kalman', t=4,
                           animate=True, duration=2, frame_rate=4,
                           show=False, return_model=True,
                           hue=(['a'] * 10 + ['b'] * 10) * 3)
    assert refused['predict']['drawn'] is False
    assert len(refused['predict']['forecasts']) == len(marker_data)

    # ...and the animated LINE case it used to use now reports drawn=True
    animated = hyp.plot(data, '-', predict='Kalman', t=4, animate=True,
                        duration=2, frame_rate=4, show=False,
                        return_model=True,
                        hue=(['a'] * 5 + ['b'] * 5) * 6)
    assert animated['predict']['drawn'] is True
    assert animated['predict']['draw_reason'] is None


def test_a_continuous_hue_does_not_silently_lose_its_forecast():
    """A continuous `hue=` draws the data as a LineCollection, so there is no
    `Line2D` to inherit a colour from -- a different failure from the
    cardinality gate, and the reason a single fix does not cover both.

    It was silently lost for a third reason, distinct from both of the
    above: `_apply_multicolor_lines` swaps the data lines for a
    `LineCollection` and cleared **every** line on the axes to do it --
    including the forecast overlays drawn moments earlier. The artists were
    created and then destroyed, which is why no cardinality fix could have
    reached this case.
    """
    data = _walks(2)
    with warnings.catch_warnings():
        warnings.simplefilter('error')      # nothing may be lost quietly
        fig = hyp.plot(data, '-', predict='Kalman', t=4,
                       hue=np.arange(60), show=False)
    assert len(_forecasts(fig)) == len(data), (
        f'{len(_forecasts(fig))} forecasts survived a continuous hue; '
        f'expected {len(data)}')


def test_a_continuous_hue_colours_the_forecast_from_its_anchor():
    """Under a continuous hue the observed trace has MANY colours, so "the
    same colour as its trace" resolves to the colour where the forecast
    begins -- the last segment of the trace it continues."""
    data = _walks(2)
    fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=np.arange(60),
                   show=False)
    fc = _forecasts(fig)
    assert len(fc) == 2
    # `> 1` colours: a 3-D axes carries several SINGLE-colour
    # `Line3DCollection`s of its own (panes and grid lines, all black), and
    # counting those as data traces let a black forecast pass this check.
    colls = [c for c in fig.axes[0].collections
             if hasattr(c, 'get_colors') and len(c.get_colors()) > 1]
    assert colls, 'expected the data to be drawn as colour collections'
    tail_colours = [np.asarray(c.get_colors())[-1][:3] for c in colls]
    for f in fc:
        rgb = np.asarray(matplotlib.colors.to_rgba(f.get_color())[:3])
        assert any(np.allclose(rgb, t, atol=0.02) for t in tail_colours), (
            f'forecast colour {rgb} matches no trace-end colour')


def test_every_matplotlib_forecast_artist_names_its_dataset():
    """`_hyp_forecast_role` says WHAT an artist is; it does not say WHICH
    series it belongs to. plotly has carried `meta['hyp_dataset']` from the
    start -- matplotlib had only list position, so anything that reorders or
    filters forecasts (`forecast_cluster=`, a per-dataset refusal) would
    silently re-pair them with the wrong data."""
    data = _walks(3)
    for kw in ({}, dict(hue=(['a'] * 5 + ['b'] * 5) * 9),
               dict(cluster='KMeans', n_clusters=3),
               dict(hue=np.arange(90))):
        fig = hyp.plot(data, '-', predict='Kalman', t=4, show=False, **kw)
        tags = sorted(getattr(f, '_hyp_forecast_dataset', None)
                      for f in _forecasts(fig))
        assert tags == list(range(len(data))), (
            f'{kw}: forecast artists carry dataset tags {tags}')


def test_a_continuous_hue_colours_each_forecast_from_ITS_OWN_dataset():
    """`_apply_multicolor_lines` paired retained forecasts with `line_colors`
    by list POSITION, which is only right while both stay one-per-dataset in
    the same order. Pair by the dataset tag instead, so the pairing survives
    anything that reorders forecasts.

    Two datasets given deliberately disjoint hue ranges: their tail colours
    are far apart, so a swapped pairing fails rather than landing within
    tolerance of the other trace's colour."""
    data = _walks(2, rows=30)
    hue = np.concatenate([np.linspace(0, 1, 30), np.linspace(9, 10, 30)])
    fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=hue, show=False)
    fc = _forecasts(fig)
    assert len(fc) == 2
    colls = [c for c in fig.axes[0].collections
             if hasattr(c, 'get_colors') and len(c.get_colors()) > 1]
    assert len(colls) == 2, 'expected one colour collection per dataset'
    tails = [np.asarray(c.get_colors())[-1][:3] for c in colls]
    assert not np.allclose(tails[0], tails[1], atol=0.05), (
        'the two datasets must end in visibly different colours for this '
        'test to be able to detect a swap')
    for f in fc:
        ds = getattr(f, '_hyp_forecast_dataset', None)
        rgb = np.asarray(matplotlib.colors.to_rgba(f.get_color())[:3])
        assert np.allclose(rgb, tails[ds], atol=0.02), (
            f'forecast for dataset {ds} is coloured {rgb}, but that '
            f"dataset's trace ends at {tails[ds]}")


# --------------------------------------------------------------------------
# plotly parity.
#
# The matplotlib fix let forecasts reach the DRAWING layer under regrouping
# for the first time -- and plotly's forecast block was written when that
# could never happen, so it still indexes `forecasts[i]` while looping over
# the drawn RUNS. With 2 datasets in 8 runs that is a hard IndexError, i.e.
# fixing one backend broke the other. Every matplotlib assertion above needs
# its plotly twin, or the two drift again.
# --------------------------------------------------------------------------

def _ply_forecasts(fig):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == 'static']


def _ply_observed(fig):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') is None
            and tr.mode is not None and 'lines' in str(tr.mode)]


@pytest.mark.parametrize('name', sorted(CATEGORICAL))
def test_plotly_also_keeps_one_forecast_per_dataset(name):
    data = _walks(2)
    with hyp.set_interactive_backend('plotly'), _categorical_warnings(name):
        fig = hyp.plot(data, '-', predict='Kalman', t=4,
                       hue=CATEGORICAL[name], show=False)
    assert len(_ply_forecasts(fig)) == len(data), (
        f'{name}: {len(_ply_forecasts(fig))} plotly forecasts for '
        f'{len(data)} datasets')


def test_plotly_tags_each_forecast_with_the_dataset_it_belongs_to():
    """`meta['hyp_dataset']` is the plotly half of the artist tag, and under
    regrouping it must still name the DATASET -- not the loop index over
    runs, which is what it recorded when the loop ran over runs."""
    data = _walks(3)
    hue = (['a'] * 5 + ['b'] * 5) * 9      # one value per observation (3x30)
    with hyp.set_interactive_backend('plotly'):
        fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=hue, show=False)
    tags = sorted((tr.meta or {})['hyp_dataset'] for tr in _ply_forecasts(fig))
    assert tags == list(range(len(data))), (
        f'forecast dataset tags {tags}; expected one per dataset')


def test_plotly_forecast_inherits_the_colour_of_the_run_it_continues():
    """The style must come from the run holding the dataset's LAST
    observation. Indexing the run list by forecast number picks run 0 and
    run 1 -- which under an alternating hue are the WRONG (and differently
    coloured) runs, so this discriminates."""
    data = _walks(2)
    hue = (['a'] * 5 + ['b'] * 5) * 6
    with hyp.set_interactive_backend('plotly'):
        fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=hue, show=False)
    obs = _ply_observed(fig)
    fc = _ply_forecasts(fig)
    assert len(fc) == 2
    # each dataset's last observation falls in the FINAL run carrying it;
    # the forecast must start exactly where that run ends
    for tr in fc:
        head = np.array([tr.x[0], tr.y[0], tr.z[0]], dtype=float)
        assert any(np.allclose(
            head, [o.x[-1], o.y[-1], o.z[-1]], atol=1e-6) for o in obs), (
            'a plotly forecast does not start at the end of any drawn run')


def test_plotly_and_matplotlib_draw_the_same_forecast_geometry_under_hue():
    """Same anchors, same forecast, whichever backend renders it."""
    data = _walks(2)
    hue = (['a'] * 5 + ['b'] * 5) * 6
    mpl_fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=hue,
                       antialias=False, show=False)
    with hyp.set_interactive_backend('plotly'):
        ply_fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=hue,
                           antialias=False, show=False)
    mpl_heads = sorted(tuple(np.round(np.array(f.get_data_3d())[:, 0], 9))
                       for f in _forecasts(mpl_fig))
    ply_heads = sorted(tuple(np.round([tr.x[0], tr.y[0], tr.z[0]], 9))
                       for tr in _ply_forecasts(ply_fig))
    assert len(mpl_heads) == len(ply_heads) == len(data)
    for m, p in zip(mpl_heads, ply_heads):
        assert np.allclose(m, p, atol=1e-9), (
            f'backends anchor the forecast differently: {m} vs {p}')
