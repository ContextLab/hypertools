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


@pytest.mark.parametrize('name', sorted(CATEGORICAL))
def test_a_categorical_hue_keeps_one_forecast_per_dataset(name):
    """However the categories fall, there is one forecast per DATASET.

    The count must not depend on how many runs the regrouping produced --
    that dependence was the bug.
    """
    data = _walks(2)
    fig = hyp.plot(data, '-', predict='Kalman', t=4, hue=CATEGORICAL[name],
                   show=False)
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


def test_the_bundle_never_claims_a_forecast_that_was_not_drawn():
    """`return_model=True` reported forecasts even when the figure showed
    none -- data saying the forecast exists beside a picture where it does
    not. Whatever the drawing decision is, the bundle must agree with it."""
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
        if drawn == 0:
            assert reported is None or not reported.get('forecasts'), (
                f'{kw}: bundle reports forecasts but none were drawn')
        else:
            assert reported is not None and len(reported['forecasts']) == drawn


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
    colls = [c for c in fig.axes[0].collections
             if hasattr(c, 'get_colors') and len(c.get_colors())]
    assert colls, 'expected the data to be drawn as colour collections'
    tail_colours = [np.asarray(c.get_colors())[-1][:3] for c in colls]
    for f in fc:
        rgb = np.asarray(matplotlib.colors.to_rgba(f.get_color())[:3])
        assert any(np.allclose(rgb, t, atol=0.02) for t in tail_colours), (
            f'forecast colour {rgb} matches no trace-end colour')
