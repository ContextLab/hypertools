"""1.2.0 review, round 3: `predict=` in a `panels=` grid -- one shared
plotly legend, one warning per grid call (the forecast-cost notice with
the grid's total), and the single-axes error for a per-observation
`forecast_hue=`.

Real calls only; the figures and the warnings are read back as issued.
"""
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp

N_PANELS = 4
TRAIN = [np.random.default_rng(i).normal(size=(40, 3)).cumsum(0)
         for i in range(N_PANELS)]
TRUTH = [np.random.default_rng(10 + i).normal(size=(10, 3)).cumsum(0)
         + train[-1] for i, train in enumerate(TRAIN)]


@pytest.fixture(autouse=True)
def _close_figures():
    """Leave no figure behind."""
    yield
    plt.close('all')


def _caught(*args, **kwargs):
    """`hyp.plot(..., show=False)`: ``(result, [warning messages])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = hyp.plot(*args, show=False, **kwargs)
    return result, [str(w.message) for w in caught]


# --------------------------------------------------------------------------
# Q2: the plotly grid lists forecast and truth entries once
# --------------------------------------------------------------------------

@pytest.mark.parametrize('ndims', [2, 3])
def test_a_plotly_predict_grid_has_one_legend(ndims):
    """Each panel's `truth=` trace lists itself, in its dataset's color;
    four of them under one name were a "conflict" that sent the grid back
    to a legend beside every panel (12 entries). They are one neutral
    key, as in the matplotlib grid and the single-axes call."""
    train = [x[:, :ndims] for x in TRAIN]
    truth = [x[:, :ndims] for x in TRUTH]
    kwargs = dict(panels=(1, N_PANELS), predict='Kalman', t=10, truth=truth,
                  legend=True, reduce=None, ndims=ndims)
    fig, _ = _caught(train, backend='plotly', **kwargs)
    listed = [trace for trace in fig.data if trace.showlegend]
    assert sorted(((t.legendrank, t.name) for t in listed)) == [
        (1000, '1'), (1001, '2'), (1002, '3'), (1003, '4'),
        (2000, 'Kalman'), (2001, 'truth')]
    # one legend for the figure
    assert {t.legend for t in listed} == {'legend'}
    assert [k for k in fig.layout if k.startswith('legend')] == ['legend']
    # the truth key is neutral; every drawn truth keeps its dataset's color
    key, = [t for t in listed if t.name == 'truth']
    drawn = [t for t in fig.data
             if (t.meta or {}).get('hyp_forecast_role') == 'truth']
    data = [t for t in listed if t.name in '1234']
    assert len(drawn) == N_PANELS and not any(t.showlegend for t in drawn)
    assert [t.line.color for t in drawn] == [t.line.color for t in data]
    assert key.line.color == key.marker.color == 'rgba(85,85,85,1.0)'
    assert key.mode == drawn[0].mode
    assert all(t.legendgroup == key.legendgroup for t in drawn)

    # the matplotlib grid lists the same six
    mpl, _ = _caught(train, backend='matplotlib', **kwargs)
    legend, = mpl.legends
    assert [t.get_text() for t in legend.get_texts()] \
        == ['1', '2', '3', '4', 'Kalman', 'truth']


# --------------------------------------------------------------------------
# Q3: one warning per grid call
# --------------------------------------------------------------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('kwargs, message', [
    (dict(forecast_n_clusters=2), 'forecast_n_clusters= has no effect'),
    (dict(forecast_cluster='KMeans'),
     'forecast_cluster= needs at least two forecasts'),
], ids=['n_clusters', 'cluster'])
def test_a_forecast_argument_warning_is_issued_once_per_grid(
        backend, kwargs, message):
    _, messages = _caught(TRAIN, panels=(1, N_PANELS), predict='Kalman',
                          t=10, backend=backend, **kwargs)
    assert len([m for m in messages if message in m]) == 1, messages


def _fits(messages, pattern):
    """The fit count of the one forecast-cost warning in `messages`."""
    found = [re.search(pattern, m) for m in messages]
    found = [m for m in found if m]
    assert len(found) == 1, messages
    return int(found[0].group(1)), found[0].string


@pytest.mark.parametrize('predict, n_models', [
    ('Kalman', 1), (['Kalman', 'AutoRegressor'], 2)],
    ids=['one-model', 'two-models'])
def test_the_forecast_cost_warning_is_one_with_the_grids_total(
        predict, n_models):
    """A threshold of 0 s makes every schedule's projection warn. Each
    dataset alone warns with its own fit count; the grid of the four
    warns ONCE, with their sum."""
    anim = dict(animate=True, duration=1, frame_rate=5, predict=predict,
                t=5, slow_warning_seconds=0)
    alone = 0
    for train in TRAIN:
        _, messages = _caught(train, **anim)
        cost = [m for m in messages if 'forecast fits' in m]
        assert len(cost) == n_models
        alone += sum(int(re.search(r'needs (\d+) forecast fits', m).group(1))
                     for m in cost)

    grid, messages = _caught(TRAIN, panels=(1, N_PANELS), **anim)
    total, text = _fits(messages, r'needs about (\d+) forecast fits in all')
    assert total == alone
    assert f'({N_PANELS} panels;' in text
    seconds = float(re.search(r'roughly ([\d.]+) s in total', text).group(1))
    assert seconds > 0

    # silenced like the single-axes notice
    _, messages = _caught(TRAIN, panels=(1, N_PANELS),
                          **dict(anim, slow_warning_seconds=None))
    assert not [m for m in messages if 'forecast fits' in m]


# --------------------------------------------------------------------------
# Q4: a per-observation forecast_hue= gets the single-axes error
# --------------------------------------------------------------------------

@pytest.mark.parametrize('hue', [
    [np.linspace(0, 1, 10)] * N_PANELS,
    [[0.0, 1.0]] * N_PANELS,
    [(1, 2)] * N_PANELS], ids=['arrays', 'lists', 'tuples'])
@pytest.mark.parametrize('grid', [{}, dict(panels=(1, N_PANELS))],
                         ids=['single', 'grid'])
def test_a_per_observation_forecast_hue_is_refused_clearly(hue, grid):
    with pytest.raises(ValueError,
                       match='takes one SCALAR value per dataset'):
        hyp.plot(TRAIN, predict='Kalman', t=10, forecast_hue=hue,
                 show=False, **grid)
