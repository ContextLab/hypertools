# -*- coding: utf-8 -*-
"""The window-aware baseline in the Market representation study.

`scripts/market_representation_study.py` is what killed a flattering
forecast result: a Laplace edge of 0.22-0.40 that held across both blocks
and both horizons collapsed to 0.266 against 0.682 once the baseline below
was added. A study that decides whether a gallery figure may claim
predictive skill is evidence, so its baseline has to be right -- and the
first version of it was right at one horizon only.

The identity under test, with `L` the log level and `W = CUM_WINDOW`:

    cum_return[t]                = L[t] - L[t - W]
    cum_return[t+h] - cum[t]     = (L[t+h] - L[t]) - (L[t+h-W] - L[t-W])
                                    ^ unknown at t     ^ known at t, h <= W

Deterministic throughout: synthetic prices, exact algebra, no network and
no model fit. The point is the arithmetic, not the data.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.market_representation_study import (       # noqa: E402
    CUM_WINDOW, MEASURES, d1_frame, d2_frame, leaf_arrays, leaf_levels,
    monthly_levels, scale_per_measure, window_dropout)

SECTORS = {'Alpha': ['a1', 'a2'], 'Beta': ['b1', 'b2'], 'Gamma': ['g1']}


def _closes(seed=0, days=3200):
    """Daily closes for a small hierarchy: a geometric random walk each."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2010-01-04', periods=days)
    columns = pd.MultiIndex.from_tuples(
        [(sector, ticker) for sector, ts in SECTORS.items() for ticker in ts],
        names=['Sector', 'Ticker'])
    steps = rng.normal(0.0003, 0.013, size=(days, len(columns)))
    return pd.DataFrame(100.0 * np.exp(steps.cumsum(axis=0)), index=index,
                        columns=columns)


def _setup(kind):
    closes = _closes()
    frame = (d1_frame(closes, SECTORS) if kind == 'D1'
             else d2_frame(closes, SECTORS))
    scaled, spread = scale_per_measure(frame)
    return (closes, frame, leaf_arrays(scaled), spread,
            leaf_levels(closes, SECTORS, kind))


@pytest.mark.parametrize('kind', ['D1', 'D2'])
@pytest.mark.parametrize('horizon', [1, 3, 6])
def test_the_dropout_baseline_is_EXACTLY_the_known_half_of_the_change(kind,
                                                                     horizon):
    """Realised change minus the baseline must leave the UNKNOWN half alone.

    This is the whole claim the baseline makes, checked as an identity
    rather than as a correlation: subtract what the window mechanically
    hands you and what is left is `L[t+h] - L[t]`, the part a forecast
    actually has to earn. It is asserted at both representations because
    a D1 sector's `cum_return` is a trailing sum of the sector level and
    carries the same dropout a D2 stock does -- the baseline used to be
    computed for D2 only.
    """
    closes, frame, arrays, spread, series = _setup(kind)
    known = window_dropout(series, frame.index, spread, horizon)
    assert len(known) == len(arrays), 'one baseline per leaf'

    axis = MEASURES.index('cum_return')
    for leaf, values in enumerate(arrays):
        cum = values[:, axis]                       # scaled cum_return
        level = series[leaf].reindex(frame.index).to_numpy()
        # anchors far enough in that both windows are fully populated
        for anchor in range(CUM_WINDOW + 2, len(cum) - horizon, 7):
            last = anchor - 1                       # last OBSERVED row
            realised = cum[last + horizon] - cum[last]
            unknown = (level[last + horizon] - level[last]) / spread['cum_return']
            assert realised - known[leaf][anchor] == pytest.approx(
                unknown, abs=1e-9), (
                f'{kind} leaf {leaf} h={horizon} anchor={anchor}: the '
                f'baseline does not account for exactly the dropped span')


@pytest.mark.parametrize('kind', ['D1', 'D2'])
def test_a_LONGER_horizon_drops_a_LONGER_span(kind):
    """The bug itself: it returned the one-step term at every horizon.

    At h=3 three known returns leave the window, not one, so the baseline
    was scored on a third of the information available to it -- understated
    in exactly the comparison where a model has most room to look good.
    """
    closes, frame, arrays, spread, series = _setup(kind)
    one = window_dropout(series, frame.index, spread, 1)
    three = window_dropout(series, frame.index, spread, 3)
    for leaf in one:
        a, b = one[leaf], three[leaf]
        usable = np.isfinite(a) & np.isfinite(b)
        assert usable.sum() > 50
        assert not np.allclose(a[usable], b[usable]), (
            f'{kind} leaf {leaf}: h=1 and h=3 produced the same baseline, '
            'so the horizon is being ignored')
        # three returns leaving the window vary more than one does
        assert np.std(b[usable]) > np.std(a[usable])


def test_the_identity_is_REFUSED_past_the_window():
    """Beyond CUM_WINDOW steps the span leaving the window has not happened
    yet, so there is nothing known to hand the baseline. Better to refuse
    than to quietly return a term that reaches into the future."""
    closes, frame, arrays, spread, series = _setup('D2')
    with pytest.raises(ValueError, match='1 <= horizon'):
        window_dropout(series, frame.index, spread, CUM_WINDOW + 1)
    with pytest.raises(ValueError, match='1 <= horizon'):
        window_dropout(series, frame.index, spread, 0)


@pytest.mark.parametrize('kind', ['D1', 'D2'])
def test_leaf_levels_are_in_the_SAME_ORDER_as_the_frame_leaves(kind):
    """The baseline is keyed by leaf POSITION, so a mismatch here would
    silently score each leaf against another leaf's window."""
    closes, frame, arrays, spread, series = _setup(kind)
    keys = list(dict.fromkeys(tuple(c[:-1]) for c in frame.columns))
    assert len(series) == len(keys) == len(arrays)

    levels = monthly_levels(closes)
    for position, key in enumerate(keys):
        expected = (levels[key[1]].mean(axis=1) if kind == 'D1'
                    else levels[(key[1], key[2])])
        aligned = expected.reindex(frame.index).to_numpy()
        assert np.allclose(series[position].reindex(frame.index).to_numpy(),
                           aligned, equal_nan=True), (
            f'leaf {position} ({key}) is backed by the wrong level series')
