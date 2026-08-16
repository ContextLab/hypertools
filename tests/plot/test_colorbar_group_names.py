# -*- coding: utf-8 -*-
"""The colorbar names the drawn GROUPS -- `legend=` must not decide that.

The colorbar is a colour key, not a legend: it says which colour belongs to
which group. Two ways that came apart, both reproduced below.

A. **A hierarchy under `legend=False`.** `plot()`'s MultiIndex branch
   installs the hierarchy's per-trace labels into `legend`, and
   `_build_colorbar_info` reads BOTH the group names AND the `'_nolegend_'`
   entries that collapse leaves and intermediate means down to the top-level
   groups out of that same list. Making the install conditional on
   `legend is not False` therefore lost both at once, and the colorbar fell
   through to `labels = [i + 1 for i in range(n_groups)]`. Measured on a
   3-level column frame: ticks `['US', 'EU']` -> `['1' .. '6']`; on the ROW
   hierarchy below, 2 segments -> 46 (one per leaf AND per derived mean).

B. **A tuple `legend=`.** `legend=('A', 'B')` labelled the drawn traces
   correctly but missed `_build_colorbar_info`'s `isinstance(legend, list)`
   branch, so the colorbar read `['1', '2']`. The container normalisation in
   `plot()` (see tests/plot/test_legend_containers.py) makes every accepted
   container reach here as a list.

Trap notes: the colorbar is a SECOND axes on the figure, so it is read from
`fig.axes[1]`, never from the data axes; on plotly the colorbar rides on a
phantom trace, found by the trace that actually carries
`marker.colorbar.ticktext`.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def column_frame(T=30, seed=0):
    """(region x sector x measure) = 2 x 2 x 3: 4 leaves + 2 region means."""
    rng = np.random.default_rng(seed)
    columns = pd.MultiIndex.from_tuples(
        [(region, sector, measure)
         for region in ('US', 'EU')
         for sector in ('tech', 'fin')
         for measure in ('a', 'b', 'c')],
        names=['region', 'sector', 'measure'])
    return pd.DataFrame(rng.normal(size=(T, 12)).cumsum(axis=0),
                        columns=columns)


def row_frame(n_time=10, seed=1):
    """(group x subject x time): 40 leaf rows-blocks + 6 derived means.

    The loud case: 2 groups, but 4 leaves + 2 subject-level means + 2
    group-level means are drawn, so a broken colorbar shows many more
    segments than there are groups.
    """
    rng = np.random.default_rng(seed)
    tuples, blocks = [], []
    for gi, group in enumerate(('G1', 'G2')):
        for subject in ('s1', 's2'):
            blocks.append(rng.standard_normal((n_time, 4)).cumsum(axis=0)
                          + gi * 5.0)
            tuples.extend([(group, subject)] * n_time)
    index = pd.MultiIndex.from_tuples(tuples, names=['group', 'subject'])
    return pd.DataFrame(np.vstack(blocks), index=index,
                        columns=list('wxyz'))


def _cbar_labels(fig):
    """The tick labels a READER sees on the colorbar axes."""
    assert len(fig.axes) == 2, 'expected the data axes + one colorbar axes'
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    if not any(labels):
        labels = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    return [lbl for lbl in labels if lbl]


def _plotly_cbar_ticktext(fig):
    for trace in fig.data:
        marker = getattr(trace, 'marker', None)
        colorbar = getattr(marker, 'colorbar', None)
        if colorbar is not None and colorbar.ticktext:
            return list(colorbar.ticktext)
    raise AssertionError('no plotly trace carries a colorbar ticktext')


def _plotly(*args, **kwargs):
    hyp.set_interactive_backend('plotly')
    try:
        return hyp.plot(*args, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')


# ---------------------------------------------------------------- Finding A

@pytest.mark.parametrize('frame_fn, expected', [
    (column_frame, ['US', 'EU']),
    (row_frame, ['G1', 'G2']),
])
def test_legend_false_does_not_rename_the_colorbars_groups(frame_fn,
                                                           expected):
    """Opting out of the LEGEND must leave the colour key alone.

    Both axes of hierarchy, both halves of the regression: the names come
    back, and the `'_nolegend_'` filter that collapses leaves and derived
    means into the top-level groups runs again (so the count is 2, not 6/46).
    """
    frame = frame_fn()
    assert _cbar_labels(hyp.plot(frame, '-', colorbar=True, legend=False,
                                 show=False)) == expected
    # ... and identically to the call that never mentions legend=
    assert _cbar_labels(hyp.plot(frame, '-', colorbar=True,
                                 show=False)) == expected


def test_legend_false_still_draws_no_legend_beside_the_colorbar():
    """The fix must not resurrect the legend it was made to suppress."""
    fig = hyp.plot(row_frame(), '-', colorbar=True, legend=False, show=False)
    assert fig.axes[0].get_legend() is None
    assert _cbar_labels(fig) == ['G1', 'G2']


def test_the_hierarchys_other_legend_values_are_unchanged():
    """The paths that already agreed must keep agreeing: legend=True and a
    caller-supplied list both name the colorbar the same way they name the
    legend."""
    frame = column_frame()
    assert _cbar_labels(hyp.plot(frame, '-', colorbar=True, legend=True,
                                 show=False)) == ['US', 'EU']
    assert _cbar_labels(hyp.plot(frame, '-', colorbar=True,
                                 legend=['A', 'B'], show=False)) == ['A', 'B']


def test_plotly_colorbar_keeps_its_group_names_under_legend_false():
    """Same regression, same fix, on the interactive backend."""
    frame = column_frame()
    assert _plotly_cbar_ticktext(
        _plotly(frame, '-', colorbar=True, legend=False,
                show=False)) == ['US', 'EU']
    assert _plotly_cbar_ticktext(
        _plotly(frame, '-', colorbar=True, show=False)) == ['US', 'EU']


def test_animated_colorbar_keeps_its_group_names_under_legend_false():
    """The animation path resolves the colorbar through the same call."""
    anim = hyp.plot(column_frame(), '-', colorbar=True, legend=False,
                    animate=True, show=False)
    assert _cbar_labels(anim.figure) == ['US', 'EU']


# ---------------------------------------------------------------- Finding B

def test_a_tuple_legend_names_the_colorbar_like_a_list_does():
    """`legend=('A', 'B')` labelled the traces but not the colorbar."""
    rng = np.random.default_rng(2)
    data = [rng.standard_normal((12, 3)).cumsum(axis=0) for _ in range(2)]
    assert _cbar_labels(hyp.plot(data, '-', colorbar=True,
                                 legend=('A', 'B'), show=False)) == ['A', 'B']
    assert _cbar_labels(hyp.plot(data, '-', colorbar=True,
                                 legend=['A', 'B'], show=False)) == ['A', 'B']
