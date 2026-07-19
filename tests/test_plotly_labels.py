# -*- coding: utf-8 -*-
"""`labels=` point annotations on the plotly backend (GH #205 F3).

F2 (see tests/test_multibyte.py's history) found that `plotly_draw` accepted
`labels=` for call-signature parity with matplotlib's `_draw` but never drew
it -- a silent no-op. This module locks in the real implementation: plotly
now renders one annotation per non-None label, at EXACTLY the same anchor
points matplotlib's `annotate_plot` uses (`layout.scene.annotations` for 3-D,
`layout.annotations` for 2-D), with the same label-to-point mapping
semantics (a list-of-lists is flattened per dataset; `None` entries are
skipped; a `labels` list shorter than the point count raises `IndexError`,
one longer just has its extra entries ignored) -- verified by calling BOTH
backends directly (`_draw` / `plotly_draw`) on the identical `data`/`labels`
and comparing matplotlib's own recorded anchor points (`labels_and_points`,
set as a module global by `annotate_plot`) against plotly's annotations.

No mocks: every assertion here calls the real backend renderers.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest

from hypertools.plot import matplotlib_backend as mb
from hypertools.plot.plotly_backend import plotly_draw


def _walks(n, k, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, d)) for _ in range(k)]


def _mpl_anchor_points(data, labels, fmt=None, kwargs_list=None):
    """Run the matplotlib backend directly and return its recorded
    (text, x, y[, z]) anchor points (`annotate_plot`'s `labels_and_points`,
    exposed as a module global on `hypertools.plot.matplotlib_backend`)."""
    n = len(data)
    fmt = fmt if fmt is not None else ['-'] * n
    kwargs_list = kwargs_list if kwargs_list is not None else [{}] * n
    fig, ax, out, ani = mb._draw(
        data, labels=labels, kwargs_list=kwargs_list, fmt=fmt, show=False)
    points = [(tup[0].get_text(),) + tuple(tup[1:]) for tup in mb.labels_and_points]
    plt.close(fig)
    return points


# --------------------------------------------------------------- 3-D static

def test_plotly_3d_labels_count_and_positions_match_mpl():
    # one point per dataset, so a flat (non-nested) `labels` list of length
    # 3 maps 1:1 onto the 3 stacked points -- `annotate_plot` only flattens
    # `labels` when it is a list of per-dataset LISTS (`any(isinstance(el,
    # list) ...)`); a flat list of scalars is used as-is, one entry per
    # STACKED point (not per dataset).
    data = _walks(1, k=3)
    labels = ['a', 'b', 'c']
    kwargs_list = [{}] * 3
    fmt = ['-'] * 3

    mpl_points = _mpl_anchor_points(data, labels, fmt=fmt,
                                    kwargs_list=kwargs_list)
    fig = plotly_draw(data, labels=labels, fmt=fmt, kwargs_list=kwargs_list,
                      show=False)
    annotations = fig.layout.scene.annotations

    assert len(annotations) == len(mpl_points) == 3
    for ann, (text, x, y, z) in zip(annotations, mpl_points):
        assert ann.text == text
        assert ann.x == pytest.approx(x)
        assert ann.y == pytest.approx(y)
        assert ann.z == pytest.approx(z)


def test_plotly_3d_labels_none_entries_skipped_matches_mpl():
    # nested per-dataset lists: each inner list's length must match its
    # OWN dataset's point count (4), since `annotate_plot` flattens them
    # with `itertools.chain` before indexing into the stacked (4+4=8)
    # points.
    data = _walks(4, k=2)
    labels = [['x', None, None, None], [None, None, None, 'y']]
    kwargs_list = [{}] * 2
    fmt = ['-'] * 2

    mpl_points = _mpl_anchor_points(data, labels, fmt=fmt,
                                    kwargs_list=kwargs_list)
    fig = plotly_draw(data, labels=labels, fmt=fmt, kwargs_list=kwargs_list,
                      show=False)
    annotations = fig.layout.scene.annotations

    assert len(annotations) == len(mpl_points) == 2
    assert {a.text for a in annotations} == {'x', 'y'}
    for ann, (text, x, y, z) in zip(annotations, mpl_points):
        assert ann.text == text
        assert ann.x == pytest.approx(x)
        assert ann.y == pytest.approx(y)
        assert ann.z == pytest.approx(z)


# --------------------------------------------------------------- 2-D static

def test_plotly_2d_labels_count_and_positions_match_mpl():
    data = _walks(3, k=2, d=2)
    labels = [['p', None, 'q'], [None, 'r', None]]
    kwargs_list = [{}] * 2
    fmt = ['-'] * 2

    mpl_points = _mpl_anchor_points(data, labels, fmt=fmt,
                                    kwargs_list=kwargs_list)
    fig = plotly_draw(data, labels=labels, fmt=fmt, kwargs_list=kwargs_list,
                      show=False)
    annotations = fig.layout.annotations

    assert len(annotations) == len(mpl_points) == 3
    for ann, (text, x, y) in zip(annotations, mpl_points):
        assert ann.text == text
        assert ann.x == pytest.approx(x)
        assert ann.y == pytest.approx(y)
        # 2-D annotations are anchored in DATA space, not paper space
        assert ann.xref == 'x'
        assert ann.yref == 'y'


# ------------------------------------------------------------ mismatched count

def test_plotly_labels_shorter_than_points_raises_indexerror_like_mpl():
    data = _walks(3, k=1)
    kwargs_list = [{}]
    fmt = ['-']
    short_labels = ['a', 'b']  # 3 points, only 2 labels

    with pytest.raises(IndexError):
        mb._draw(data, labels=short_labels, kwargs_list=kwargs_list,
                fmt=fmt, show=False)
    with pytest.raises(IndexError):
        plotly_draw(data, labels=short_labels, kwargs_list=kwargs_list,
                   fmt=fmt, show=False)


def test_plotly_labels_longer_than_points_extras_ignored_like_mpl():
    data = _walks(3, k=1)
    kwargs_list = [{}]
    fmt = ['-']
    long_labels = ['a', 'b', 'c', 'd']  # 3 points, 4 labels

    mpl_points = _mpl_anchor_points(data, long_labels, fmt=fmt,
                                    kwargs_list=kwargs_list)
    fig = plotly_draw(data, labels=long_labels, fmt=fmt,
                      kwargs_list=kwargs_list, show=False)
    annotations = fig.layout.scene.annotations

    assert len(mpl_points) == len(annotations) == 3
    assert [a.text for a in annotations] == ['a', 'b', 'c']


# ---------------------------------------------------------------- 1-D no-op

def test_plotly_1d_labels_draw_nothing_like_mpl():
    data = _walks(5, k=1, d=1)
    kwargs_list = [{}]
    fmt = ['-']
    labels = ['a', 'b', 'c', 'd', 'e']

    fig, ax, out, ani = mb._draw(data, labels=labels, kwargs_list=kwargs_list,
                                 fmt=fmt, show=False)
    # 1-D has neither the >2 nor the ==2 branch in annotate_plot -- no
    # labels_and_points entries are recorded at all
    assert mb.labels_and_points == []
    plt.close(fig)

    fig2 = plotly_draw(data, labels=labels, fmt=fmt, kwargs_list=kwargs_list,
                       show=False)
    assert not fig2.layout.annotations
    assert fig2.layout.scene is None or not fig2.layout.scene.annotations


# ------------------------------------------------------------------ animate

def test_plotly_animated_3d_labels_drawn_same_as_mpl_not_skipped_or_raised():
    # matplotlib neither skips nor raises for animate=True + labels= (`_draw`
    # calls `add_labels` unconditionally after dispatching either the static
    # or animated path) -- plotly must match that (not invent a skip/raise).
    data = np.cumsum(
        np.random.default_rng(0).standard_normal((30, 3)), axis=0)
    labels = ['start'] + [None] * 28 + ['end']

    fig_mpl, ax, out, ani = mb._draw(
        [data], labels=[labels], kwargs_list=[{}], fmt=['-'],
        animate=True, duration=1, frame_rate=5, show=False)
    mpl_points = [(tup[0].get_text(),) + tuple(tup[1:])
                 for tup in mb.labels_and_points]
    plt.close(fig_mpl)

    fig_plotly = plotly_draw(
        [data], labels=[labels], kwargs_list=[{}], fmt=['-'],
        animate=True, duration=1, frame_rate=5, show=False)
    annotations = fig_plotly.layout.scene.annotations

    assert len(annotations) == len(mpl_points) > 0
    for ann, (text, x, y, z) in zip(annotations, mpl_points):
        assert ann.text == text
        assert ann.x == pytest.approx(x)
        assert ann.y == pytest.approx(y)
        assert ann.z == pytest.approx(z)
