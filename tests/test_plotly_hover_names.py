# -*- coding: utf-8 -*-
"""Every hoverable plotly data trace is named like its legend entry.

1.1 release review, maintainer finding: hovering a plotly figure showed
"trace 0", "trace 1", ... -- data traces were unnamed in every plot without
an explicit legend, in the repeat runs of a hue/cluster category and the
leaves of a hierarchy even WITH one, under a continuous hue given
``legend=[...]``, and for ``ndims=1`` series. The contract: a hoverable
data trace is named by the label the legend shows (or would show under
``legend=True``); the legend itself is drawn only when asked for; a lone
unlabelled dataset hides the name box instead of printing a label.

Also L7: an animation's legend entries appeared and vanished frame by frame
because they rode on data traces that are empty until the reveal reaches
them; they now ride on data-free proxies that share the data's legendgroup.

Real figures; the legend=True labels are read from the same figure drawn
with ``legend=True``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp

pytest.importorskip('plotly')


def _walks():
    a = np.asarray(hyp.load('random_walk', n_samples=48, n_features=6,
                            random_state=41))
    b = np.asarray(hyp.load('random_walk', n_samples=48, n_features=6,
                            random_state=42))
    return a, b


def _hoverable_data(fig):
    return [t for t in fig.data
            if t.type in ('scatter', 'scatter3d')
            and (t.meta or {}).get('hyp_trace_index') is not None
            and t.hoverinfo != 'skip']


def _hides_name_box(trace):
    return (trace.hovertemplate or '').endswith('<extra></extra>')


def _assert_every_hoverable_trace_named(fig):
    for t in fig.data:
        if t.type not in ('scatter', 'scatter3d') or t.hoverinfo == 'skip':
            continue
        if t.x is not None and len(t.x) == 1 and t.x[0] is None:
            continue       # data-free legend key
        assert t.name is not None or _hides_name_box(t), t


def _legend_names(fig):
    return [t.name for t in fig.data if t.showlegend]


CAT = ['x'] * 20 + ['y'] * 14 + ['x'] * 14


def _cases():
    a, b = _walks()
    lin = np.linspace(0, 1, 48)
    return {
        'datasets': (lambda **k: hyp.plot([a, b], **k)),
        'hue_line': (lambda **k: hyp.plot(a, hue=CAT, **k)),
        'hue_markers': (lambda **k: hyp.plot(a, '.', hue=CAT, **k)),
        'cluster': (lambda **k: hyp.plot([a, b], cluster='KMeans',
                                         n_clusters=3, random_state=0, **k)),
        'forecast': (lambda **k: hyp.plot([a, b], predict='Kalman', t=5,
                                          **k)),
        'animated': (lambda **k: hyp.plot([a, b], animate=True, duration=1,
                                          frame_rate=4, **k)),
        'panels': (lambda **k: hyp.plot([a, b], panels=True, **k)),
        'cont_hue': (lambda **k: hyp.plot([a, b], hue=[lin, lin], **k)),
    }


@pytest.mark.parametrize('case', list(_cases()))
def test_no_hoverable_trace_is_unnamed(case):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig = _cases()[case](backend='plotly', show=False)
    _assert_every_hoverable_trace_named(fig)
    # no legend was asked for, so none is drawn
    assert _legend_names(fig) == []


@pytest.mark.parametrize('case', ['datasets', 'hue_line', 'hue_markers',
                                  'cluster'])
def test_names_are_the_labels_legend_true_draws(case):
    make = _cases()[case]
    plain = make(backend='plotly', show=False)
    listed = make(backend='plotly', show=False, legend=True)
    legend = set(_legend_names(listed))
    names = [t.name for t in _hoverable_data(plain)]
    assert names and set(names) == legend
    # and with the legend drawn, every run/segment of a group is named too
    assert set(t.name for t in _hoverable_data(listed)) == legend


def test_repeat_runs_share_their_categorys_legendgroup():
    a, _ = _walks()
    fig = hyp.plot(a, hue=CAT, legend=True, backend='plotly', show=False)
    runs = _hoverable_data(fig)
    assert [t.name for t in runs] == ['x', 'y', 'x']
    # one legend entry for 'x', toggling both of its runs
    assert [t.showlegend for t in runs] == [True, True, False]
    assert runs[0].legendgroup == runs[2].legendgroup == 'x'


def test_hierarchy_leaves_and_means_are_named_by_their_group():
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_product([['US', 'EU'], ['tech', 'fin'],
                                       ['a', 'b']])
    df = pd.DataFrame(np.cumsum(rng.standard_normal((30, 8)), 0),
                      columns=cols)
    for kw in ({}, {'legend': True}):
        fig = hyp.plot(df, backend='plotly', show=False, **kw)
        data = _hoverable_data(fig)
        assert {t.name for t in data} == {'US', 'EU'}
        # grouped so the top-level entry toggles the whole group
        assert {t.legendgroup for t in data} == {'US', 'EU'}
    assert sorted(_legend_names(fig)) == ['EU', 'US']
    # a legend list renames the groups -- leaves included
    fig = hyp.plot(df, backend='plotly', show=False,
                   legend=['America', 'Europe'])
    assert {t.name for t in _hoverable_data(fig)} == {'America', 'Europe'}


def test_continuous_hue_keeps_the_legend_list_as_names():
    a, b = _walks()
    lin = np.linspace(0, 1, 48)
    with pytest.warns(UserWarning, match='legend is not supported'):
        fig = hyp.plot([a, b], hue=[lin, lin], legend=['A', 'B'],
                       backend='plotly', show=False)
    assert [t.name for t in _hoverable_data(fig)] == ['A', 'B']
    assert _legend_names(fig) == []          # still no legend drawn


def test_series_mode_curves_are_named_by_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'temp': np.cumsum(rng.standard_normal(40)),
                       'rain': np.cumsum(rng.standard_normal(40))})
    fig = hyp.plot(df, ndims=1, reduce=None, backend='plotly', show=False)
    assert [t.name for t in _hoverable_data(fig)] == ['temp', 'rain']
    assert _legend_names(fig) == []


def test_lone_unlabelled_dataset_hides_the_name_box():
    a, _ = _walks()
    for ndims in (2, 3):
        fig = hyp.plot(a, ndims=ndims, backend='plotly', show=False)
        tr, = _hoverable_data(fig)
        assert tr.name is None and _hides_name_box(tr)
    # ...but a lone NAMED dataset shows its name
    fig = hyp.plot(a, names=['walk'], backend='plotly', show=False)
    tr, = _hoverable_data(fig)
    assert tr.name == 'walk' and not _hides_name_box(tr)


def test_animation_names_persist_into_frames():
    a, b = _walks()
    fig = hyp.plot([a, b], animate=True, duration=1, frame_rate=4,
                   backend='plotly', show=False)
    base = {fig.data.index(t): t.name for t in _hoverable_data(fig)}
    assert set(base.values()) == {'1', '2'}
    for frame in fig.frames:
        for k, tr in zip(frame.traces, frame.data):
            # a frame rewrites geometry only; it never renames a trace
            assert tr.name is None or tr.name == base.get(k, tr.name)


# --------------------------------------------------------------------------
# L7: the animated legend is complete from frame 0


def test_animated_legend_entries_ride_on_data_free_proxies():
    a, b = _walks()
    fig = hyp.plot([a, b], cluster='KMeans', n_clusters=3, random_state=0,
                   legend=True, animate=True, duration=1.5, frame_rate=6,
                   backend='plotly', show=False)
    entries = [t for t in fig.data if t.showlegend]
    assert sorted(t.name for t in entries) == ['0', '1', '2']
    for t in entries:
        # data-free: nothing a frame could empty
        assert list(t.x) == [None]
        assert t.meta["hyp_legend_proxy"] == t.name
        members = [d for d in _hoverable_data(fig)
                   if d.legendgroup == t.legendgroup]
        assert members and all(d.name == t.name for d in members)
    # the data traces themselves carry no legend entry, and no frame
    # touches a proxy
    assert not any(d.showlegend for d in _hoverable_data(fig))
    proxy_idx = {fig.data.index(t) for t in entries}
    for frame in fig.frames:
        assert not proxy_idx & set(frame.traces)


def test_static_legend_stays_on_the_data_traces():
    a, b = _walks()
    fig = hyp.plot([a, b], legend=True, backend='plotly', show=False)
    assert [t.name for t in _hoverable_data(fig) if t.showlegend] == \
        ['1', '2']


# --------------------------------------------------------------------------
# nested dataset lists: every leaf is named (and grouped) by its OUTER group
# (1.1 release review: the leaves were named '1', '2', '2', '4' -- the
# group label on each group's summary leaf, the flat leaf NUMBER on the
# rest -- so hover named the wrong group and the legendgroups paired a
# leaf of group 1 with a leaf of group 2)


def _nested_leaves(fig):
    return sorted(_hoverable_data(fig),
                  key=lambda t: t.meta['hyp_trace_index'])


def _assert_grouped_like(leaves, expected):
    assert [t.name for t in leaves] == expected
    for name in set(expected):
        members = [t for t in leaves if t.name == name]
        others = [t for t in leaves if t.name != name]
        if len(members) > 1:
            # one legendgroup per outer group, shared by exactly its leaves
            assert {t.legendgroup for t in members} == {name}
            assert not any(t.legendgroup == name for t in others)


@pytest.mark.parametrize('legend', [None, True])
def test_nested_list_leaves_are_named_by_their_outer_group(legend):
    a, b = _walks()
    kw = {} if legend is None else {'legend': legend}
    fig = hyp.plot([[a, b], [a + 2, b + 2]], backend='plotly', show=False,
                   **kw)
    leaves = _nested_leaves(fig)
    _assert_grouped_like(leaves, ['1', '1', '2', '2'])
    # colours were already right: a group's leaves share its colour
    assert leaves[0].line.color == leaves[1].line.color
    assert leaves[2].line.color == leaves[3].line.color
    assert leaves[0].line.color != leaves[2].line.color
    if legend:
        # one shown entry per group, toggling that group's leaves
        assert [t.showlegend for t in leaves] == [True, False, True, False]
    else:
        assert _legend_names(fig) == []


def test_nested_list_legend_list_names_the_groups_and_uneven_depths():
    a, b = _walks()
    fig = hyp.plot([[a, b, a - 1], [a + 2, b + 2]], backend='plotly',
                   show=False, legend=['left', 'right'])
    _assert_grouped_like(_nested_leaves(fig),
                         ['left', 'left', 'left', 'right', 'right'])
    assert sorted(_legend_names(fig)) == ['left', 'right']
    # varying depth: the group's summary leaf carries the entry
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig = hyp.plot([a, [b, [a + 1]]], backend='plotly', show=False,
                       legend=True)
    leaves = _nested_leaves(fig)
    _assert_grouped_like(leaves, ['1', '2', '2'])
    assert [t.showlegend for t in leaves] == [True, True, False]


def test_nested_list_per_leaf_legend_list_still_names_the_leaves():
    a, b = _walks()
    fig = hyp.plot([[a, b], [a + 2, b + 2]], backend='plotly', show=False,
                   legend=['p', 'q', 'r', 's'])
    assert [t.name for t in _nested_leaves(fig)] == ['p', 'q', 'r', 's']


def test_nested_list_animation_names_and_groups_by_outer_group():
    a, b = _walks()
    fig = hyp.plot([[a, b], [a + 2, b + 2]], animate=True, duration=1,
                   frame_rate=4, legend=True, backend='plotly', show=False)
    leaves = _nested_leaves(fig)
    _assert_grouped_like(leaves, ['1', '1', '2', '2'])
    entries = [t for t in fig.data if t.showlegend]
    assert sorted(t.name for t in entries) == ['1', '2']
    for t in entries:
        members = [d for d in leaves if d.legendgroup == t.legendgroup]
        assert len(members) == 2 and all(d.name == t.name for d in members)


def test_stacked_dict_hierarchy_leaves_are_named_by_their_group():
    a, b = _walks()
    stacked = hyp.stack({'US': {'x': a, 'y': b},
                         'EU': {'x': a + 2, 'y': b + 2}})
    fig = hyp.plot(stacked, backend='plotly', show=False, legend=True)
    data = _hoverable_data(fig)
    assert {t.name for t in data} == {'US', 'EU'}
    assert {t.legendgroup for t in data} == {'US', 'EU'}
    assert sorted(_legend_names(fig)) == ['EU', 'US']
