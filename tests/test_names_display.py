"""Per-dataset `names=` and the notebook double-display fix (QC 2026-07).

names= gives each dataset in a list its own legend name -- distinct from
per-point `labels=` (text call-outs on observations) and the `legend=True`
auto-numbering. Jeremy's Smooth-kernel comparison had used labels= for dataset
names, which mis-rendered them as point annotations.

Double-display: the plotly backend called fig.show() internally AND plot()
returns the Figure, so a notebook rich-displayed it twice (and plotly jumped
ahead of matplotlib). fig.show() is now skipped in an interactive shell.

Real data, no mocks; plots run headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp


def _datasets(n=3, rows=40, cols=3):
    rng = np.random.default_rng(0)
    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0) for _ in range(n)]


# --- names= ------------------------------------------------------------

def test_names_set_per_dataset_legend_matplotlib():
    data = _datasets(4)
    fig = hyp.plot(data, names=['raw', 'a', 'b', 'c'], show=False)
    leg = fig.axes[0].get_legend()
    assert leg is not None
    assert [t.get_text() for t in leg.get_texts()] == ['raw', 'a', 'b', 'c']


def test_names_do_not_create_point_annotations():
    # regression: names must NOT become per-point text (the old labels= misuse)
    data = _datasets(4)
    fig = hyp.plot(data, names=['raw', 'a', 'b', 'c'], show=False)
    assert len(fig.axes[0].texts) == 0


def test_names_render_on_plotly_backend():
    pytest.importorskip('plotly')
    data = _datasets(3)
    fig = hyp.plot(data, names=['alpha', 'beta', 'gamma'], backend='plotly',
                   show=False)
    trace_names = [tr.name for tr in fig.data if tr.name]
    for want in ('alpha', 'beta', 'gamma'):
        assert want in trace_names


def test_names_wrong_length_raises():
    data = _datasets(4)
    with pytest.raises(ValueError, match='one entry per dataset'):
        hyp.plot(data, names=['a', 'b'], show=False)


def test_names_and_legend_list_conflict_raises():
    data = _datasets(4)
    with pytest.raises(ValueError, match='names= OR a legend='):
        hyp.plot(data, names=['a', 'b', 'c', 'd'],
                 legend=['w', 'x', 'y', 'z'], show=False)


# --- double-display ----------------------------------------------------

def test_plotly_show_not_called_in_interactive_shell(monkeypatch):
    pytest.importorskip('plotly')
    import plotly.graph_objects as go
    import IPython
    calls = {'n': 0}
    monkeypatch.setattr(go.Figure, 'show',
                        lambda self, *a, **k: calls.__setitem__('n', calls['n'] + 1))
    # simulate an interactive notebook: the returned figure auto-displays, so
    # fig.show() must NOT be called (that was the double display)
    monkeypatch.setattr(IPython, 'get_ipython', lambda: object())
    fig = hyp.plot(_datasets(2), backend='plotly', show=True)
    assert calls['n'] == 0
    assert fig is not None  # figure still returned for the user


def test_plotly_show_called_in_plain_script(monkeypatch):
    pytest.importorskip('plotly')
    import plotly.graph_objects as go
    import IPython
    calls = {'n': 0}
    monkeypatch.setattr(go.Figure, 'show',
                        lambda self, *a, **k: calls.__setitem__('n', calls['n'] + 1))
    # plain script (no IPython frontend): fig.show() IS called so the plot
    # still displays
    monkeypatch.setattr(IPython, 'get_ipython', lambda: None)
    hyp.plot(_datasets(2), backend='plotly', show=True)
    assert calls['n'] == 1
