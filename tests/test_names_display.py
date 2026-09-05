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

class _FakeEvents:
    def __init__(self):
        self.callbacks = {}

    def register(self, name, cb):
        self.callbacks.setdefault(name, []).append(cb)

    def unregister(self, name, cb):
        self.callbacks[name].remove(cb)

    def fire(self, name):
        for cb in list(self.callbacks.get(name, [])):
            cb()


class _FakeShell:
    """Enough of an InteractiveShell for the display path: a post_execute
    event registry and an execution count."""
    def __init__(self):
        self.events = _FakeEvents()
        self.execution_count = 1


def _count_shows(monkeypatch):
    import plotly.io as pio
    calls = {'n': 0}
    monkeypatch.setattr(pio, 'show', lambda fig, *a, **k: calls.__setitem__('n', calls['n'] + 1))
    return calls


def test_plotly_plot_displays_once_at_the_end_of_the_cell(monkeypatch):
    """`fig = hyp.plot(x)` draws in a notebook (as on matplotlib), but only
    when the cell finishes -- after matplotlib-inline's flush -- not mid-cell."""
    pytest.importorskip('plotly')
    import IPython
    calls = _count_shows(monkeypatch)
    shell = _FakeShell()
    monkeypatch.setattr(IPython, 'get_ipython', lambda: shell)
    fig = hyp.plot(_datasets(2), backend='plotly', show=True)
    assert calls['n'] == 0                       # nothing mid-cell
    assert shell.events.callbacks['post_execute']
    shell.events.fire('post_execute')            # the cell ends
    assert calls['n'] == 1
    assert not shell.events.callbacks['post_execute']   # one-shot
    assert fig is not None


def test_plotly_plot_as_the_last_expression_is_not_drawn_twice(monkeypatch):
    pytest.importorskip('plotly')
    import IPython
    calls = _count_shows(monkeypatch)
    shell = _FakeShell()
    monkeypatch.setattr(IPython, 'get_ipython', lambda: shell)
    fig = hyp.plot(_datasets(2), backend='plotly', show=True)
    fig._ipython_display_()                      # the rich-display hook (cell ends with `fig`)
    assert calls['n'] == 1
    shell.events.fire('post_execute')
    assert calls['n'] == 1                       # skipped: already displayed
    shell.execution_count += 1
    fig._ipython_display_()                      # a later cell displays it again
    assert calls['n'] == 2


def test_plotly_two_figures_in_one_cell_display_in_creation_order(monkeypatch):
    pytest.importorskip('plotly')
    import IPython
    import plotly.io as pio
    order = []
    monkeypatch.setattr(pio, 'show', lambda fig, *a, **k: order.append(fig))
    shell = _FakeShell()
    monkeypatch.setattr(IPython, 'get_ipython', lambda: shell)
    a = hyp.plot(_datasets(1), backend='plotly', show=True)
    b = hyp.plot(_datasets(2), backend='plotly', show=True)
    shell.events.fire('post_execute')
    assert order == [a, b]


def test_plotly_show_false_defers_entirely_to_the_display_hook(monkeypatch):
    pytest.importorskip('plotly')
    import IPython
    calls = _count_shows(monkeypatch)
    shell = _FakeShell()
    monkeypatch.setattr(IPython, 'get_ipython', lambda: shell)
    fig = hyp.plot(_datasets(2), backend='plotly', show=False)
    assert calls['n'] == 0 and 'post_execute' not in shell.events.callbacks
    fig._ipython_display_()
    assert calls['n'] == 1


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
