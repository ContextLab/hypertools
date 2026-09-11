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


# --- names= vs an explicit legend=False (1.1 review) -------------------
#
# names= turns the legend on by default, but it used to override an
# explicit legend=False as well (found in the stock_forecasting tutorial):
# the opt-out must win on both backends.

def _mpl_legend_texts(fig):
    """Every legend entry drawn anywhere on a matplotlib figure (axes
    legends and figure-level legends)."""
    legends = [ax.get_legend() for ax in fig.axes] + list(fig.legends)
    return [t.get_text() for leg in legends if leg is not None
            for t in leg.get_texts()]


def test_names_legend_false_draws_no_legend_matplotlib():
    data = _datasets(3)
    fig = hyp.plot(data, names=['a', 'b', 'c'], legend=False, show=False)
    assert _mpl_legend_texts(fig) == []
    # the default (no legend=) still shows the names
    fig = hyp.plot(data, names=['a', 'b', 'c'], show=False)
    assert _mpl_legend_texts(fig) == ['a', 'b', 'c']


def test_names_legend_false_draws_no_legend_plotly():
    pytest.importorskip('plotly')
    data = _datasets(3)
    fig = hyp.plot(data, names=['a', 'b', 'c'], legend=False,
                   backend='plotly', show=False)
    shown = [tr.name for tr in fig.data if tr.showlegend is not False]
    assert fig.layout.showlegend is not True
    assert not any(name in ('a', 'b', 'c') for name in shown)
    # identical legend state to the same call without names=
    bare = hyp.plot(data, legend=False, backend='plotly', show=False)
    assert fig.layout.showlegend == bare.layout.showlegend
    assert ([tr.showlegend for tr in fig.data]
            == [tr.showlegend for tr in bare.data])


def test_names_legend_false_panels_draw_no_legend():
    data = _datasets(3)
    fig = hyp.plot(data, names=['a', 'b', 'c'], legend=False, panels=True,
                   show=False)
    assert len(fig.axes) >= 3
    assert _mpl_legend_texts(fig) == []


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('extra', [{'predict': 'Kalman', 't': 5},
                                   {'animate': True}],
                         ids=['predict', 'animate'])
def test_names_legend_false_forecast_and_animation(backend, extra):
    # the stock_forecasting tutorial's shape: named datasets plus a
    # forecast overlay (and the animated form), with legend=False
    if backend == 'plotly':
        pytest.importorskip('plotly')
    data = _datasets(3, rows=60, cols=2)
    for legend_kw, want in (({}, True), ({'legend': False}, False)):
        out = hyp.plot(data, names=['a', 'b', 'c'], backend=backend,
                       show=False, **extra, **legend_kw)
        if backend == 'plotly':
            shown = [tr.name for tr in out.data if tr.showlegend]
            drawn = bool(out.layout.showlegend) and bool(shown)
            assert (set('abc') <= set(shown)) is want
        else:
            fig = getattr(out, 'figure', out)
            texts = _mpl_legend_texts(fig)
            drawn = bool(texts)
            assert (set('abc') <= set(texts)) is want
        assert drawn is want


def test_names_legend_false_still_validates_names():
    # legend=False hides the names; it does not make a malformed names=
    # list acceptable
    data = _datasets(3)
    with pytest.raises(ValueError, match='one entry per dataset'):
        hyp.plot(data, names=['a', 'b'], legend=False, show=False)


# --- double-display ----------------------------------------------------
#
# Every notebook scenario below runs on a REAL `IPython.InteractiveShell`
# (in-process, history disabled): `hyp.plot` is executed as cell source via
# `shell.run_cell`, so `get_ipython()`, the `post_execute` event registry, the
# rich-display hook (`_ipython_display_` of a cell's last expression) and the
# execution counter are IPython's own. Display events are observed with
# IPython's `capture_output` (the machinery behind `%%capture`), which records
# every published mime-bundle in order; plotly is pointed at its `json`
# renderer, whose bundle is the figure's JSON, so each captured
# `application/json` output IS one `pio.show` of an identifiable figure.

_CELL_SETUP = (
    "import numpy as np\n"
    "import hypertools as hyp\n"
    "from IPython.display import display\n"
    "from IPython import get_ipython\n"
    "from hypertools.plot.plotly_backend import _flush_pending_display\n"
    "def _datasets(n=3, rows=40, cols=3):\n"
    "    rng = np.random.default_rng(0)\n"
    "    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0)"
    " for _ in range(n)]\n"
)


@pytest.fixture
def ipython_shell():
    """A real in-process IPython shell, torn down so later tests run as a
    plain script again (`get_ipython()` is None after teardown)."""
    pytest.importorskip('plotly')
    import IPython
    import plotly.io as pio
    from IPython.core.interactiveshell import InteractiveShell
    from traitlets.config import Config
    from hypertools.plot import plotly_backend

    assert IPython.get_ipython() is None, 'a shell is already running'
    saved_renderer = pio.renderers.default
    plotly_backend._PENDING_DISPLAY.clear()
    config = Config()
    config.HistoryManager.enabled = False
    shell = InteractiveShell.instance(config=config)
    try:
        pio.renderers.default = 'json'
        result = shell.run_cell(_CELL_SETUP, store_history=True)
        assert result.success, result.error_in_exec
        yield shell
    finally:
        pio.renderers.default = saved_renderer
        plotly_backend._PENDING_DISPLAY.clear()
        InteractiveShell.clear_instance()
        shell.restore_sys_module_state()
        assert IPython.get_ipython() is None


def _run(shell, source):
    """Run one cell; return the mime-bundles it displayed, in order."""
    from IPython.utils.capture import capture_output
    with capture_output(display=True) as captured:
        result = shell.run_cell(source, store_history=True)
    assert result.success, result.error_in_exec
    return [dict(out.data) for out in captured.outputs]


def _figure_shows(outputs):
    return [out['application/json'] for out in outputs
            if 'application/json' in out]


def _flush_registered(shell):
    from hypertools.plot.plotly_backend import _flush_pending_display
    return _flush_pending_display in shell.events.callbacks['post_execute']


def test_plotly_plot_displays_once_at_the_end_of_the_cell(ipython_shell):
    """`fig = hyp.plot(x)` draws in a notebook (as on matplotlib), but only
    when the cell finishes -- after matplotlib-inline's flush -- not mid-cell."""
    shell = ipython_shell
    outputs = _run(shell, (
        "fig = hyp.plot(_datasets(2), backend='plotly', show=True)\n"
        "registered = _flush_pending_display in"
        " get_ipython().events.callbacks['post_execute']\n"
        "display('end of cell body')\n"))
    # the plot call queued a cell-end hook ...
    assert shell.user_ns['registered'] is True
    # ... and nothing was drawn mid-cell: the marker displayed by the LAST
    # statement precedes the one figure display, which the cell end produced
    assert [list(out) for out in outputs] == [['text/plain'],
                                             ['application/json']]
    assert outputs[0]['text/plain'] == "'end of cell body'"
    assert not _flush_registered(shell)                  # one-shot
    assert _figure_shows(_run(shell, "pass")) == []       # nothing queued
    assert shell.user_ns['fig'] is not None


def test_plotly_plot_as_the_last_expression_is_not_drawn_twice(ipython_shell):
    shell = ipython_shell
    # the rich-display hook (cell ends with `fig`) draws it; the cell-end
    # flush must then skip it
    outputs = _run(shell, (
        "fig = hyp.plot(_datasets(2), backend='plotly', show=True)\n"
        "fig\n"))
    assert len(_figure_shows(outputs)) == 1
    assert not _flush_registered(shell)
    # a later cell displays it again
    later = shell.execution_count
    outputs = _run(shell, "fig")
    assert shell.execution_count > later
    assert len(_figure_shows(outputs)) == 1


def test_plotly_two_figures_in_one_cell_display_in_creation_order(ipython_shell):
    import json
    shell = ipython_shell
    outputs = _run(shell, (
        "a = hyp.plot(_datasets(1), backend='plotly', show=True)\n"
        "b = hyp.plot(_datasets(2), backend='plotly', show=True)\n"))
    a, b = shell.user_ns['a'], shell.user_ns['b']
    assert len(a.data) != len(b.data)          # distinguishable figures
    shown = _figure_shows(outputs)
    assert [len(fig['data']) for fig in shown] == [len(a.data), len(b.data)]
    assert shown == [json.loads(a.to_json()), json.loads(b.to_json())]


def test_plotly_show_false_defers_entirely_to_the_display_hook(ipython_shell):
    shell = ipython_shell
    outputs = _run(shell, (
        "fig = hyp.plot(_datasets(2), backend='plotly', show=False)\n"
        "registered = _flush_pending_display in"
        " get_ipython().events.callbacks['post_execute']\n"))
    assert shell.user_ns['registered'] is False
    assert outputs == []
    assert len(_figure_shows(_run(shell, "fig"))) == 1


def test_plotly_show_called_in_plain_script(capsys):
    """Plain script (no IPython frontend): fig.show() IS called so the plot
    still displays. Outside IPython the json renderer's bundle is printed to
    stdout by `IPython.display.display`, so one printed bundle is one show."""
    pytest.importorskip('plotly')
    import json
    import IPython
    import plotly.io as pio
    from hypertools.plot import plotly_backend

    assert IPython.get_ipython() is None
    saved_renderer = pio.renderers.default
    pio.renderers.default = 'json'
    try:
        capsys.readouterr()
        fig = hyp.plot(_datasets(2), backend='plotly', show=True)
        out = capsys.readouterr().out
        assert out.count("'application/json'") == 1
        assert str({'application/json': json.loads(fig.to_json())}) in out
        assert plotly_backend._PENDING_DISPLAY == []   # nothing deferred
        hyp.plot(_datasets(2), backend='plotly', show=False)
        assert capsys.readouterr().out == ''           # show=False: no show
    finally:
        pio.renderers.default = saved_renderer
