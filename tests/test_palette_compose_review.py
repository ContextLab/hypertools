"""1.1 release review (2026-09-11): palette / marker / legend / panel / ax=
composition findings, each pinned at the public API on both backends where
the path is shared.

1  a per-dataset list of {category: color} dicts was ignored on the
   marker-only (fmt='o') categorical path; the line path applied it.

No mocks: every assertion reads the drawn artists or traces.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

import hypertools as hyp
from tests._plotly_colors import rgba as effective_rgba


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walks(n=2, rows=40, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((rows, 3)), 0) for _ in range(n)]


def _mpl_data_lines(ax):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None
            and len(ln.get_xdata()) > 1]


def _pl_data(fig):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_trace_index') is not None]


def _pl_rgb(trace, component='line'):
    # 8-bit, the precision a plotly colour string carries
    return tuple(round(v * 255) for v in effective_rgba(trace, component)[:3])


def _r3(c):
    return tuple(round(float(v) * 255) for v in to_rgb(c[:3]
                                                      if not isinstance(c, str)
                                                      else c))


# --- 1: per-dataset {category: color} dicts on the marker path -------------

_HUE = [np.repeat(['a', 'b'], 20), np.repeat(['b', 'c'], 20)]
_DICTS = [{'a': 'red', 'b': 'blue'}, {'b': 'blue', 'c': 'green'}]
_BY_NAME = {'a': _r3('red'), 'b': _r3('blue'), 'c': _r3('green')}


@pytest.mark.parametrize('fmt', ['o', '.', 'o-', '-'])
def test_per_dataset_dict_palettes_colour_markers_matplotlib(fmt):
    fig = hyp.plot(_walks(), hue=_HUE, palette=_DICTS, fmt=fmt,
                   legend=True, show=False)
    legend = fig.axes[0].get_legend()
    got = {t.get_text(): _r3(h.get_color())
           for t, h in zip(legend.get_texts(), legend.legend_handles)}
    assert got == _BY_NAME
    drawn = {_r3(ln.get_color()) for ln in fig.axes[0].lines}
    assert drawn == set(_BY_NAME.values())


@pytest.mark.parametrize('fmt', ['o', 'o-', '-'])
def test_per_dataset_dict_palettes_colour_markers_plotly(fmt):
    fig = hyp.plot(_walks(), hue=_HUE, palette=_DICTS, fmt=fmt,
                   legend=True, show=False, backend='plotly')
    comp = 'marker' if fmt == 'o' else 'line'
    got = {tr.name: _pl_rgb(tr, comp) for tr in _pl_data(fig)
           if tr.showlegend is not False}
    assert got == _BY_NAME


def test_per_dataset_dict_palette_bundle_matches_the_markers():
    out = hyp.plot(_walks(), hue=_HUE, palette=_DICTS, fmt='o',
                   show=False, return_model=True)
    cats = {k: _r3(c) for k, c in out['colors']['categories'].items()}
    assert cats == _BY_NAME
