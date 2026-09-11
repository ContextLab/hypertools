"""Forecast / truth= / time-axis findings from the 1.1 release review.

Each test pins one reproduced finding against real rendered artists
(matplotlib ``Line2D`` colours, linestyles and data; plotly trace
properties), on both backends where the path is shared.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                           # noqa: E402
import numpy as np                                        # noqa: E402
import pytest                                            # noqa: E402
import seaborn as sns                                     # noqa: E402
from matplotlib.colors import to_hex, to_rgb              # noqa: E402

import hypertools as hyp                                  # noqa: E402


def _walks(n=3, rows=20, dims=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(0) + 10 * i
            for i in range(n)]


def _role(ax, role):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) == role]


def _ply_role(fig, role):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == role]


def _ply_rgb(color):
    """An ``rgb(...)``/``rgba(...)`` plotly colour string as a 0-1 RGB."""
    body = color[color.index('(') + 1:color.rindex(')')]
    vals = [float(v) for v in body.split(',')[:3]]
    return tuple(round(v / 255.0, 3) for v in vals)


# --- F1: a marker+line fmt draws each dataset as TWO artists ----------------

@pytest.mark.parametrize('fmt, dash', [('o-', '-'), ('s--', '--'),
                                       (['o-', '-', 'x:'], None)])
def test_split_fmt_forecasts_and_truths_take_their_own_datasets_style(
        fmt, dash):
    """``fmt='o-'`` draws a smoothed line plus a markers-only artist per
    dataset; the forecast and truth overlays indexed ONE artist per
    dataset, so three datasets' forecasts came out red, red, green (the
    marker artist of dataset 0 was read as dataset 1's line)."""
    data = _walks()
    truth = [d[-4:] + 1.0 for d in data]
    fig = hyp.plot(data, fmt=fmt, predict='Kalman', t=4, truth=truth,
                   show=False)
    ax = fig.axes[0]
    palette = [to_hex(c) for c in sns.color_palette('hls', 3)]
    fmts = fmt if isinstance(fmt, list) else [fmt] * 3
    want_dash = [dash] if dash else None
    fcs = sorted(_role(ax, 'static'), key=lambda a: a._hyp_forecast_dataset)
    assert [a._hyp_forecast_dataset for a in fcs] == [0, 1, 2]
    assert [to_hex(a.get_color()) for a in fcs] == palette
    expected_ls = [f.lstrip('osx') or '-' for f in fmts]
    if want_dash:
        expected_ls = want_dash * 3
    assert [a.get_linestyle() for a in fcs] == expected_ls
    # every truth artist (curve and its markers) wears its OWN dataset's
    # colour
    for tr in _role(ax, 'truth'):
        assert to_hex(tr.get_color()) == palette[tr._hyp_forecast_dataset]
    plt.close(fig)


def test_split_fmt_forecast_under_hue_runs_continues_the_last_run():
    """Under ``hue=`` runs a forecast continues the run holding the last
    observation -- with ``fmt='o-'`` as with ``fmt='-'``."""
    data = _walks(n=1)[0]
    hue = ['a'] * 10 + ['b'] * 10
    colours = {}
    for fmt in ('-', 'o-'):
        fig = hyp.plot(data, fmt=fmt, hue=hue, predict='Kalman', t=4,
                       show=False)
        ax = fig.axes[0]
        (fc,) = _role(ax, 'static')
        drawn = [ln for ln in ax.lines
                 if getattr(ln, '_hyp_forecast_role', None) is None
                 and ln.get_linestyle() not in ('None', 'none', '')]
        # the LAST drawn run holds the final observation
        assert to_hex(fc.get_color()) == to_hex(drawn[-1].get_color())
        colours[fmt] = to_hex(fc.get_color())
        plt.close(fig)
    assert colours['o-'] == colours['-']


def test_split_fmt_forecast_colours_match_on_plotly():
    data = _walks()
    fig = hyp.plot(data, fmt='o-', predict='Kalman', t=4, backend='plotly',
                   show=False)
    palette = [tuple(round(v, 3) for v in to_rgb(c))
               for c in sns.color_palette('hls', 3)]
    fcs = sorted(_ply_role(fig, 'static'),
                 key=lambda tr: tr.meta['hyp_dataset'])
    got = [_ply_rgb(tr.line.color) for tr in fcs]
    assert len(got) == 3
    for g, p in zip(got, palette):
        assert np.allclose(g, p, atol=0.01)
