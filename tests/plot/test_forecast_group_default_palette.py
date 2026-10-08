"""Default `forecast_hue=`/`forecast_cluster=` colours continue the palette
PAST the observed data's colours (maintainer decision, 1.1 review).

They used to start the palette over, so forecast group 0 was drawn in
exactly dataset 0's observed colour (and, with 'hls' over four datasets,
group 1 in dataset 2's). An explicit `forecast_palette=` still overrides.
"""
import re

import matplotlib
matplotlib.use('Agg')

import numpy as np                                        # noqa: E402
import pytest                                             # noqa: E402
import seaborn as sns                                     # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.colors import to_hex                      # noqa: E402

import hypertools as hyp                                  # noqa: E402


def _walks():
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.standard_normal((36, 5)), 0)
    b = np.cumsum(rng.standard_normal((36, 5)), 0)
    return [a, b, a + 2, b + 2]


GROUPINGS = {
    'cluster': dict(forecast_cluster='KMeans', forecast_n_clusters=2),
    'hue': dict(forecast_hue=['x', 'y', 'x', 'y']),
}


def _hex(c):
    if isinstance(c, str) and c.startswith('rgb'):
        v = [float(x) for x in re.findall(r'[\d.]+', c)[:3]]
        return to_hex([x / 255 for x in v])
    return to_hex(c)


def _mpl(fig, role):
    ax = fig.axes[0]
    obs = [_hex(ln.get_color()) for ln in ax.lines
           if getattr(ln, '_hyp_forecast_role', None) is None
           and len(ln.get_xdata())]
    fc = [_hex(ln.get_color()) for ln in ax.lines
          if getattr(ln, '_hyp_forecast_role', None) == role]
    return obs, fc


def _plotly(fig, role):
    obs = [_hex(tr.line.color) for tr in fig.data
           if (tr.meta or {}).get('hyp_forecast_role') is None
           and tr.line is not None and isinstance(tr.line.color, str)
           and tr.x is not None and len(tr.x)
           and _hex(tr.line.color) != to_hex('black')]   # not the frame
    fc = [_hex(tr.line.color) for tr in fig.data
          if (tr.meta or {}).get('hyp_forecast_role') == role]
    return obs, fc


def _drawn(backend, animated, **kw):
    data = _walks()
    kw = dict(predict='Kalman', t=6, random_state=0, show=False, **kw)
    if animated:
        kw.update(animate=True, duration=1, frame_rate=4)
    out = hyp.plot(data, backend=backend, **kw)
    role = 'live' if animated else 'static'
    if backend == 'plotly':
        return _plotly(out, role)
    fig = out.figure if animated else out
    if animated:
        out.draw_frame(out.n_frames - 1)
    colours = _mpl(fig, role)
    plt.close(fig)
    return colours


@pytest.mark.parametrize('grouping', sorted(GROUPINGS))
@pytest.mark.parametrize('animated', [False, True])
@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_default_group_colours_continue_past_the_observed_ones(
        backend, animated, grouping):
    if backend == 'plotly':
        pytest.importorskip('plotly')
    obs, fc = _drawn(backend, animated, **GROUPINGS[grouping])
    hls4 = [to_hex(c) for c in sns.color_palette('hls', 4)]
    assert obs[:4] == hls4
    # the palette continued past the 4 observed slots: 'hls' at 8 holds
    # all four observed colours, and the groups take its free slots in
    # order (45 and 135 degrees)
    hls8 = [to_hex(c) for c in sns.color_palette('hls', 8)]
    groups = list(dict.fromkeys(fc))
    assert groups == [hls8[1], hls8[3]], (fc, hls8)
    assert not set(fc) & set(obs), 'a forecast group wears an observed colour'
    # the grouping itself is unchanged: datasets 0/2 and 1/3 pair up
    assert fc[0] == fc[2] and fc[1] == fc[3] and fc[0] != fc[1]


def test_a_named_palette_is_continued_not_restarted():
    obs, fc = _drawn('matplotlib', False, palette='Set2',
                     **GROUPINGS['hue'])
    set2 = [to_hex(c) for c in sns.color_palette('Set2', 6)]
    assert obs == set2[:4]
    assert list(dict.fromkeys(fc)) == set2[4:6]


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_an_explicit_forecast_palette_still_overrides(backend):
    if backend == 'plotly':
        pytest.importorskip('plotly')
    _, fc = _drawn(backend, False, forecast_palette=['black', 'orange'],
                   **GROUPINGS['cluster'])
    assert set(fc) == {to_hex('black'), to_hex('orange')}


def test_panels_give_each_label_the_single_axes_colour():
    data = _walks()
    kw = dict(predict='Kalman', t=6, show=False, **GROUPINGS['hue'])
    single = hyp.plot(data, **kw)
    _, single_fc = _mpl(single, 'static')
    plt.close(single)
    grid = hyp.plot(data, panels=True, **kw)
    panel_fc = []
    for ax in grid.axes:
        panel_fc += [_hex(ln.get_color()) for ln in ax.lines
                     if getattr(ln, '_hyp_forecast_role', None) == 'static']
        obs = [_hex(ln.get_color()) for ln in ax.lines
               if getattr(ln, '_hyp_forecast_role', None) is None
               and len(ln.get_xdata())]
        assert not set(obs) & set(panel_fc[-1:])
    plt.close(grid)
    assert panel_fc == single_fc


def test_a_reused_axes_continues_past_every_earlier_observed_colour():
    data = _walks()
    fig, axes = hyp.subplots(1, 1)
    hyp.plot(data[0], ax=axes[0], show=False)
    hyp.plot(data[1:], ax=axes[0], predict='Kalman', t=6, show=False,
             forecast_hue=['x', 'y', 'x'])
    obs, fc = _mpl(fig, 'static')
    plt.close(fig)
    assert len(set(obs)) == 4
    assert len(set(fc)) == 2
    assert not set(fc) & set(obs), (obs, fc)


def test_a_fmt_colour_letter_takes_no_slot_and_no_group_collides():
    """A lettered dataset ('r-') takes no palette slot; the other three
    take the first three colours of the 4-colour sampling, exactly as the
    observed lines are drawn -- so the groups must avoid those three."""
    obs, fc = _drawn('matplotlib', False, fmt=['r-', '-', '-', '-'],
                     **GROUPINGS['hue'])
    assert len(set(fc)) == 2
    assert not set(fc) & set(obs), (obs, fc)


def test_panels_with_a_categorical_observed_hue_match_the_single_axes():
    data = _walks()
    kw = dict(predict='Kalman', t=6, show=False, hue=['p', 'p', 'q', 'q'],
              **GROUPINGS['hue'])
    single = hyp.plot(data, **kw)
    obs, single_fc = _mpl(single, 'static')
    plt.close(single)
    assert not set(single_fc) & set(obs)
    grid = hyp.plot(data, panels=True, **kw)
    panel_fc = [_hex(ln.get_color()) for ax in grid.axes for ln in ax.lines
                if getattr(ln, '_hyp_forecast_role', None) == 'static']
    plt.close(grid)
    assert panel_fc == single_fc


@pytest.mark.parametrize('animated', [False, True])
def test_a_continuous_hue_keeps_an_explicit_forecast_colour(animated):
    """Under a continuous `hue=` matplotlib repainted EVERY forecast in
    its trace's final hue colour, discarding `forecast_hue=` (plotly kept
    it, as `_forecast_style_from` documents): the backends disagreed."""
    pytest.importorskip('plotly')
    kw = dict(hue=[np.arange(36.0)] * 4, **GROUPINGS['hue'])
    _, mpl_fc = _drawn('matplotlib', animated, **kw)
    _, pl_fc = _drawn('plotly', animated, **kw)
    assert len(set(mpl_fc)) == 2
    assert mpl_fc == pl_fc
