"""`truth=` in a time-progressing animation is revealed as the forecast
reaches it (maintainer decision, 1.1 review).

Drawing the whole held-out continuation from frame 0 showed the answer
before the animation had forecast anywhere near it. Each truth row now
appears on the first frame whose DRAWN forecast reaches or passes its
position, and stays on every later frame; the 'truth' legend entry is on
every frame. Static plots (and 'spin') still draw it in full.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import pytest                                             # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

import hypertools as hyp                                  # noqa: E402
from hypertools.plot.forecast import truth_reveal_counts  # noqa: E402


def _series():
    """The feature-tour example: 24 hourly samples of a sine, and its next
    3 hours as `truth=`."""
    t = np.arange(24.0)
    data = pd.DataFrame({'signal': np.sin(t / 4)},
                        index=pd.date_range('2026-01-01', periods=24,
                                            freq='h'))
    truth = pd.DataFrame({'signal': np.sin(np.arange(24, 27) / 4)},
                         index=pd.date_range(
                             data.index[-1] + pd.Timedelta('1h'), periods=3,
                             freq='h'))
    return data, truth


def _long_horizon():
    """A 3-D walk forecast 12 steps out, so the truth is reached a few rows
    at a time rather than all at once on the last frame."""
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal((20, 3)), 0)
    held = x[-1] + np.cumsum(rng.standard_normal((12, 3)), 0)
    return x, held


def _mpl_truth_rows(ax):
    """{dataset: visible held-out rows} from the truth MARKER artists (one
    marker per row, the first is the seam)."""
    out = {}
    for ln in ax.lines:
        if (getattr(ln, '_hyp_forecast_role', None) == 'truth'
                and ln.get_linestyle() == 'None'):
            data = (ln.get_data_3d() if hasattr(ln, 'get_data_3d')
                    else ln.get_data())
            n = len(np.asarray(data[0])) if ln.get_visible() else 0
            out[ln._hyp_forecast_dataset] = max(n - 1, 0)
    return out


def _mpl_legend(ax):
    leg = ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def _pl_truth_points(fig, frame):
    """{dataset: drawn truth vertices} at plotly frame `frame`."""
    meta = {i: tr.meta for i, tr in enumerate(fig.data)
            if (tr.meta or {}).get('hyp_forecast_role') == 'truth'}
    fr = fig.frames[frame]
    assert set(meta) <= set(fr.traces), 'a frame does not rewrite the truth'
    return {meta[i]['hyp_dataset']: len(tr.x)
            for i, tr in zip(fr.traces, fr.data) if i in meta}


def test_feature_tour_truth_is_hidden_until_the_forecast_reaches_it():
    data, truth = _series()
    anim = hyp.plot(data, ndims=1, reduce=None,
                    predict=['Kalman', 'GaussianProcess'], t=3, truth=truth,
                    forecast_palette=['orange', 'purple'],
                    forecast_fmt=['--', ':'], legend=True, animate=True,
                    duration=1.5, frame_rate=6, forecast_trail=3, show=False)
    ax = anim.figure.axes[0]
    seen = []
    for k in range(anim.n_frames):
        anim.draw_frame(k)
        rows = _mpl_truth_rows(ax)[0]
        # independent check: every revealed truth row's x lies within the
        # reach of some LIVE forecast drawn on this or an earlier frame,
        # and the next hidden row lies beyond all of them
        live = [ln for ln in ax.lines
                if getattr(ln, '_hyp_forecast_role', None) == 'live'
                and ln.get_visible() and len(ln.get_xdata())]
        reach = max([np.max(np.asarray(ln.get_xdata(), dtype=float))
                     for ln in live], default=-np.inf)
        seen.append((rows, reach))
        assert 'truth' in _mpl_legend(ax), f'truth legend gone on frame {k}'
    # the held-out rows' x (date numbers), read off the final frame's
    # truth markers once everything is revealed (the first is the seam)
    marker = [ln for ln in ax.lines
              if getattr(ln, '_hyp_forecast_role', None) == 'truth'
              and ln.get_linestyle() == 'None'][0]
    x_truth = np.asarray(marker.get_xdata(), dtype=float)[1:]
    assert len(x_truth) == 3
    best = -np.inf
    for rows, reach in seen:
        best = max(best, reach)
        assert rows == int(np.sum(x_truth <= best + 1e-9))
    assert seen[0][0] == 0, 'the truth is on screen before any forecast'
    assert seen[-1][0] == 3, 'the full truth is not shown at the end'
    plt.close(anim.figure)


@pytest.mark.parametrize('mode', [True, 'serial', 'window'])
def test_truth_reveal_is_monotone_and_matches_across_backends(mode):
    pytest.importorskip('plotly')
    x, held = _long_horizon()
    kw = dict(reduce=None, predict='Kalman', t=12, truth=held, legend=True,
              duration=2, frame_rate=5, antialias=False, show=False,
              animate=mode)
    anim = hyp.plot(x, **kw)
    ax = anim.figure.axes[0]
    mpl = []
    for k in range(anim.n_frames):
        anim.draw_frame(k)
        mpl.append(_mpl_truth_rows(ax)[0])
        assert 'truth' in _mpl_legend(ax)
    plt.close(anim.figure)
    assert mpl[0] == 0
    assert mpl == sorted(mpl), f'a revealed truth row disappeared: {mpl}'
    assert mpl[-1] == 12
    # revealed a few rows at a time, not all at once on the last frame
    assert len(set(mpl)) > 3, mpl

    fig = hyp.plot(x, backend='plotly', **kw)
    # antialias=False: one vertex per row, seam included
    pl = [max(_pl_truth_points(fig, k)[0] - 1, 0)
          for k in range(len(fig.frames))]
    assert pl == mpl
    # the base trace -- what shows before playback -- is frame 0's state
    base = [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == 'truth'][0]
    assert len(base.x) == 0
    # one 'truth' legend key, data-free, so it never flickers
    keys = [tr for tr in fig.data if tr.name == 'truth' and tr.showlegend]
    assert len(keys) == 1
    assert (keys[0].meta or {}).get('hyp_legend_entry') == 'truth'


def test_serial_3d_truths_are_revealed_dataset_by_dataset():
    rng = np.random.default_rng(0)
    ds = [np.cumsum(rng.standard_normal((30, 3)), 0) for _ in range(2)]
    tr = [d[-1] + np.cumsum(rng.standard_normal((4, 3)), 0) for d in ds]
    anim = hyp.plot(ds, reduce=None, predict='Kalman', t=4, truth=tr,
                    legend=True, animate='serial', duration=2, frame_rate=5,
                    show=False)
    ax = anim.figure.axes[0]
    rows = []
    for k in range(anim.n_frames):
        anim.draw_frame(k)
        rows.append(_mpl_truth_rows(ax))
    plt.close(anim.figure)
    assert rows[0] == {0: 0, 1: 0}
    # dataset 0 finishes (and shows its truth) while dataset 1 is still
    # being revealed, so dataset 1's truth is still hidden
    assert any(r[0] == 4 and r[1] == 0 for r in rows), rows
    assert rows[-1] == {0: 4, 1: 4}


def test_regrouped_hue_animation_reveals_truth_progressively():
    pytest.importorskip('plotly')
    x, held = _long_horizon()
    kw = dict(reduce=None, predict='Kalman', t=12, truth=held,
              hue=['a'] * 10 + ['b'] * 10, legend=True, duration=2,
              frame_rate=5, antialias=False, animate=True, show=False)
    anim = hyp.plot(x, **kw)
    ax = anim.figure.axes[0]
    mpl = []
    for k in range(anim.n_frames):
        anim.draw_frame(k)
        mpl.append(_mpl_truth_rows(ax)[0])
    plt.close(anim.figure)
    assert mpl[0] == 0 and mpl[-1] == 12 and mpl == sorted(mpl)
    fig = hyp.plot(x, backend='plotly', **kw)
    pl = [max(_pl_truth_points(fig, k)[0] - 1, 0)
          for k in range(len(fig.frames))]
    assert pl == mpl


def test_spin_and_static_still_draw_the_truth_in_full():
    x, held = _long_horizon()
    for extra in ({}, {'animate': 'spin', 'duration': 1, 'frame_rate': 3}):
        out = hyp.plot(x, reduce=None, predict='Kalman', t=12, truth=held,
                       antialias=False, show=False, **extra)
        fig = out.figure if extra else out
        if extra:
            out.draw_frame(0)
        assert _mpl_truth_rows(fig.axes[0])[0] == 12
        plt.close(fig)


def test_no_drawn_forecast_reveals_no_truth():
    """`truth_reveal_counts` with no schedule (nothing forecast) keeps every
    truth hidden on every frame."""
    assert truth_reveal_counts(None, 2, [3, 3], 4) == [[0, 0]] * 4
