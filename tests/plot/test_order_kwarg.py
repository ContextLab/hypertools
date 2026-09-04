# tests/plot/test_order_kwarg.py
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pytest

import hypertools as hyp

DURATION, FRAME_RATE = 3, 4
PROBE_FRAME = 3        # early: a serial reveal has only dataset 0 started


def _datasets(n=3, rows=40, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _started(result, n=3, frame=PROBE_FRAME):
    """How many of the n head artists have any vertices at `frame`.

    This is the discriminator artist COUNTS cannot provide: measured at
    frame 3 of 12, parallel gives [247, 247, 247] (3 started) and serial
    gives [657, 0, 0] (1 started), while len(lines)+len(collections) is 9
    for BOTH.
    """
    fig, ani = result
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(frame, *ani._args)
    return sum(1 for ln in ax.lines[:n] if len(ln.get_data_3d()[0]) > 0)


def _plot(**kw):
    return hyp.plot(_datasets(), '-', duration=DURATION,
                    frame_rate=FRAME_RATE, show=False, **kw)


# --- the ordering is actually honoured -------------------------------------

def test_default_order_reveals_every_dataset_together():
    assert _started(_plot(animate=True)) == 3


def test_order_serial_reveals_one_dataset_at_a_time():
    assert _started(_plot(animate=True, order='serial')) == 1


def test_explicit_order_parallel_matches_the_default():
    assert _started(_plot(animate=True, order='parallel')) == 3


def test_order_serial_matches_the_legacy_animate_serial_alias():
    assert (_started(_plot(animate=True, order='serial'))
            == _started(_plot(animate='serial')) == 1)


def test_order_serial_composes_with_chemtrails():
    """Trail artists appear AND the reveal stays serial -- artist counts
    alone cannot tell serial+chemtrails (12) from parallel+chemtrails (12)."""
    fig, ani = _plot(animate=True, order='serial', chemtrails=True)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(PROBE_FRAME, *ani._args)
    heads = [len(ln.get_data_3d()[0]) for ln in ax.lines[:3]]
    trails = [len(ln.get_data_3d()[0]) for ln in ax.lines[3:6]]
    assert sum(1 for h in heads if h) == 1, 'reveal must stay serial'
    assert sum(1 for t in trails if t) == 1, 'the revealing dataset trails'


def test_order_serial_matches_animate_serial_for_hue_overlays():
    """plot.py:4379 passes style=animate into _apply_multicolor_animation,
    which branches on `style == 'serial'` at plot.py:5258 to recover the
    reveal position. A one-site backend_mode substitution would desync."""
    ds = _datasets()
    hue = np.linspace(0.0, 1.0, sum(d.shape[0] for d in ds))

    def segments(**kw):
        fig, ani = hyp.plot(ds, '-', hue=hue, duration=DURATION,
                            frame_rate=FRAME_RATE, show=False, **kw)
        ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
        ani._func(PROBE_FRAME, *ani._args)
        return [len(c._segments3d) for c in ax.collections
                if c.get_label() == '_nolegend_']

    assert segments(animate=True, order='serial') == segments(animate='serial')


# --- conflicts and errors ---------------------------------------------------

def test_conflicting_order_parallel_with_animate_serial_raises():
    with pytest.raises(ValueError, match="animate='serial'"):
        _plot(animate='serial', order='parallel')


def test_conflicting_order_parallel_with_animate_morph_raises():
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(150, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(ValueError, match='inherently serial'):
        hyp.plot(clouds, '.', animate='morph', order='parallel',
                 duration=1, frame_rate=2, show=False)


def test_order_serial_without_animation_raises():
    with pytest.raises(ValueError, match="order='serial' requires an animated"):
        hyp.plot(_datasets(), '-', order='serial', show=False)


@pytest.mark.parametrize('bad', ['Serial', 'sequential', True, 1])
def test_invalid_order_raises(bad):
    with pytest.raises(ValueError, match="order must be"):
        _plot(animate=True, order=bad)


def test_numeric_order_still_offers_the_zorder_hint():
    """Before order= existed: TypeError "...did you mean 'zorder'?".
    That hint must survive the parameter's promotion (review G6)."""
    with pytest.raises(ValueError, match='zorder'):
        _plot(animate=True, order=3)


# --- styles with no serial analog ------------------------------------------

@pytest.mark.parametrize('style', ['spin', 'window'])
def test_serial_ordering_warns_and_is_ignored_for_spin_and_window(style):
    """Matches the established convention at plot.py:3760-3781 (warn, do not
    hard-error, when a flag has no meaning in the requested mode)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = _plot(animate=style, order='serial')
    assert [w for w in caught
            if "order='serial'" in str(w.message) and style in str(w.message)]
    if style == 'window':
        assert _started(result) == 3, 'ordering ignored: still parallel-ish'


# --- C5: the list form of animate= -----------------------------------------

def test_per_dataset_morph_list_accepts_order_serial():
    """animate=['morph', None, 'morph'] resolves to 'morph' (plot.py:480-505),
    which IS serial-capable. Gating on the RAW argument would raise here."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(150, 3)) + off for off in (0.0, 4.0, 8.0)]
    hyp.plot(clouds, '.', animate=['morph', None, 'morph'], order='serial',
             morph_samples=150, duration=1, frame_rate=2, show=False)


def test_morph_accepts_order_serial():
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(150, 3)) + off for off in (0.0, 4.0)]
    hyp.plot(clouds, '.', animate='morph', order='serial',
             morph_samples=150, duration=1, frame_rate=2, show=False)


# --- backend parity ---------------------------------------------------------

def test_order_serial_is_identical_on_plotly():
    """Maintainer requirement: the same call must mean the same thing."""
    pytest.importorskip('plotly')
    fig, ani = _plot(animate=True, order='serial')
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(PROBE_FRAME, *ani._args)
    mpl = [len(ln.get_data_3d()[0]) for ln in ax.lines[:3]]

    hyp.set_interactive_backend('plotly')
    try:
        pfig = _plot(animate=True, order='serial')
    finally:
        hyp.set_interactive_backend('matplotlib')
    ply = [0 if t.x is None else len(t.x)
           for t in pfig.frames[PROBE_FRAME].data][:3]
    assert ply == mpl


def test_order_serial_with_chemtrails_is_identical_on_plotly():
    pytest.importorskip('plotly')
    fig, ani = _plot(animate=True, order='serial', chemtrails=True)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(PROBE_FRAME, *ani._args)
    mpl = [len(ln.get_data_3d()[0]) for ln in ax.lines[:6]]

    hyp.set_interactive_backend('plotly')
    try:
        pfig = _plot(animate=True, order='serial', chemtrails=True)
    finally:
        hyp.set_interactive_backend('matplotlib')
    ply = [0 if t.x is None else len(t.x)
           for t in pfig.frames[PROBE_FRAME].data][:6]
    assert ply == mpl
