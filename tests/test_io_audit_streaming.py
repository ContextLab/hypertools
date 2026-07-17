# -*- coding: utf-8 -*-
"""Regression tests for the 2026-07 release audit findings on streaming
plots (unit F22-io-streaming-lsl, hypertools/io/streaming.py side). Real
generators, real figures (Agg), real GIF files -- no mocks."""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import hypertools as hyp


def walk_gen(n=300, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((3, dim))
    p = np.zeros(3)
    for _ in range(n):
        p = p + 0.1 * rng.standard_normal(3)
        yield p @ W


# --------------------------------------------------------------- F22-001
# single-channel streams used to crash with a bare IndexError at the first
# post-head redraw.

def test_single_channel_stream_raises_clear_error():
    def g():
        for i in range(100):
            yield float(np.sin(i / 8))

    with pytest.raises(ValueError, match='single feature'):
        hyp.plot(g(), stream_init=30, stream_chunk=20, show=False)
    plt.close('all')


def test_single_channel_head_only_stream_still_plots():
    # a 1-channel stream that ends within the head keeps working (it is
    # drawn like a static 1-D plot)
    def g():
        for i in range(20):
            yield float(np.sin(i / 8))

    fig = hyp.plot(g(), stream_init=30, stream_chunk=20, show=False)
    assert fig.stream_info['n_samples'] == 20
    plt.close('all')


# --------------------------------------------------------------- F22-003
# a mid-stream error must not lose the figure, the consumed data, and the
# animation file.

def test_midstream_error_returns_partial_results(tmp_path):
    out = tmp_path / 'partial.gif'

    def dying():
        for i in range(200):
            if i == 120:
                raise RuntimeError('sensor unplugged')
            yield [float(i), float(np.sin(i / 5.)), float(np.cos(i / 3.))]

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(dying(), stream_init=50, stream_chunk=25,
                       save_path=str(out), show=False)
    assert fig is not None
    assert fig.stream_info['n_samples'] == 120
    assert fig.stream_info['truncated']
    assert isinstance(fig.stream_info['error'], RuntimeError)
    assert fig.stream_info['data'][0].shape == (120, 3)
    assert out.exists(), 'animation file was lost on a mid-stream error'
    with Image.open(out) as im:
        assert im.n_frames >= 2, 'animation was not finalized'
    plt.close('all')


def test_midstream_nan_returns_partial_results():
    def nan_stream():
        rng = np.random.default_rng(3)
        for i in range(200):
            row = rng.standard_normal(5)
            if i == 120:
                row[0] = np.nan
            yield row

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(nan_stream(), stream_init=50, stream_chunk=25,
                       show=False)
    assert fig.stream_info['n_samples'] >= 100
    assert fig.stream_info['error'] is not None
    plt.close('all')


# --------------------------------------------------------------- F22-006
# stream_* parameters must be validated with messages naming the parameter.

@pytest.mark.parametrize('kwargs,param', [
    (dict(stream_init=0), 'stream_init'),
    (dict(stream_init=-5), 'stream_init'),
    (dict(stream_chunk=0), 'stream_chunk'),
    (dict(stream_chunk=-10), 'stream_chunk'),
    (dict(stream_window=0), 'stream_window'),
    (dict(stream_window=-50), 'stream_window'),
    (dict(stream_max=0), 'stream_max'),
])
def test_stream_parameter_validation(kwargs, param):
    base = dict(stream_init=30, stream_chunk=20, show=False)
    base.update(kwargs)
    with pytest.raises(ValueError, match=param):
        hyp.plot(walk_gen(100), **base)
    plt.close('all')


# --------------------------------------------------------------- F22-007
# stream_max smaller than stream_init must cap the head too.

def test_stream_max_caps_head_consumption():
    consumed = {'n': 0}

    def counting(n=1000):
        rng = np.random.default_rng(4)
        for _ in range(n):
            consumed['n'] += 1
            yield rng.standard_normal(3)

    fig = hyp.plot(counting(), stream_init=200, stream_max=100,
                   stream_chunk=50, show=False)
    assert fig.stream_info['n_samples'] == 100
    assert fig.stream_info['truncated']
    # head + the truncation peek must not overshoot the documented cap
    assert consumed['n'] <= 101
    plt.close('all')


# --------------------------------------------------------------- F22-008
# unplottable dimensionality must fail clearly BEFORE consuming samples.

def test_stream_ndims_validated_before_consuming():
    consumed = {'n': 0}

    def counting(n=100):
        rng = np.random.default_rng(5)
        for _ in range(n):
            consumed['n'] += 1
            yield rng.standard_normal(6)

    with pytest.raises(ValueError, match='ndims'):
        hyp.plot(counting(), stream_init=40, stream_chunk=20, ndims=5,
                 show=False)
    assert consumed['n'] == 0, 'samples were consumed before validation'
    plt.close('all')


def test_stream_reduce_none_highdim_clear_error():
    with pytest.raises(ValueError, match='reduce'):
        hyp.plot(walk_gen(100, dim=4), stream_init=40, stream_chunk=20,
                 reduce=None, show=False)
    plt.close('all')


# --------------------------------------------------------------- F22-002
# heavy clamping against the frozen head box must produce a runtime
# warning (the display is distorted; true values live on stream_info).

def test_clamp_warning_when_stream_leaves_head_box():
    def shifting(n=150):
        rng = np.random.default_rng(6)
        for i in range(n):
            base = 0.0 if i < 50 else 10.0   # distribution shift post-head
            yield base + 0.1 * rng.standard_normal(3)

    with pytest.warns(RuntimeWarning, match='clamped'):
        fig = hyp.plot(shifting(), stream_init=50, stream_chunk=25,
                       show=False)
    # the true (unclamped) projections are retained
    assert np.abs(fig.stream_info['xform_data'][0][50:]).max() > 1.0
    plt.close('all')


def test_no_clamp_warning_for_stationary_stream():
    import warnings as _warnings

    def stationary(n=150):
        rng = np.random.default_rng(7)
        for _ in range(n):
            yield rng.standard_normal(3)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        hyp.plot(stationary(), stream_init=100, stream_chunk=25, show=False)
    assert not [w for w in caught if 'clamped' in str(w.message)]
    plt.close('all')


# --------------------------------------------------------------- F22-005
# streamed trajectories are drawn as raw polylines from the very first
# frame (the head used to be drawn smoothed/interpolated, snapping to a
# raw polyline on the first redraw).

def test_head_frame_drawn_raw_not_interpolated(tmp_path):
    out = tmp_path / 'consistent.gif'
    fig = hyp.plot(walk_gen(60, seed=11), stream_init=30, stream_chunk=30,
                   save_path=str(out), show=False)
    fig.canvas.draw()
    line = next(ln for ln in fig.axes[0].lines
                if len(ln.get_data_3d()[0]))
    # raw polyline: one vertex per consumed sample (not ~15x interpolated)
    assert len(line.get_data_3d()[0]) == 60
    plt.close('all')
