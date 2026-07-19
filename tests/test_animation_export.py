# -*- coding: utf-8 -*-
"""Animation export tests: real files, verified frame counts."""

import os
import shutil

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import hypertools as hyp


walk = np.cumsum(np.random.default_rng(0).standard_normal((50, 5)), axis=0)
overlap = np.vstack([np.random.default_rng(42).standard_normal((100, 5))
                     + 1.5 * i for i in range(3)])

HAS_FFMPEG = shutil.which('ffmpeg') is not None


def _animated_frames(path):
    with Image.open(path) as im:
        return getattr(im, 'n_frames', 1)


def _gif_playback(path):
    """(n_frames, total_playback_seconds) read from an exported gif's
    per-frame delays -- the real-time duration a viewer experiences."""
    with Image.open(path) as im:
        n = getattr(im, 'n_frames', 1)
        total_ms = 0
        for i in range(n):
            im.seek(i)
            total_ms += im.info.get('duration', 0)
    return n, total_ms / 1000.0


def test_matplotlib_gif_export(tmp_path):
    out = str(tmp_path / 'anim.gif')
    hyp.plot(walk, animate=True, duration=2, frame_rate=10,
             save_path=out, show=False)
    plt.close('all')
    assert os.path.getsize(out) > 0
    assert _animated_frames(out) > 1


def test_matplotlib_apng_export(tmp_path):
    out = str(tmp_path / 'anim.png')
    hyp.plot(walk, animate=True, duration=2, frame_rate=10,
             save_path=out, show=False)
    plt.close('all')
    assert _animated_frames(out) > 1  # animated PNG, not a static frame


@pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg not installed')
def test_matplotlib_mp4_export(tmp_path):
    out = str(tmp_path / 'anim.mp4')
    hyp.plot(walk, animate=True, duration=2, frame_rate=10,
             save_path=out, show=False)
    plt.close('all')
    assert os.path.getsize(out) > 1000


def test_plotly_gif_export(tmp_path):
    out = str(tmp_path / 'anim.gif')
    hyp.plot(walk, animate=True, duration=2, backend='plotly',
             save_path=out, show=False)
    assert _animated_frames(out) > 1


def test_plotly_spin_gif_export(tmp_path):
    out = str(tmp_path / 'spin.gif')
    hyp.plot(walk, animate='spin', duration=2, backend='plotly',
             save_path=out, show=False)
    assert _animated_frames(out) > 1


@pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg not installed')
def test_plotly_mp4_export(tmp_path):
    out = str(tmp_path / 'anim.mp4')
    hyp.plot(walk, animate=True, duration=2, backend='plotly',
             save_path=out, show=False)
    assert os.path.getsize(out) > 1000


# --- exported gifs must preserve real-time playback duration ---------------
# Regression guard for the "exported gifs play ~6x too fast" bug: an exported
# gif must contain the FULL frame set (frame_rate * duration frames) at the
# true per-frame delay (1000 / frame_rate ms), so total playback ~= duration
# seconds. frame_rate=10 -> 100 ms/frame, which the gif format's 10 ms
# (centisecond) delay granularity stores exactly, keeping the assertion tight.

def test_matplotlib_spin_gif_preserves_realtime_duration(tmp_path):
    duration, frame_rate = 2, 10
    out = str(tmp_path / 'spin.gif')
    hyp.plot(walk, animate='spin', duration=duration, frame_rate=frame_rate,
             save_path=out, show=False)
    plt.close('all')
    n, total = _gif_playback(out)
    assert n == duration * frame_rate       # full frame set, not subsampled
    assert abs(total - duration) <= 0.25 * duration   # ~real-time, not 6x fast


def test_plotly_spin_gif_preserves_realtime_duration(tmp_path):
    duration, frame_rate = 2, 10
    out = str(tmp_path / 'spin.gif')
    hyp.plot(walk, animate='spin', duration=duration, frame_rate=frame_rate,
             backend='plotly', save_path=out, show=False)
    n, total = _gif_playback(out)
    assert n == duration * frame_rate       # full frame set, not subsampled
    assert abs(total - duration) <= 0.25 * duration   # ~real-time, not 6x fast


# --- animated legends must not duplicate in-focus entries with their trails ---
# Regression guard for "each label appears twice": animated line plots draw a
# faint alpha=0.3 trail artist per dataset in addition to the in-focus window.
# Both used to carry the dataset's label, so `ax.legend()` collected each label
# twice (once for the window, once for the tail). Only the in-focus lines should
# appear in the legend.

def _animated_legend_labels(fig):
    lg = fig.axes[0].get_legend()
    assert lg is not None, 'no legend was drawn'
    return [t.get_text() for t in lg.get_texts()]


@pytest.mark.parametrize('extra', [{}, {'chemtrails': True}])
def test_animation_legend_has_no_duplicate_trail_entries(extra):
    a = np.cumsum(np.random.default_rng(1).standard_normal((40, 3)), axis=0)
    b = np.cumsum(np.random.default_rng(2).standard_normal((40, 3)), axis=0)
    fig, _ani = hyp.plot([a, b], animate=True, legend=['first', 'second'],
                         show=False, **extra)
    labels = _animated_legend_labels(fig)
    plt.close('all')
    assert labels == ['first', 'second']  # exactly one entry per dataset


def test_serial_animation_legend_is_static_union():
    """Serial animation brings datasets into focus one at a time, but the
    legend is built once from the upfront line artists, so it shows the
    union of all in-focus datasets and never changes across frames."""
    a = np.cumsum(np.random.default_rng(1).standard_normal((30, 3)), axis=0)
    b = np.cumsum(np.random.default_rng(2).standard_normal((30, 3)), axis=0)
    c = np.cumsum(np.random.default_rng(3).standard_normal((30, 3)), axis=0)
    fig, _ani = hyp.plot([a, b, c], animate='serial', duration=2,
                         legend=['x', 'y', 'z'], show=False)
    labels = _animated_legend_labels(fig)
    plt.close('all')
    assert labels == ['x', 'y', 'z']


def test_mixture_soft_membership_on_overlapping_clusters():
    """Overlapping blobs must yield genuinely MIXED memberships -- the
    mixture demo requirement: a substantial fraction of points belong
    meaningfully to more than one component."""
    props = hyp.cluster(overlap, cluster='GaussianMixture', n_clusters=3)
    mixed = np.mean(props.max(axis=1) < 0.9)
    assert mixed > 0.15, (
        f'only {mixed:.0%} of points show mixed membership; overlap data '
        'should produce substantially soft assignments')
