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


def test_mixture_soft_membership_on_overlapping_clusters():
    """Overlapping blobs must yield genuinely MIXED memberships -- the
    mixture demo requirement: a substantial fraction of points belong
    meaningfully to more than one component."""
    props = hyp.cluster(overlap, cluster='GaussianMixture', n_clusters=3)
    mixed = np.mean(props.max(axis=1) < 0.9)
    assert mixed > 0.15, (
        f'only {mixed:.0%} of points show mixed membership; overlap data '
        'should produce substantially soft assignments')
