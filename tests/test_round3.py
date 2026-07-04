# -*- coding: utf-8 -*-
"""Round-3 review items: SVG export, multi-panel figures, hyperalign
n_iter, shapes-zoo datasets, and download-cache hygiene."""

import os

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools.io.load import DATA_DIR


walk = np.cumsum(np.random.default_rng(0).standard_normal((50, 5)), axis=0)


# ------------------------------------------------------------- SVG export
def test_static_svg_matplotlib(tmp_path):
    out = str(tmp_path / 'plot.svg')
    hyp.plot(walk, save_path=out, show=False)
    plt.close('all')
    content = open(out).read()
    assert '<svg' in content and '<animate' not in content


def test_static_svg_plotly(tmp_path):
    out = str(tmp_path / 'plot.svg')
    hyp.plot(walk, backend='plotly', save_path=out, show=False)
    content = open(out).read()
    assert '<svg' in content and '<animate' not in content


def test_animated_svg_matplotlib(tmp_path):
    out = str(tmp_path / 'anim.svg')
    hyp.plot(walk, animate=True, duration=2, frame_rate=10,
             save_path=out, show=False)
    plt.close('all')
    content = open(out).read()
    # SMIL animation with multiple distinct frames
    assert content.count('<animate ') > 5
    assert 'calcMode="discrete"' in content
    assert 'repeatCount="indefinite"' in content


def test_animated_svg_plotly(tmp_path):
    out = str(tmp_path / 'anim.svg')
    hyp.plot(walk, animate='spin', duration=2, backend='plotly',
             save_path=out, show=False)
    content = open(out).read()
    assert content.count('<animate ') > 5


# ------------------------------------------------------------ multi-panel
def test_multipanel_ax_argument():
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    hyp.plot(walk, ax=ax1, show=False)
    hyp.plot(walk, ndims=2, ax=ax2, show=False)
    assert len(ax1.lines) > 0 or len(ax1.collections) > 0
    assert len(ax2.lines) > 0
    # both hypertools panels live in the SAME user figure
    assert ax1.figure is fig and ax2.figure is fig
    plt.close('all')


# -------------------------------------------------------- hyperalign n_iter
def test_hyperalign_n_iter_flag():
    d1 = np.cumsum(np.random.default_rng(1).standard_normal((60, 4)), axis=0)
    d2 = d1 @ np.linalg.qr(
        np.random.default_rng(2).standard_normal((4, 4)))[0]
    one = hyp.align([d1, d2], n_iter=1)
    ten = hyp.align([d1, d2])  # default n_iter=10

    def mean_corr(pair):
        a, b = pair
        return np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1]
                        for i in range(a.shape[1])])

    assert mean_corr(ten) >= mean_corr(one) - 1e-6
    # dict form threads n_iter and returns aligned data (not None)
    via_dict = hyp.align([d1, d2],
                         align={'model': 'hyper', 'params': {'n_iter': 3}})
    assert via_dict is not None and len(via_dict) == 2


# ------------------------------------------------------------- shapes zoo
def test_load_shapes_zoo_teapot():
    teapot = hyp.load('teapot')  # smallest zoo member (~42KB download)
    assert teapot.shape[1] == 3 and teapot.shape[0] > 1000
    fig = hyp.plot(teapot, 'o', show=False)
    assert fig is not None
    plt.close('all')


def test_load_datasaurus():
    dozen = hyp.load('datasaurus')
    assert isinstance(dozen, list) and len(dozen) == 13


# ------------------------------------------------- download-cache hygiene
def test_reload_does_not_duplicate_or_redownload():
    hyp.load('spiral')
    before = sorted(os.listdir(DATA_DIR))
    mtime = (DATA_DIR / 'spiral').stat().st_mtime
    for _ in range(3):
        hyp.load('spiral')
    assert sorted(os.listdir(DATA_DIR)) == before
    assert (DATA_DIR / 'spiral').stat().st_mtime == mtime
