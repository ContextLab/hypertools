# -*- coding: utf-8 -*-
"""Round-4 review items: animation frame clipping."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

import hypertools as hyp


def test_animation_frames_not_clipped(tmp_path):
    """Zoomed, rotating animations must keep the cube and data fully inside
    the canvas at EVERY rotation angle (no content touching the border)."""
    walk = np.cumsum(np.random.default_rng(0).standard_normal((60, 5)),
                     axis=0)
    out = str(tmp_path / 'spin.gif')
    hyp.plot(walk, animate='spin', zoom=2.5, duration=2, frame_rate=10,
             save_path=out, show=False)
    plt.close('all')

    clipped = []
    with Image.open(out) as im:
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            a = np.asarray(frame.convert('L'))
            border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
            if (border < 200).any():
                clipped.append(i)
    assert not clipped, f'content clipped at frames {clipped}'


def test_animate_serial_sequential_reveal(tmp_path):
    """animate='serial': datasets appear one at a time in list order, each
    growing while earlier ones stay drawn, never connected to each other."""
    rng = np.random.default_rng(0)
    # three well-separated segments so pixel analysis can distinguish them
    sets = [np.cumsum(rng.standard_normal((30, 3)), axis=0) + 8 * i
            for i in range(3)]
    out = str(tmp_path / 'serial.gif')
    geo = hyp.plot(sets, animate='serial', duration=3, frame_rate=10,
                   save_path=out, show=False)
    plt.close('all')

    with Image.open(out) as im:
        frames = [np.asarray(f.convert('L'), float)
                  for f in ImageSequence.Iterator(im)]
    ink = [float((f < 200).sum()) for f in frames]
    # drawn content must grow over the animation (accumulating datasets)
    assert ink[-1] > ink[len(ink) // 2] > ink[2]

    # separate artists per dataset (plus one unused trail artist each):
    # datasets render disconnected, and after the final frame all three
    # data artists are fully revealed
    populated = [ln for ln in geo.ax.lines if len(ln.get_data()[0]) > 1]
    assert len(populated) == 3


def test_animate_serial_plotly():
    rng = np.random.default_rng(0)
    sets = [np.cumsum(rng.standard_normal((30, 3)), axis=0) + 8 * i
            for i in range(3)]
    geo = hyp.plot(sets, animate='serial', duration=3, backend='plotly',
                   show=False)
    assert len(geo.fig.frames) > 0
    # first frame: only the first dataset has begun to appear
    first = geo.fig.frames[1]
    lengths = [len(t.x) if t.x is not None else 0 for t in first.data]
    assert lengths[1] == 0 and lengths[2] == 0
    # final frame: all datasets fully revealed
    last = geo.fig.frames[-1]
    lengths = [len(t.x) if t.x is not None else 0 for t in last.data]
    assert all(n > 0 for n in lengths)


def test_cluster_dict_single_call_syntax():
    """Round-4.5: one-call mixture coloring with top-level n_clusters and
    class specs: hyp.plot(x, '.', cluster={'model': GaussianMixture,
    'n_clusters': k}) colors by proportions automatically."""
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(42)
    overlap = np.vstack([rng.standard_normal((80, 5)) + 1.5 * i
                         for i in range(3)])

    # string form with top-level n_clusters
    props = hyp.cluster(overlap, cluster={'model': 'GaussianMixture',
                                          'n_clusters': 3})
    assert props.shape == (240, 3)

    # class form
    props = hyp.cluster(overlap, cluster={'model': GaussianMixture,
                                          'n_clusters': 3})
    assert props.shape == (240, 3)

    # single plot call: exact per-point blended marker colors
    geo = hyp.plot(overlap, '.', markersize=2,
                   cluster={'model': GaussianMixture, 'n_clusters': 3},
                   show=False)
    scatters = [c for c in geo.ax.collections
                if type(c).__name__.startswith('Path')]
    assert scatters and len(scatters[0].get_facecolors()) > 100
    plt.close('all')
