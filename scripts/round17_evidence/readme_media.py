"""Round17 Task 18 evidence: regenerates every image/GIF embedded in
README.md (GH #277) with current 1.0 code.

README.md predates the 1.0 rewrite (media from 2016-2017, code samples
using retired kwargs like `group=`/`n_clusters=` on `hyp.plot`, and
`hyp.tools.describe`). This script re-renders every embedded asset with
the current API, in place, at the same filenames the README already
references (so README.md and docs/index.rst need no path changes) --
plus one new-in-1.0 asset (`images/surface_example.png`, a hull-surface
render) not present in the original README.

Run from the repo root (needs network the first time, to populate the
`hyp.load` cache under `~/hypertools_data`; subsequent runs are fast):

    MPLBACKEND=Agg .venv/bin/python scripts/round17_evidence/readme_media.py

Outputs (images/, repo root):
    hypertools.gif       -- hero: hyperaligned story-trajectories-style
                             `animate='window'` demo (new-in-1.0 animate
                             style), weights_sample subjects
    plot.gif              -- classic animated trajectory plot (animate=True)
    align_before.gif      -- two averaged, UNaligned group trajectories
    align_after.gif        -- the same two groups after hyp.align(align='hyper')
    cluster_example.png   -- mixture-model ("soft") clustering (new-in-1.0)
    describe_example.png  -- hyp.describe output
    surface_example.png   -- hull-surface overlay (new-in-1.0)

All GIFs are rendered at a modest figure size / frame count / dpi so the
files stay in the low hundreds of KB to low single-digit MB, matching
(not ballooning) the repo's existing media budget.
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(REPO, 'images')

# modest dpi/figure size keeps GIF file sizes reasonable without needing
# any post-hoc palette optimization step
plt.rcParams['figure.dpi'] = 80
FIGSIZE = [6.0, 4.2]


def _report(path):
    size_kb = os.path.getsize(path) / 1024
    print(f'  wrote {os.path.relpath(path, REPO)} ({size_kb:.0f} KB)')


def hero_gif():
    """images/hypertools.gif -- hyperaligned `animate='window'` demo.

    `animate='window'` (new in 1.0) plays a moving trail of the most
    recent `focused` timepoints per trajectory while slowly spinning the
    camera -- a livelier, more informative hero than a static screenshot.
    """
    print('hero_gif -> images/hypertools.gif')
    data = hyp.load('weights_sample')  # 3 subjects, (300, 100) each
    path = os.path.join(IMG_DIR, 'hypertools.gif')
    hyp.plot(
        data,
        align='hyper',
        animate='window',
        duration=6,
        frame_rate=15,
        focused=25,
        size=FIGSIZE,
        legend=['subject 1', 'subject 2', 'subject 3'],
        title='HyperTools 1.0',
        save_path=path,
        show=False,
    )
    plt.close('all')
    _report(path)


def plot_gif():
    """images/plot.gif -- classic animated trajectory plot (animate=True)."""
    print('plot_gif -> images/plot.gif')
    data = hyp.load('weights_sample')
    path = os.path.join(IMG_DIR, 'plot.gif')
    hyp.plot(
        data,
        animate=True,
        duration=6,
        frame_rate=15,
        size=FIGSIZE,
        title='hyp.plot',
        save_path=path,
        show=False,
    )
    plt.close('all')
    _report(path)


def align_gifs():
    """images/align_before.gif + images/align_after.gif -- hyp.align demo.

    Mirrors examples/plot_align.py: 36 'weights' subjects averaged into
    two groups, plotted before and after `hyp.align(align='hyper')`.
    """
    print('align_gifs -> images/align_before.gif, images/align_after.gif')
    data = hyp.load('weights')  # 36 subjects, (300, 100) each

    group1_before = np.mean(data[:17], axis=0)
    group2_before = np.mean(data[17:], axis=0)
    before_path = os.path.join(IMG_DIR, 'align_before.gif')
    hyp.plot(
        [group1_before, group2_before],
        animate=True,
        duration=6,
        frame_rate=15,
        size=FIGSIZE,
        legend=['group 1', 'group 2'],
        title='BEFORE alignment',
        save_path=before_path,
        show=False,
    )
    plt.close('all')
    _report(before_path)

    aligned = hyp.align(data, align='hyper')
    group1_after = np.mean(aligned[:17], axis=0)
    group2_after = np.mean(aligned[17:], axis=0)
    after_path = os.path.join(IMG_DIR, 'align_after.gif')
    hyp.plot(
        [group1_after, group2_after],
        animate=True,
        duration=6,
        frame_rate=15,
        size=FIGSIZE,
        legend=['group 1', 'group 2'],
        title='AFTER alignment',
        save_path=after_path,
        show=False,
    )
    plt.close('all')
    _report(after_path)


def cluster_png():
    """images/cluster_example.png -- mixture-model soft clustering
    (new-in-1.0): `hyp.plot` colors each point by blending component
    colors according to its GaussianMixture membership weights."""
    print('cluster_png -> images/cluster_example.png')
    rng = np.random.default_rng(42)
    data = np.vstack([rng.standard_normal((150, 5)) + 1.5 * i
                      for i in range(3)])
    path = os.path.join(IMG_DIR, 'cluster_example.png')
    hyp.plot(
        data, 'o',
        cluster='GaussianMixture', n_clusters=3,
        size=FIGSIZE,
        title='Soft clustering (GaussianMixture)',
        save_path=path,
        show=False,
    )
    plt.close('all')
    _report(path)


def describe_png():
    """images/describe_example.png -- hyp.describe output."""
    print('describe_png -> images/describe_example.png')
    data = hyp.load('weights_sample')
    path = os.path.join(IMG_DIR, 'describe_example.png')
    hyp.describe(data, reduce='PCA', max_dims=14, show=True)
    plt.gcf().set_size_inches(*FIGSIZE)
    plt.gcf().savefig(path, dpi=100)
    plt.close('all')
    _report(path)


def surface_png():
    """images/surface_example.png -- hull-surface overlay (new-in-1.0,
    `hyp.plot(..., surface=True)`, GH #109); not in the original README."""
    print('surface_png -> images/surface_example.png')
    rng = np.random.default_rng(0)
    blob_a = rng.standard_normal((250, 3)) * [1.0, 0.6, 0.8]
    blob_b = rng.standard_normal((250, 3)) * [0.7, 1.0, 0.6] + [3.5, 0, 0]
    path = os.path.join(IMG_DIR, 'surface_example.png')
    hyp.plot(
        [blob_a, blob_b], '.',
        surface=True,
        size=FIGSIZE,
        title='Hull surfaces (new in 1.0)',
        save_path=path,
        show=False,
    )
    plt.close('all')
    _report(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    hero_gif()
    plot_gif()
    align_gifs()
    cluster_png()
    describe_png()
    surface_png()
    print('done.')


if __name__ == '__main__':
    main()
