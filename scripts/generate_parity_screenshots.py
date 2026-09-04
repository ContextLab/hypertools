"""Backend visual-parity matrix: render identical cases on the matplotlib
AND plotly backends and save side-by-side montages for direct comparison.

Line/marker styles, sizing, colors, and the signature cube/square framing
must match across backends (per Jeremy's requirement on PR #270).

Run from the repo root:
    .venv/bin/python scripts/generate_parity_screenshots.py

Outputs: tests/screenshots/parity_v1.0/<case>.png  (matplotlib | plotly)
plus an INDEX.md manifest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from screenshot_harness import SCREENSHOT_ROOT
import hypertools as hyp

TAG = 'parity_v1.0'
OUT_DIR = os.path.join(SCREENSHOT_ROOT, TAG)
SEED = 42
SIZE = [7, 5]  # inches; plotly renders at size*100 px, mpl saved at dpi=100


def make_walk(n=200, d=10, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, d)), axis=0)


def make_overlapping_clusters(n_per=100, d=5, k=3, sep=1.5, seed=SEED):
    """Blobs sep standard deviations apart: overlap regions have genuinely
    mixed memberships, so mixture-model color blending is visible."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.standard_normal((n_per, d)) + sep * i
                      for i in range(k)])


def make_clusters(n_per=60, d=5, k=3, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.standard_normal((n_per, d)) + 6 * i for i in range(k)])


def render_both(case, plot_kwargs_fn):
    """Render one case on both backends; save side-by-side montage."""
    os.makedirs(OUT_DIR, exist_ok=True)
    mpl_path = os.path.join(OUT_DIR, f'_{case}_mpl.png')
    ply_path = os.path.join(OUT_DIR, f'_{case}_plotly.png')
    out_path = os.path.join(OUT_DIR, f'{case}.png')
    record = {'case': case, 'path': out_path, 'ok': False, 'error': None}
    try:
        args, kwargs = plot_kwargs_fn()

        geo = hyp.plot(*args, backend='matplotlib', show=False,
                       size=SIZE, **kwargs)
        geo.fig.savefig(mpl_path, dpi=100, bbox_inches=None,
                        facecolor='white')
        plt.close('all')

        geo = hyp.plot(*args, backend='plotly', show=False,
                       size=SIZE, **kwargs)
        geo.fig.write_image(ply_path, width=SIZE[0] * 100,
                            height=SIZE[1] * 100)

        left, right = Image.open(mpl_path), Image.open(ply_path)
        h = max(left.height, right.height)
        montage = Image.new('RGB', (left.width + right.width + 8, h),
                            'white')
        montage.paste(left, (0, 0))
        montage.paste(right, (left.width + 8, 0))
        montage.save(out_path)
        os.remove(mpl_path)
        os.remove(ply_path)
        record['ok'] = True
    except Exception as e:  # noqa: BLE001
        record['error'] = f'{type(e).__name__}: {e}'
    finally:
        plt.close('all')
    return record


def main():
    walk = make_walk()
    walk2 = make_walk(seed=1)
    walk3 = make_walk(seed=2)
    clusters = make_clusters()
    oclusters = make_overlapping_clusters()
    labels = [f'group{i}' for i in range(3) for _ in range(60)]

    cases = [
        ('line_3d_single', lambda: ((walk,), {})),
        ('line_3d_list', lambda: (([walk, walk2, walk3],), {})),
        ('scatter_3d_markers', lambda: ((clusters, 'o'), {})),
        ('lines_plus_markers', lambda: ((walk, '.-'), {})),
        ('dashed_lines', lambda: ((walk, '--'), {})),
        ('dotted_lines', lambda: ((walk, ':'), {})),
        ('dashdot_lines', lambda: ((walk, '-.'), {})),
        ('marker_square', lambda: ((clusters, 's'), {})),
        ('marker_triangle', lambda: ((clusters, '^'), {})),
        ('line_2d', lambda: ((walk,), {'ndims': 2})),
        ('scatter_2d', lambda: ((clusters, 'o'), {'ndims': 2})),
        ('legend_two_groups',
         lambda: (([walk, walk2],), {'legend': ['A', 'B']})),
        ('title', lambda: ((walk,), {'title': 'parity check'})),
        ('hue_categorical',
         lambda: ((clusters, 'o'), {'hue': labels})),
        ('hue_continuous',
         lambda: ((walk, 'o'),
                  {'hue': np.arange(len(walk), dtype=float)})),
        # NOTE: cluster fits are precomputed ONCE and passed as hue so both
        # backends receive identical assignments -- re-fitting per backend
        # permutes component order (fit nondeterminism, not backend skew)
        ('cluster_kmeans',
         lambda: ((clusters, 'o'),
                  {'hue': [str(lab) for lab in
                           hyp.cluster(clusters, n_clusters=3)]})),
        ('cluster_hdbscan',
         lambda: ((clusters, 'o'),
                  {'hue': [str(lab) for lab in
                           hyp.cluster(clusters, cluster='HDBSCAN')]})),
        ('mixture_gaussian',
         lambda: ((oclusters, 'o'),
                  {'hue': hyp.cluster(oclusters, cluster='GaussianMixture',
                                      n_clusters=3)})),
        ('matrix_hue_blend',
         lambda: ((oclusters, 'o'),
                  {'hue': hyp.cluster(oclusters,
                                      cluster='BayesianGaussianMixture',
                                      n_clusters=3)})),
        ('nested_multilevel', lambda: (([[walk, walk2], [walk3]],), {})),
        ('nested_mixed_depth', lambda: (([[walk, [walk2]], walk3],), {})),
        ('multicolored_line',
         lambda: ((walk,), {'hue': np.arange(len(walk), dtype=float)})),
    ]

    records = [render_both(case, fn) for case, fn in cases]

    n_ok = sum(r['ok'] for r in records)
    print(f'\n{"=" * 60}\nParity montages: {n_ok}/{len(records)} rendered')
    for r in records:
        status = 'ok ' if r['ok'] else 'FAIL'
        print(f"  [{status}] {r['case']}"
              + ('' if r['ok'] else f"  -> {r['error']}"))

    index = ['# Backend parity matrix (matplotlib | plotly)', '',
             f'{n_ok}/{len(records)} cases rendered. Each image shows the '
             'matplotlib render (left) and plotly render (right) of an '
             'identical call.', '',
             '| case | status | file |', '|-|-|-|']
    for r in records:
        status = 'pass' if r['ok'] else f"FAIL: {r['error']}"
        index.append(f"| {r['case']} | {status} | {r['case']}.png |")
    with open(os.path.join(OUT_DIR, 'INDEX.md'), 'w') as f:
        f.write('\n'.join(index) + '\n')

    sys.exit(0 if n_ok == len(records) else 1)


if __name__ == '__main__':
    main()
