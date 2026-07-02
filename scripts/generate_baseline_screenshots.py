"""Generate baseline screenshots of hypertools v0.8.2 behavior.

Run from the repo root with the project venv:
    .venv/bin/python scripts/generate_baseline_screenshots.py

These baselines document CURRENT behavior on master/v0.8.2 across the core
use-case matrix, so the dev-2.0 modernization can be visually diffed against
them (aesthetic parity was the main unresolved gap in the earlier
matplotlib-backend attempt — see notes/hypertools_2.0_roadmap.md).

Outputs: tests/screenshots/baseline_v0.8.2/<function>/<case>.png
(directory is gitignored; screenshots are reviewed locally / uploaded as CI
artifacts, not committed).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from screenshot_harness import capture, summarize

import hypertools as hyp

TAG = 'baseline_v0.8.2'
SEED = 42


def make_walk(n=200, d=10, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, d)), axis=0)


def make_clusters(n_per=60, d=5, k=3, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.standard_normal((n_per, d)) + 6 * i for i in range(k)])


def main():
    walk = make_walk()
    walk2 = make_walk(seed=1)
    walk3 = make_walk(seed=2)
    clusters = make_clusters()
    df = pd.DataFrame(walk, columns=[f'f{i}' for i in range(walk.shape[1])])
    labels = ([f'group{i}' for i in range(3) for _ in range(60)])

    cases = [
        # (function, case, thunk)
        ('plot', 'line_3d_single_array',
         lambda: hyp.plot(walk, show=False)),
        ('plot', 'line_3d_list_of_arrays',
         lambda: hyp.plot([walk, walk2, walk3], show=False)),
        ('plot', 'scatter_3d_fmt_o',
         lambda: hyp.plot(clusters, 'o', show=False)),
        ('plot', 'line_2d_ndims2',
         lambda: hyp.plot(walk, ndims=2, show=False)),
        ('plot', 'fmt_dashed_linestyle',
         lambda: hyp.plot(walk, '--', show=False)),
        ('plot', 'dataframe_input',
         lambda: hyp.plot(df, show=False)),
        ('plot', 'hue_grouped_scatter',
         lambda: hyp.plot(clusters, 'o', hue=labels, show=False)),
        ('plot', 'cluster_kmeans_3',
         lambda: hyp.plot(clusters, 'o', cluster='KMeans', n_clusters=3,
                          show=False)),
        ('plot', 'reduce_tsne',
         lambda: hyp.plot(clusters, 'o', reduce='TSNE', show=False)),
        ('plot', 'legend_and_labels',
         lambda: hyp.plot([walk, walk2], legend=['A', 'B'],
                          title='baseline legend test', show=False)),
        ('plot', 'align_hyperalignment',
         lambda: hyp.plot([walk, walk + 0.5], align='hyper', show=False)),
        ('plot', 'missing_data_ppca',
         lambda: hyp.plot(_with_missing(walk), show=False)),
        # NOTE: describe(show=False) skips figure CREATION entirely (unlike
        # plot(show=False), which builds but doesn't display). show=True is
        # safe headless because plt.show() is a no-op under Agg. The
        # inconsistency is a 2.0 API cleanup item.
        ('describe', 'default',
         lambda: hyp.describe(walk, show=True)),
    ]

    records = [capture(TAG, fn, case, thunk) for fn, case, thunk in cases]
    failures = summarize(records)
    sys.exit(1 if failures else 0)


def _with_missing(x, frac=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    x = x.copy()
    mask = rng.random(x.shape) < frac
    x[mask] = np.nan
    return x


if __name__ == '__main__':
    main()
