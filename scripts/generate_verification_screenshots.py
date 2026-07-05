"""Comprehensive visual verification matrix for hypertools 1.0.

Exercises EVERY public API function across representative use cases and
captures a PNG for each, so correctness can be verified by eye and diffed
against the v0.8.2 baselines. Writes an INDEX.md manifest alongside the
images (used as evidence in the 1.0 PR).

Run from the repo root:
    .venv/bin/python scripts/generate_verification_screenshots.py

Outputs: tests/screenshots/verification_v1.0/<function>/<case>.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from screenshot_harness import capture, summarize, SCREENSHOT_ROOT

import matplotlib.pyplot as plt
import hypertools as hyp

TAG = 'verification_v1.0'
SEED = 42


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


def scatter_result(arrs, title):
    """Visualize a non-plot function's output through hyp.plot for capture."""
    return hyp.plot(arrs, 'o', title=title, show=False)


def main():
    walk = make_walk()
    walk2 = make_walk(seed=1)
    walk3 = make_walk(seed=2)
    clusters = make_clusters()
    oclusters = make_overlapping_clusters()
    df = pd.DataFrame(walk, columns=[f'f{i}' for i in range(walk.shape[1])])
    labels = [f'group{i}' for i in range(3) for _ in range(60)]
    walk_missing = walk.copy()
    walk_missing[np.random.default_rng(SEED).random(walk.shape) < 0.05] = np.nan

    cases = []

    # ------------------------------------------------------------ hyp.plot
    P = 'plot'
    cases += [
        (P, 'line_3d_single_array', lambda: hyp.plot(walk, show=False)),
        (P, 'line_3d_list_of_arrays',
         lambda: hyp.plot([walk, walk2, walk3], show=False)),
        (P, 'scatter_3d_fmt_o', lambda: hyp.plot(clusters, 'o', show=False)),
        (P, 'line_2d_ndims2', lambda: hyp.plot(walk, ndims=2, show=False)),
        (P, 'fmt_dashed_linestyle', lambda: hyp.plot(walk, '--', show=False)),
        (P, 'dataframe_input', lambda: hyp.plot(df, show=False)),
        (P, 'list_of_dataframes',
         lambda: hyp.plot([df, df + 3], show=False)),
        (P, 'hue_categorical',
         lambda: hyp.plot(clusters, 'o', hue=labels, legend=True, show=False)),
        (P, 'hue_continuous',
         lambda: hyp.plot(walk, 'o', hue=np.arange(len(walk), dtype=float),
                          show=False)),
        (P, 'hue_matrix_blended',
         lambda: hyp.plot(oclusters, 'o',
                          hue=hyp.cluster(oclusters,
                                          cluster='GaussianMixture',
                                          n_clusters=3),
                          show=False)),
        (P, 'cluster_kmeans_3',
         lambda: hyp.plot(clusters, 'o', cluster='KMeans', n_clusters=3,
                          show=False)),
        (P, 'cluster_hdbscan',
         lambda: hyp.plot(clusters, 'o', cluster='HDBSCAN', show=False)),
        (P, 'cluster_gaussian_mixture_blend',
         lambda: hyp.plot(oclusters, 'o', cluster='GaussianMixture',
                          n_clusters=3, show=False)),
        (P, 'nested_list_multilevel',
         lambda: hyp.plot([[walk, walk2], [walk3]], show=False)),
        (P, 'nested_mixed_depth_styling',
         lambda: hyp.plot([[walk, [walk2]], walk3], show=False)),
        (P, 'reduce_tsne', lambda: hyp.plot(clusters, 'o', reduce='TSNE',
                                            show=False)),
        (P, 'reduce_umap_lazy',
         lambda: hyp.plot(clusters, 'o', reduce='UMAP', show=False)),
        (P, 'align_hyperalignment',
         lambda: hyp.plot([walk, walk + 0.5], align='hyper', show=False)),
        (P, 'normalize_within',
         lambda: hyp.plot([walk, walk2], normalize='within', show=False)),
        (P, 'missing_data_ppca', lambda: hyp.plot(walk_missing, show=False)),
        (P, 'legend_title_labels',
         lambda: hyp.plot([walk, walk2], legend=['A', 'B'],
                          title='verification: legend/title', show=False)),
        (P, 'animate_frame_capture',
         lambda: hyp.plot(walk, animate=True, duration=5, show=False)),
        (P, 'animate_spin_frame_capture',
         lambda: hyp.plot(walk, animate='spin', duration=5, show=False)),
        (P, 'ndims_gt3_plots_3d',
         lambda: hyp.plot(walk, ndims=6, show=False)),
    ]

    # ------------------------------ multicolored lines + marker/line styles
    cases += [
        (P, 'multicolored_line_continuous',
         lambda: hyp.plot(walk, hue=np.arange(len(walk), dtype=float),
                          show=False)),
        (P, 'multicolored_line_matrix_hue',
         lambda: hyp.plot(walk,
                          hue=np.column_stack([
                              np.linspace(0, 1, len(walk)),
                              np.linspace(1, 0, len(walk))]),
                          show=False)),
        (P, 'multicolored_lines_two_datasets',
         lambda: hyp.plot([walk, walk2],
                          hue=np.arange(2 * len(walk), dtype=float),
                          show=False)),
        (P, 'multicolored_line_2d',
         lambda: hyp.plot(walk, ndims=2,
                          hue=np.arange(len(walk), dtype=float),
                          show=False)),
        (P, 'marker_square', lambda: hyp.plot(clusters, 's', show=False)),
        (P, 'marker_triangle', lambda: hyp.plot(clusters, '^', show=False)),
        (P, 'lines_plus_markers', lambda: hyp.plot(walk, '.-', show=False)),
        (P, 'dotted_lines', lambda: hyp.plot(walk, ':', show=False)),
        (P, 'dashdot_lines', lambda: hyp.plot(walk, '-.', show=False)),
        (P, 'mixture_bayesian_gm',
         lambda: hyp.plot(oclusters, 'o', cluster='BayesianGaussianMixture',
                          n_clusters=3, show=False)),
        (P, 'cluster_spectral',
         lambda: hyp.plot(clusters, 'o', cluster='SpectralClustering',
                          n_clusters=3, show=False)),
    ]

    # -------------------------------------------------- plotly backend
    # every feature above, rendered through the interactive backend
    B = 'plot_backend_plotly'

    def ply(*args, **kw):
        return lambda: hyp.plot(*args, backend='plotly', show=False, **kw)

    cases += [
        (B, 'line_3d', ply(walk)),
        (B, 'line_3d_list', ply([walk, walk2, walk3])),
        (B, 'scatter_3d_groups',
         ply([w[:, :3] for w in (walk, walk2, walk3)], 'o',
             legend=['a', 'b', 'c'])),
        (B, 'line_2d', ply(walk, ndims=2)),
        (B, 'scatter_2d', ply(clusters, 'o', ndims=2)),
        (B, 'dashed_lines', ply(walk, '--')),
        (B, 'dotted_lines', ply(walk, ':')),
        (B, 'dashdot_lines', ply(walk, '-.')),
        (B, 'lines_plus_markers', ply(walk, '.-')),
        (B, 'marker_square', ply(clusters, 's')),
        (B, 'hue_categorical', ply(clusters, 'o', hue=labels)),
        (B, 'hue_continuous',
         ply(walk, 'o', hue=np.arange(len(walk), dtype=float))),
        (B, 'hue_matrix_blended',
         ply(oclusters, 'o',
             hue=hyp.cluster(oclusters, cluster='GaussianMixture',
                             n_clusters=3))),
        (B, 'cluster_kmeans', ply(clusters, 'o', cluster='KMeans',
                                  n_clusters=3)),
        (B, 'cluster_hdbscan', ply(clusters, 'o', cluster='HDBSCAN')),
        (B, 'mixture_blend', ply(oclusters, 'o', cluster='GaussianMixture',
                                 n_clusters=3)),
        (B, 'nested_list_multilevel', ply([[walk, walk2], [walk3]])),
        (B, 'nested_mixed_depth', ply([[walk, [walk2]], walk3])),
        (B, 'multicolored_line',
         ply(walk, hue=np.arange(len(walk), dtype=float))),
        (B, 'multicolored_line_2d',
         ply(walk, ndims=2, hue=np.arange(len(walk), dtype=float))),
        (B, 'legend_title',
         ply([walk, walk2], legend=['A', 'B'], title='plotly legend/title')),
        (B, 'animate_window_firstframe', ply(walk, animate=True)),
        (B, 'animate_spin_firstframe', ply(walk, animate='spin')),
    ]

    # -------------------------------------------------- apply_model
    cases += [
        ('apply_model', 'shared_embedding',
         lambda: scatter_result(
             hyp.apply_model([clusters[:90], clusters[90:]], 'PCA', ndims=3),
             'apply_model: one PCA fit across stacked datasets')),
        ('apply_model', 'pipeline',
         lambda: scatter_result(
             hyp.apply_model(clusters,
                             [{'model': 'PCA', 'params': {'n_components': 4}},
                              {'model': 'TSNE',
                               'params': {'n_components': 2}}]),
             'apply_model: PCA -> TSNE pipeline')),
    ]

    # ------------------------------------------------------------ tools
    cases += [
        ('reduce', 'pca_default',
         lambda: scatter_result(hyp.reduce(clusters, ndims=3),
                                'reduce: IncrementalPCA -> 3D')),
        ('reduce', 'tsne',
         lambda: scatter_result(
             hyp.reduce(clusters, reduce='TSNE', ndims=2),
             'reduce: TSNE -> 2D')),
        ('reduce', 'umap',
         lambda: scatter_result(
             hyp.reduce(clusters, reduce='UMAP', ndims=2),
             'reduce: UMAP -> 2D')),
        ('reduce', 'list_input',
         lambda: scatter_result(hyp.reduce([walk, walk2], ndims=3),
                                'reduce: list input')),
        ('align', 'hyperalignment',
         lambda: scatter_result(hyp.align([walk, walk + 0.5]),
                                'align: hyper')),
        ('align', 'srm',
         lambda: scatter_result(hyp.align([walk, walk + 0.5], align='SRM'),
                                'align: SRM')),
        ('normalize', 'across',
         lambda: scatter_result(hyp.normalize([walk, walk2],
                                              normalize='across'),
                                'normalize: across')),
        ('normalize', 'within',
         lambda: scatter_result(hyp.normalize([walk, walk2],
                                              normalize='within'),
                                'normalize: within')),
        ('cluster', 'kmeans_labels',
         lambda: hyp.plot(clusters, 'o',
                          hue=[str(l) for l in
                               hyp.cluster(clusters, n_clusters=3)],
                          title='cluster: KMeans labels', show=False)),
        ('cluster', 'mixture_proportions',
         lambda: hyp.plot(oclusters, 'o',
                          hue=hyp.cluster(oclusters,
                                          cluster='GaussianMixture',
                                          n_clusters=3),
                          title='cluster: GaussianMixture proportions',
                          show=False)),
        ('analyze', 'norm_reduce_align',
         lambda: scatter_result(
             hyp.analyze([walk, walk + 0.5], normalize='within', ndims=3,
                         align='hyper'),
             'analyze: normalize+reduce+align')),
        ('describe', 'default', lambda: hyp.describe(walk, show=True)),
        ('format_data', 'mixed_inputs',
         lambda: scatter_result(
             hyp.tools.format_data([walk[:, :5], df.iloc[:, :5]]),
             'format_data: array + DataFrame')),
    ]

    # ------------------------------------------- network-dependent cases
    def load_and_plot():
        geo = hyp.load('weights_sample')
        return geo.plot(show=False)

    def text_plot():
        return hyp.plot(['the dog ran fast', 'a cat sat quietly',
                         'dogs and cats are pets', 'the weather is sunny',
                         'it rained all day', 'clouds cover the sky'],
                        'o', title='text input', show=False)

    cases += [
        ('load', 'weights_sample_geo_plot', load_and_plot),
        ('plot', 'text_input', text_plot),
    ]

    records = [capture(TAG, fn, case, thunk) for fn, case, thunk in cases]
    failures = summarize(records)
    _write_index(records)
    sys.exit(1 if failures else 0)


def _write_index(records):
    index_path = os.path.join(SCREENSHOT_ROOT, TAG, 'INDEX.md')
    lines = [
        '# hypertools 1.0 visual verification matrix',
        '',
        f'{sum(r["ok"] for r in records)}/{len(records)} cases passed.',
        '',
        '| function | case | status | file |',
        '|-|-|-|-|',
    ]
    for r in records:
        rel = os.path.relpath(r['path'], os.path.join(SCREENSHOT_ROOT, TAG))
        status = 'pass' if r['ok'] else f"FAIL: {r['error']}"
        lines.append(f"| {r['function']} | {r['case']} | {status} | {rel} |")
    with open(index_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nManifest written to {index_path}')


if __name__ == '__main__':
    main()
