"""Tests for round17 Task 15: the GH #275 "story trajectories" demo
(`scripts/round17_evidence/story_trajectories.py`) and the GH #274 jumps
evidence it produces.

`hyp.load('weights')` + `reduce='UMAP'` are both slow/network-heavy, so the
gate-fast tests below run the SAME code paths (real `hyp.manip`/`hyp.reduce`/
`hyp.align`/`hyp.plot` calls -- no mocks) on a tiny synthetic proxy: 3
"subjects" wandering along a SHARED latent 3-D path, each observed through
its own random linear projection into a higher-dimensional native space plus
noise -- shaped like the real `weights` dataset (per-subject, per-timepoint
matrices), and exactly the setup `HyperAlign` is meant to undo. The one real
`hyp.load('weights')` + UMAP run is a separate, explicitly-opt-in
`@pytest.mark.bigdata` test (deselected by default, matching this repo's
existing bigdata-marker convention for slow/network tests).
"""
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts',
                                'round17_evidence'))
from story_trajectories import (  # noqa: E402
    MANIP_SPEC, pairwise_trajectory_correlation, mean_turning_angle,
    max_jump_distance,
)

DURATION = 2
FRAME_RATE = 10  # duration * frame_rate = 20, exact w/ n_samples=1000 (see
                 # MANIP_SPEC's Resample step -- the interpolation arithmetic
                 # in hypertools/_shared/helpers.interp_array_list lands on
                 # an exact integer frame count for this (n, frame_rate,
                 # duration) combination)
FOCUSED = 1


def _wandering_subjects(n_subjects=3, n_time=60, n_features=10, seed=0):
    """3 subjects observing the SAME shared wandering 3-D latent path, each
    through its own random linear projection into native space plus noise --
    shaped like `weights` (per-subject matrices over shared story time) and
    exactly what `HyperAlign` is meant to recover: real (not tautological)
    shared structure that only becomes visible after alignment."""
    rng = np.random.default_rng(seed)
    latent = np.cumsum(rng.standard_normal((n_time, 3)), axis=0)
    datasets = []
    for _ in range(n_subjects):
        proj = rng.standard_normal((3, n_features))
        noise = rng.standard_normal((n_time, n_features)) * 0.5
        datasets.append(latent @ proj + noise)
    return datasets


@pytest.fixture
def subjects():
    return _wandering_subjects()


# ---------------------------------------------------------------------------
# GH #275: the full snippet-shape call, fast proxy
# ---------------------------------------------------------------------------

def test_snippet_shape_call_returns_figure_and_animation(subjects):
    """Jeremy's exact GH #275 snippet, shape-for-shape: manip= chain,
    align=HyperAlign (n_iter=2 for speed), animate='window',
    reduce='PCA' (UMAP is real but slow -- PCA is real and fast, and
    exercises the identical dispatch code path), duration=/focused=."""
    fig, ani = hyp.plot(
        subjects, manip=MANIP_SPEC,
        align={'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 2}},
        animate='window', reduce='PCA', duration=DURATION, focused=FOCUSED,
        frame_rate=FRAME_RATE, show=False,
    )
    assert isinstance(fig, plt.Figure)
    assert ani is not None
    plt.close(fig)


def test_window_animation_frame_count_matches_duration_times_frame_rate(
        subjects):
    bundle = hyp.plot(
        subjects, manip=MANIP_SPEC,
        align={'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 2}},
        animate='window', reduce='PCA', duration=DURATION, focused=FOCUSED,
        frame_rate=FRAME_RATE, show=False, return_model=True,
    )
    ani = bundle['animation']
    assert ani._save_count == DURATION * FRAME_RATE
    plt.close(bundle['fig'])


def test_manip_chain_produces_n_samples_rows(subjects):
    manipulated = hyp.manip(subjects, model=MANIP_SPEC)
    for d in manipulated:
        assert d.shape[0] == 1000  # MANIP_SPEC's Resample(n_samples=1000)


def test_post_align_correlation_exceeds_pre_align(subjects):
    """The GH #275 acceptance metric: HyperAlign must make the (same-seed,
    independently-projected) subjects' trajectories correlate MORE, not
    less -- computed the same way the evidence script's real-data run
    reports it."""
    manipulated = hyp.manip(subjects, model=MANIP_SPEC)
    pre_align = hyp.reduce(
        manipulated, reduce={'model': 'PCA', 'kwargs': {'random_state': 0}},
        ndims=3)
    post_align = hyp.align(
        pre_align, model={'model': 'HyperAlign', 'kwargs': {'n_iter': 5}})

    pre_corr = pairwise_trajectory_correlation(pre_align)
    post_corr = pairwise_trajectory_correlation(post_align)
    assert post_corr > pre_corr, (
        f'post-align correlation ({post_corr:.4f}) must exceed pre-align '
        f'({pre_corr:.4f})')


def test_post_align_paths_are_non_linear(subjects):
    """The GH #275 "interesting (not just straight-line) paths" acceptance
    metric: mean turning angle along a real wandering path is well above a
    straight line's ~0."""
    manipulated = hyp.manip(subjects, model=MANIP_SPEC)
    reduced = hyp.reduce(
        manipulated, reduce={'model': 'PCA', 'kwargs': {'random_state': 0}},
        ndims=3)
    aligned = hyp.align(
        reduced, model={'model': 'HyperAlign', 'kwargs': {'n_iter': 5}})

    straight_line = np.column_stack([np.linspace(0, 1, 50)] * 3)
    straight_baseline = mean_turning_angle(straight_line)
    assert straight_baseline < 1e-6

    turning_angles = [mean_turning_angle(t) for t in aligned]
    assert np.mean(turning_angles) > 0.05  # radians -- clearly non-linear


# ---------------------------------------------------------------------------
# GH #274: jumps evidence, fast proxy
# ---------------------------------------------------------------------------

def test_chained_manip_reduces_max_jump_vs_no_manip(subjects):
    """The GH #274 numeric relationship: smoothing (chained manip) must
    reduce the max inter-frame jump distance relative to raw (no-manip)
    data."""
    no_manip = subjects
    chained_manip = hyp.manip(subjects, model=MANIP_SPEC)

    reduced_none = hyp.reduce(
        no_manip, reduce={'model': 'PCA', 'kwargs': {'random_state': 0}},
        ndims=3)
    reduced_smooth = hyp.reduce(
        chained_manip, reduce={'model': 'PCA', 'kwargs': {'random_state': 0}},
        ndims=3)

    max_jump_none = max(max_jump_distance(t) for t in reduced_none)
    max_jump_smooth = max(max_jump_distance(t) for t in reduced_smooth)
    assert max_jump_smooth < max_jump_none, (
        f'chained-manip max-jump ({max_jump_smooth:.4f}) must be < '
        f'no-manip max-jump ({max_jump_none:.4f})')


# ---------------------------------------------------------------------------
# real weights data + UMAP -- slow/network, opt-in only
# ---------------------------------------------------------------------------

@pytest.mark.bigdata
def test_real_weights_snippet_end_to_end():
    """Jeremy's exact GH #275 snippet against the real `weights` dataset
    with `reduce='UMAP'` -- slow (network load + UMAP fit on 36 x 1000 x 100
    points) and deselected by default (`pytest -m bigdata` to opt in).
    The full evidence generator (`scripts/round17_evidence/
    story_trajectories.py`) is the authoritative real-data run; this test
    just guards that the exact snippet keeps working end to end."""
    data = hyp.load('weights')
    manip = [
        {'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 25}},
        {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}},
        'ZScore',
    ]
    hyperalign = {'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 10}}
    fig, ani = hyp.plot(
        data, manip=manip, align=hyperalign, animate='window',
        reduce='UMAP', duration=30, focused=4, show=False,
    )
    assert isinstance(fig, plt.Figure)
    # NOT an exact duration*frame_rate=900: the GH #141 line-smoothing
    # interpolation (hypertools/_shared/helpers.py:interp_array) re-samples
    # via np.arange(0, n-1, 1/interp_val), whose float step can round to one
    # extra point depending on the (n, frame_rate, duration) combination --
    # here n=1000 (Resample's n_samples=1000) lands on 901, not 900 (the
    # synthetic-data test above uses an (n, frame_rate, duration) combo that
    # happens to land exactly on frame_rate*duration). Both are correct;
    # only the exact-equality assumption was wrong.
    assert abs(ani._save_count - 30 * 30) <= 2
    plt.close(fig)
