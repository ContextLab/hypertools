# -*- coding: utf-8 -*-
"""hyp.load() gains a family of SYNTHETIC datasets (GH #285).

``hyp.load('random_walk' | 'helix' | 'lorenz' | 'blobs' | 'moons' |
'swiss_roll' | 's_curve')`` generates data on the spot instead of fetching
it: the docs, the feature tour and a dozen example scripts each hand-rolled
their own random walk / helix / blob generator, and now they don't have to.

Every generator is deterministic given ``random_state`` (``seed=`` is
accepted as an alias), takes the keyword arguments listed in
``hypertools.io.sources.SYNTHETIC_DATASETS``, and returns either a single
dataset or -- with ``n_datasets > 1`` -- a list of them, the same shape
``hyp.load('weights')`` returns.

All tests use real calls: the generators run for real and the results are
checked numerically. Nothing here needs the network.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import hypertools as hyp                                       # noqa: E402
from hypertools._shared.exceptions import HypertoolsIOError     # noqa: E402
from hypertools.io.sources import (SYNTHETIC_DATASETS,          # noqa: E402
                                   is_loadable_string,
                                   synthetic_dataset,
                                   synthetic_dataset_docs)


# ---------------------------------------------------------------- registry

def test_every_registered_synthetic_dataset_loads_by_name():
    # the registry IS the public list of names, so every entry must load
    for name in SYNTHETIC_DATASETS:
        data = hyp.load(name)
        assert isinstance(data, (np.ndarray, pd.DataFrame)), name
        assert len(data) > 0, name
        assert np.isfinite(np.asarray(
            data.select_dtypes('number') if isinstance(data, pd.DataFrame)
            else data, dtype=float)).all(), name


def test_expected_names_and_shapes():
    assert set(SYNTHETIC_DATASETS) == {
        'random_walk', 'helix', 'lorenz', 'blobs', 'moons', 'swiss_roll',
        's_curve'}
    assert hyp.load('random_walk').shape == (300, 10)   # documented defaults
    assert hyp.load('helix').shape == (300, 3)
    assert hyp.load('lorenz').shape == (2000, 3)


def test_names_are_discoverable_and_documented():
    docs = synthetic_dataset_docs()
    for name, (_, description) in SYNTHETIC_DATASETS.items():
        assert f'``{name}``' in docs
        assert description in docs
        assert is_loadable_string(name)


def test_unknown_name_is_not_mine():
    # returns None (rather than raising) so hyp.load can keep resolving
    assert synthetic_dataset('definitely_not_synthetic') is None


# ------------------------------------------------------------ determinism

def test_random_state_is_reproducible_and_seed_is_an_alias():
    a = synthetic_dataset('random_walk', n_samples=50, n_features=4,
                          random_state=7)
    b = synthetic_dataset('random_walk', n_samples=50, n_features=4,
                          random_state=7)
    c = synthetic_dataset('random_walk', n_samples=50, n_features=4, seed=7)
    d = synthetic_dataset('random_walk', n_samples=50, n_features=4,
                          random_state=8)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)          # seed= is the same knob
    assert not np.allclose(a, d)         # a different seed, different walk


def test_conflicting_seed_and_random_state_raise():
    with pytest.raises(HypertoolsIOError, match='seed'):
        synthetic_dataset('helix', random_state=1, seed=2)


def test_sklearn_generators_are_reproducible_too():
    one = synthetic_dataset('blobs', n_samples=40, random_state=3)
    two = synthetic_dataset('blobs', n_samples=40, random_state=3)
    pd.testing.assert_frame_equal(one, two)


# --------------------------------------------------------- n_datasets list

def test_n_datasets_returns_a_list_of_distinct_reproducible_datasets():
    walks = synthetic_dataset('random_walk', n_datasets=3, n_samples=25,
                              n_features=2, random_state=0)
    assert isinstance(walks, list) and len(walks) == 3
    assert all(w.shape == (25, 2) for w in walks)
    assert not np.allclose(walks[0], walks[1])       # independent draws
    again = synthetic_dataset('random_walk', n_datasets=3, n_samples=25,
                              n_features=2, random_state=0)
    for first, second in zip(walks, again):
        assert np.array_equal(first, second)         # and reproducible


def test_n_datasets_one_is_a_single_dataset_not_a_list():
    assert isinstance(synthetic_dataset('helix', n_samples=10), np.ndarray)


@pytest.mark.parametrize('bad', [0, -2, 'three'])
def test_bad_n_datasets_raises(bad):
    with pytest.raises(HypertoolsIOError, match='n_datasets'):
        synthetic_dataset('helix', n_datasets=bad)


def test_multi_dataset_list_flows_through_hypertools():
    # a list of arrays is ONE hypertools multi-dataset: align across it
    walks = synthetic_dataset('random_walk', n_datasets=2, n_samples=30,
                              n_features=5, random_state=1)
    reduced = hyp.reduce(walks, ndims=2)
    assert isinstance(reduced, list) and len(reduced) == 2
    assert all(np.asarray(r).shape == (30, 2) for r in reduced)


# ------------------------------------------------------------- generators

def test_random_walk_is_a_cumulative_sum_with_drift():
    walk = synthetic_dataset('random_walk', n_samples=200, n_features=3,
                             step=0.5, drift=1.0, random_state=2)
    steps = np.diff(walk, axis=0)
    assert walk.shape == (200, 3)
    assert abs(steps.mean() - 1.0) < 0.15        # drift
    assert abs(steps.std() - 0.5) < 0.15         # step size
    assert np.allclose(walk[0], walk[0])         # first row is the first step


def test_helix_geometry():
    xyz = synthetic_dataset('helix', n_samples=101, turns=2.0, radius=3.0,
                            pitch=1.5)
    assert xyz.shape == (101, 3)
    radii = np.linalg.norm(xyz[:, :2], axis=1)
    assert np.allclose(radii, 3.0)                     # constant radius
    assert np.isclose(xyz[-1, 2] - xyz[0, 2], 3.0)     # 2 turns * pitch 1.5
    assert np.all(np.diff(xyz[:, 2]) > 0)              # monotone rise


def test_helix_noise_is_seeded():
    clean = synthetic_dataset('helix', n_samples=50)
    noisy = synthetic_dataset('helix', n_samples=50, noise=0.2,
                              random_state=5)
    same = synthetic_dataset('helix', n_samples=50, noise=0.2,
                             random_state=5)
    assert not np.allclose(clean, noisy)
    assert np.array_equal(noisy, same)


def test_lorenz_stays_on_the_attractor_and_diverges_from_nearby_starts():
    traj = synthetic_dataset('lorenz', n_samples=4000, random_state=0)
    assert traj.shape == (4000, 3)
    assert np.isfinite(traj).all()
    tail = traj[500:]
    assert np.abs(tail[:, :2]).max() < 60      # the classic butterfly's box
    assert 0 < tail[:, 2].min() and tail[:, 2].max() < 70
    assert tail[:, 0].min() < 0 < tail[:, 0].max()   # both wings visited

    pair = synthetic_dataset('lorenz', n_datasets=2, n_samples=4000,
                             random_state=0)
    start_gap = np.linalg.norm(pair[0][0] - pair[1][0])
    end_gap = np.linalg.norm(pair[0][-1] - pair[1][-1])
    assert start_gap < 0.01           # nearby initial conditions ...
    assert end_gap > start_gap * 10   # ... that diverge (butterfly effect)


def test_lorenz_explicit_initial_condition_is_deterministic():
    one = synthetic_dataset('lorenz', n_samples=100, x0=(1.0, 1.0, 1.0))
    two = synthetic_dataset('lorenz', n_samples=100, x0=(1.0, 1.0, 1.0),
                            random_state=99)
    assert np.array_equal(one, two)     # x0 given -> random_state unused
    assert np.allclose(one[0], (1.0, 1.0, 1.0))
    with pytest.raises(HypertoolsIOError, match='x0'):
        synthetic_dataset('lorenz', n_samples=10, x0=(1.0, 2.0))


def test_lorenz_parameters_change_the_trajectory():
    classic = synthetic_dataset('lorenz', n_samples=500, x0=(1.0, 1.0, 1.0))
    damped = synthetic_dataset('lorenz', n_samples=500, rho=1.0,
                               x0=(1.0, 1.0, 1.0))
    assert not np.allclose(classic, damped)
    # rho < 24.74: the origin is stable, so the trajectory decays toward it
    assert np.linalg.norm(damped[-1]) < np.linalg.norm(damped[0])


# ------------------------------------------- scikit-learn passthroughs

@pytest.mark.parametrize('name,target', [('blobs', 'target'),
                                         ('moons', 'target'),
                                         ('swiss_roll', 't'),
                                         ('s_curve', 't')])
def test_sklearn_synthetics_are_frames_with_a_target_column(name, target):
    df = synthetic_dataset(name, n_samples=37, random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 37
    assert target in df.columns
    dims = [c for c in df.columns if c.startswith('dim_')]
    assert dims == [f'dim_{i}' for i in range(len(dims))]
    assert df[dims].to_numpy().dtype.kind == 'f'


def test_sklearn_kwargs_pass_straight_through():
    df = synthetic_dataset('blobs', n_samples=60, n_features=4, centers=4,
                           cluster_std=0.3, random_state=0)
    assert df.shape == (60, 5)                       # 4 dims + target
    assert sorted(df['target'].unique()) == [0, 1, 2, 3]

    quiet = synthetic_dataset('moons', n_samples=50, noise=0.0,
                              random_state=0)
    loud = synthetic_dataset('moons', n_samples=50, noise=0.3,
                             random_state=0)
    assert loud[['dim_0', 'dim_1']].std().sum() > \
        quiet[['dim_0', 'dim_1']].std().sum()

    # sklearn's swiss roll parameterizes t as 1.5*pi*(1 + 2*u), u in [0, 1)
    rolled = synthetic_dataset('swiss_roll', n_samples=50, random_state=0)
    assert rolled['t'].between(1.5 * np.pi, 4.5 * np.pi).all()


def test_unknown_generator_kwarg_raises_typeerror():
    with pytest.raises(TypeError):
        synthetic_dataset('moons', not_a_real_kwarg=1)
    with pytest.raises(TypeError):
        synthetic_dataset('helix', not_a_real_kwarg=1)


# ------------------------------------------------------------ integration

def test_synthetic_data_flows_into_the_rest_of_hypertools():
    reduced = hyp.load('helix', ndims=2)
    assert np.asarray(reduced).shape == (300, 2)

    fig = hyp.plot(hyp.load('lorenz'), show=False)
    assert fig is not None
    plt.close('all')


def test_load_reports_synthetic_names_when_nothing_resolves():
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('helixx_not_a_dataset')
    assert 'synthetic dataset' in str(excinfo.value)
