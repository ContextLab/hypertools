"""Tests for the six torch-backed autoencoder reducers (GH #162,
`hypertools.reduce.autoencoders`): Autoencoder, DeepAutoencoder,
SparseAutoencoder, ConvolutionalAutoencoder, SequenceAutoencoder, and
VariationalAutoencoder.

All data is real (tiny, synthetic) numeric arrays; all models are real
(tiny) torch networks trained for a handful of epochs on CPU -- no mocks.
`torch` is required to run this module (it ships via the optional
`[torch]` extra and is listed in the `dev` extra so CI always exercises
it); the torch-ABSENT `ImportError` path (`resolve_reducer`) is exercised
separately, in a subprocess with a real import-blocking `sys.meta_path`
hook (no fake/mocked `torch`).
"""
import subprocess
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('torch')

import hypertools as hyp
from hypertools.reduce.reduce import reduce as reducer
from hypertools.reduce.common import Reducer, resolve_reducer
from hypertools.reduce.autoencoders import (
    Autoencoder, DeepAutoencoder, SparseAutoencoder,
    ConvolutionalAutoencoder, SequenceAutoencoder, VariationalAutoencoder,
    AUTOENCODER_NAMES,
)

ALL_CLASSES = [
    Autoencoder, DeepAutoencoder, SparseAutoencoder,
    ConvolutionalAutoencoder, SequenceAutoencoder, VariationalAutoencoder,
]


def _rng():
    return np.random.RandomState(0)


def _data(n=200, n_features=10, n_latent=3):
    """A tiny 200x10 low-rank-plus-noise synthetic dataset."""
    rng = _rng()
    latent = rng.randn(n, n_latent)
    proj = rng.randn(n_latent, n_features)
    return latent @ proj + 0.1 * rng.randn(n, n_features)


def _mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# --- registry -------------------------------------------------------------

def test_autoencoder_names_lazily_resolvable():
    for name in AUTOENCODER_NAMES:
        cls = resolve_reducer(name)
        assert cls.__name__ == name


def test_autoencoder_names_excluded_from_eager_reducers():
    from hypertools.reduce.common import REDUCERS
    for name in AUTOENCODER_NAMES:
        assert name not in REDUCERS


# --- per-variant contract (parametrized over all six classes) ------------

@pytest.mark.parametrize('cls', ALL_CLASSES, ids=lambda c: c.__name__)
def test_fit_transform_shape(cls):
    x = _data()
    model = cls(n_components=3, epochs=20, random_state=0)
    z = model.fit_transform(x)
    assert np.asarray(z).shape == (200, 3)


@pytest.mark.parametrize('cls', ALL_CLASSES, ids=lambda c: c.__name__)
def test_inverse_transform_round_trip_shape(cls):
    x = _data()
    model = cls(n_components=3, epochs=20, random_state=0)
    z = model.fit_transform(x)
    xhat = model.inverse_transform(z)
    assert np.asarray(xhat).shape == x.shape


@pytest.mark.parametrize('cls', ALL_CLASSES, ids=lambda c: c.__name__)
def test_training_improves_reconstruction_mse(cls):
    x = _data()

    trained = cls(n_components=3, epochs=25, random_state=0)
    z_trained = trained.fit_transform(x)
    mse_trained = _mse(trained.inverse_transform(z_trained), x)

    # untrained baseline: same random_state (identical weight init) but
    # epochs=0 -- the training loop never runs, so the network is
    # evaluated at its random initialization only.
    untrained = cls(n_components=3, epochs=0, random_state=0)
    z_untrained = untrained.fit_transform(x)
    mse_untrained = _mse(untrained.inverse_transform(z_untrained), x)

    assert mse_trained < mse_untrained


@pytest.mark.parametrize('cls', ALL_CLASSES, ids=lambda c: c.__name__)
def test_determinism_same_seed_identical_different_seed_differs(cls):
    x = _data()
    a = cls(n_components=3, epochs=10, random_state=42).fit_transform(x)
    b = cls(n_components=3, epochs=10, random_state=42).fit_transform(x)
    c = cls(n_components=3, epochs=10, random_state=43).fit_transform(x)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


@pytest.mark.parametrize('cls', ALL_CLASSES, ids=lambda c: c.__name__)
def test_transform_reuses_fitted_network_without_refit(cls):
    x = _data()
    model = cls(n_components=3, epochs=10, random_state=0)
    model.fit_transform(x)

    def _poison(*args, **kwargs):
        raise AssertionError('must not refit an already-fitted autoencoder')
    model.fit = _poison
    model.fit_transform = _poison

    new_x = _rng().rand(15, 10)
    z = model.transform(new_x)
    assert np.asarray(z).shape == (15, 3)


# --- VariationalAutoencoder-specific -------------------------------------

def test_vae_latent_means_roughly_centered():
    x = _data()
    vae = VariationalAutoencoder(n_components=3, epochs=30, random_state=0)
    z = vae.fit_transform(x)
    assert np.all(np.abs(z.mean(axis=0)) < 1.0)  # loose bound


def test_vae_transform_returns_means_not_samples():
    # transform() must be deterministic (returns mu, not a stochastic
    # reparameterized sample) -- repeated calls on the same data give
    # identical output.
    x = _data()
    vae = VariationalAutoencoder(n_components=3, epochs=15, random_state=0)
    vae.fit_transform(x)
    z1 = vae.transform(x)
    z2 = vae.transform(x)
    assert np.array_equal(z1, z2)


# --- SparseAutoencoder-specific --------------------------------------------

def test_sparse_autoencoder_higher_sparsity_weight_reduces_hidden_activation():
    x = _data()
    low = SparseAutoencoder(n_components=3, sparsity_weight=1e-5, epochs=25,
                             random_state=0)
    low.fit_transform(x)
    high = SparseAutoencoder(n_components=3, sparsity_weight=1.0, epochs=25,
                              random_state=0)
    high.fit_transform(x)

    import torch
    with torch.no_grad():
        low_x_t = torch.as_tensor((x - low.mean_) / low.std_, dtype=torch.float32,
                                   device=low.device_)
        high_x_t = torch.as_tensor((x - high.mean_) / high.std_, dtype=torch.float32,
                                    device=high.device_)
        low_hidden = low.net_.act(low.net_.enc1(low_x_t)).abs().mean().item()
        high_hidden = high.net_.act(high.net_.enc1(high_x_t)).abs().mean().item()
    assert high_hidden < low_hidden


# --- through the dispatcher (hyp.reduce) -----------------------------------

def test_dispatcher_string_spec():
    x = _data()
    out = reducer(x, reduce='Autoencoder', ndims=3)
    assert np.asarray(out).shape == (200, 3)


def test_dispatcher_dict_spec_vae():
    x = _data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out = reducer(
            x, reduce={'model': 'VariationalAutoencoder',
                       'kwargs': {'epochs': 20, 'random_state': 0}},
            ndims=2)
    assert np.asarray(out).shape[1] > 0


def test_dispatcher_return_model_reuse_poison_pill():
    x = _data()
    out, fitted = reducer(x, reduce='Autoencoder', ndims=3, return_model=True)
    assert isinstance(fitted, Reducer)
    assert fitted.is_fitted
    assert np.asarray(out).shape == (200, 3)

    def _poison(*args, **kwargs):
        raise AssertionError('must not refit an already-fitted Reducer')
    fitted.model_.fit = _poison
    fitted.model_.fit_transform = _poison

    new_x = _rng().rand(12, 10)
    out2 = reducer([new_x], reduce=fitted, format_data=False)
    assert np.asarray(out2).shape == (12, 3)


def test_hyp_plot_with_autoencoder_produces_figure():
    x = _data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig = hyp.plot(
            x, reduce={'model': 'Autoencoder', 'kwargs': {'epochs': 10, 'random_state': 0}},
            show=False)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure) or hasattr(fig, 'savefig')


# --- torch-absent ImportError path (real subprocess, no mocks) -----------

def test_resolving_autoencoder_without_torch_raises_friendly_import_error():
    script = """
import sys
import numpy as np
import hypertools as hyp  # import fully first, with real torch available

class _BlockTorch:
    def find_spec(self, name, path, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ImportError("No module named 'torch' (blocked for testing)")
        return None

sys.meta_path.insert(0, _BlockTorch())

try:
    hyp.reduce(np.random.RandomState(0).rand(20, 5), reduce='Autoencoder', ndims=2)
except ImportError as e:
    assert 'hypertools[torch]' in str(e), str(e)
    print('IMPORT_ERROR_OK')
else:
    raise SystemExit('expected ImportError')
"""
    result = subprocess.run([sys.executable, '-c', script],
                             capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'IMPORT_ERROR_OK' in result.stdout
