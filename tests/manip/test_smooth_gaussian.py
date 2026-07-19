import numpy as np
import pandas as pd
from hypertools.manip.smooth import Smooth
from hypertools.manip.manip import manip


def test_gaussian_smooth_reduces_variance():
    rng = np.random.RandomState(0)
    t = np.linspace(0, 4 * np.pi, 200)
    clean = np.sin(t)
    noisy = clean + rng.normal(0, 0.5, size=t.shape)
    df = pd.DataFrame({"x": noisy})
    out = Smooth(mode="gaussian", var=300).fit_transform(df)
    # gaussian-smoothed signal is closer to clean than the noisy input
    assert np.mean((out["x"].to_numpy() - clean) ** 2) < np.mean((noisy - clean) ** 2)


def test_gaussian_matches_scipy_reference():
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.RandomState(1)
    x = rng.rand(120, 3)
    df = pd.DataFrame(x)
    out = np.asarray(Smooth(mode="gaussian", var=300, axis=0).fit_transform(df))
    ref = gaussian_filter1d(x.astype(float), sigma=np.sqrt(300), axis=0)
    assert np.allclose(out, ref, atol=1e-8)


def test_savgol_still_default():
    rng = np.random.RandomState(2)
    df = pd.DataFrame(rng.rand(100, 2))
    # default mode is savgol; must not raise and must change the data
    out = Smooth(kernel_width=11, order=3).fit_transform(df)
    assert np.asarray(out).shape == (100, 2)


def test_gaussian_via_dispatcher():
    df = pd.DataFrame(np.random.RandomState(3).rand(80, 2))
    out = manip(df, model="Smooth", mode="gaussian", var=300)
    assert np.asarray(out).shape == (80, 2)
