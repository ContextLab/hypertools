import numpy as np
import pandas as pd
from hypertools.manip.smooth import Smooth
from hypertools.manip.resample import Resample
from hypertools.manip.manip import manip


def test_smooth_reduces_variance_of_noisy_signal():
    rng = np.random.RandomState(0)
    t = np.linspace(0, 4 * np.pi, 200)
    clean = np.sin(t)
    noisy = clean + rng.normal(0, 0.5, size=t.shape)
    df = pd.DataFrame({"x": noisy})
    out = Smooth(kernel_width=21, order=3).fit_transform(df)
    # smoothed signal is closer to the clean signal than the noisy input
    assert np.mean((out["x"].to_numpy() - clean) ** 2) < np.mean((noisy - clean) ** 2)


def test_resample_changes_row_count():
    df = pd.DataFrame({"x": np.linspace(0, 1, 50), "y": np.linspace(1, 2, 50)})
    out = Resample(n_samples=17).fit_transform(df)
    assert out.shape[0] == 17


def test_manip_dispatcher_by_name():
    df = pd.DataFrame(np.random.RandomState(1).rand(20, 3))
    out = manip(df, model="ZScore")
    assert np.allclose(np.asarray(out).mean(axis=0), 0.0, atol=1e-9)
