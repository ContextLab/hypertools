import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def _datasets(n=3, rows=20, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _alphas(fig):
    return [ln.get_alpha() for ln in _ax(fig).lines]


def test_scalar_alpha_still_applies_to_every_dataset():
    """Guards tests/test_gh206_extra_kwargs.py::test_alpha_kwarg_reaches_line
    _artists, which must keep passing after alpha leaves **kwargs."""
    fig = hyp.plot(_datasets(), '-', alpha=0.25, show=False)
    assert _alphas(fig) == pytest.approx([0.25, 0.25, 0.25])


def test_per_dataset_alpha_list():
    fig = hyp.plot(_datasets(), '-', alpha=[0.1, 0.5, 1.0], show=False)
    assert _alphas(fig) == pytest.approx([0.1, 0.5, 1.0])


def test_per_dataset_alpha_length_mismatch_raises():
    with pytest.raises(ValueError, match='alpha has 2 entries'):
        hyp.plot(_datasets(), '-', alpha=[0.1, 0.5], show=False)


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match='between 0 and 1'):
        hyp.plot(_datasets(), '-', alpha=[0.1, 0.5, 1.7], show=False)


def test_non_numeric_alpha_raises():
    with pytest.raises(ValueError, match='alpha'):
        hyp.plot(_datasets(), '-', alpha=['a', 'b', 'c'], show=False)


def test_per_dataset_alpha_survives_animation():
    fig, ani = hyp.plot(_datasets(), '-', alpha=[0.1, 0.5, 1.0],
                        animate=True, duration=1, frame_rate=2, show=False)
    ani._func(1, *ani._args)
    assert _alphas(fig) == pytest.approx([0.1, 0.5, 1.0])


def test_per_dataset_alpha_reaches_plotly_traces():
    """plotly_backend.py:776 already reads alpha off kwargs_list."""
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', alpha=[0.1, 0.5, 1.0], show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    alphas = [float(t.line.color.rsplit(',', 1)[1].rstrip(') '))
              for t in fig.data
              if t.line is not None and t.line.color is not None
              and t.line.color.startswith('rgba')]
    assert alphas[:3] == pytest.approx([0.1, 0.5, 1.0])


# --- precedence (review G1) -------------------------------------------------

def _multiindex_frame(seed=0):
    idx = pd.MultiIndex.from_tuples(
        [('cond1', s) for s in range(3)] + [('cond2', s) for s in range(3)],
        names=['cond', 'subj'])
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(6, 4)), index=idx)


def test_multiindex_level_fading_wins_and_says_so():
    """Mirrors the linewidth= precedent at plot.py:3045-3050: internal
    styling wins over a same-named user kwarg, with a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(_multiindex_frame(), '-', alpha=0.9, show=False)
    assert [w for w in caught
            if 'alpha' in str(w.message) and 'MultiIndex' in str(w.message)]
    alphas = [ln.get_alpha() for ln in _ax(fig).lines]
    assert not all(a == pytest.approx(0.9) for a in alphas if a is not None)


def test_nested_list_depth_fading_wins_and_says_so():
    """plot.py:3629 writes a depth-derived alpha list for nested inputs."""
    rng = np.random.default_rng(0)
    nested = [[rng.normal(size=(10, 4)).cumsum(axis=0) for _ in range(2)],
              rng.normal(size=(10, 4)).cumsum(axis=0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(nested, '-', alpha=0.9, show=False)
    assert [w for w in caught if 'alpha' in str(w.message)]


def test_alpha_survives_contiguous_run_segmentation():
    """A categorical hue turns N datasets into >= N runs
    (_expand_styles_to_runs, plot.py:231-263); a per-dataset alpha must be
    expanded, not length-checked against the run count."""
    ds = _datasets(n=2, rows=20)
    labels = np.array(['a'] * 10 + ['b'] * 10 + ['a'] * 10 + ['b'] * 10)
    fig = hyp.plot(ds, '-', hue=labels, alpha=[0.2, 0.8], show=False)
    alphas = [a for a in (ln.get_alpha() for ln in _ax(fig).lines)
              if a is not None]
    assert set(np.round(alphas, 6)) <= {0.2, 0.8}
    assert len(alphas) > 2, 'expected more runs than datasets'
