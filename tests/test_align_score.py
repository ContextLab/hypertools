"""Tests for the GH #285 alignment-quality score:
`hypertools.align.score.alignment_score` and `hyp.align(..., return_score=True)`.
"""
import numpy as np
import pytest

import hypertools as hyp
from hypertools.align.align import align
from hypertools.align.score import alignment_score


# -- reproduce examples/plot_story_trajectories.py's inline dispersion() -----

def _example_dispersion(trajectories):
    """Verbatim copy of the `dispersion()` helper in
    examples/plot_story_trajectories.py:65-77, so its numbers can be
    compared against `alignment_score(..., metric='dispersion')`."""
    stack = np.stack([np.asarray(t) for t in trajectories])
    centroid = stack.mean(axis=0, keepdims=True)
    spread = np.linalg.norm(stack - centroid, axis=2).mean()
    scale = np.linalg.norm(stack - stack.mean(axis=(0, 1)), axis=2).mean()
    return spread / scale


def _rotated_copies(rng, n=4, n_obs=30, n_features=5, noise=0.2):
    """`n` noisy rotated copies of one base dataset -- an easy case where
    HyperAlign should visibly reduce dispersion / raise ISC."""
    base = rng.standard_normal((n_obs, n_features))
    datasets = []
    for _ in range(n):
        rot, _ = np.linalg.qr(rng.standard_normal((n_features, n_features)))
        datasets.append(base @ rot + noise * rng.standard_normal((n_obs, n_features)))
    return datasets


def test_dispersion_matches_the_story_example_exactly():
    rng = np.random.default_rng(0)
    datasets = _rotated_copies(rng)
    expected = _example_dispersion(datasets)
    score = alignment_score(datasets, metric='dispersion')
    assert score['metric'] == 'dispersion'
    assert score['after'] is None
    assert score['before'] == pytest.approx(expected)


def test_dispersion_matches_story_example_before_and_after_alignment():
    rng = np.random.default_rng(1)
    datasets = _rotated_copies(rng, n=6, n_obs=40, n_features=8)
    aligned = align(datasets, model='HyperAlign', n_iter=10)

    expected_before = _example_dispersion(datasets)
    expected_after = _example_dispersion(aligned)

    score = alignment_score(datasets, aligned=aligned, metric='dispersion')
    assert score['before'] == pytest.approx(expected_before)
    assert score['after'] == pytest.approx(expected_after)


def test_score_improves_after_hyperalign_on_rotated_copies():
    rng = np.random.default_rng(2)
    datasets = _rotated_copies(rng, n=5, n_obs=50, n_features=6, noise=0.05)
    aligned, score = align(datasets, model='HyperAlign', n_iter=10,
                           return_score=True)
    assert score['metric'] == 'dispersion'
    assert score['after'] < score['before']

    # isc should IMPROVE (increase) after alignment too
    aligned2, isc_score = align(datasets, model='HyperAlign', n_iter=10,
                                return_score=True, score_metric='isc')
    assert isc_score['metric'] == 'isc'
    assert isc_score['after'] > isc_score['before']


def test_isc_is_bounded():
    rng = np.random.default_rng(3)
    datasets = _rotated_copies(rng, n=3, n_obs=25, n_features=4)
    score = alignment_score(datasets, metric='isc')
    assert -1.0 <= score['before'] <= 1.0

    aligned = align(datasets, model='HyperAlign', n_iter=10)
    score2 = alignment_score(datasets, aligned=aligned, metric='isc')
    assert -1.0 <= score2['after'] <= 1.0


def test_ragged_datasets_raise_value_error():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((10, 3))
    b = rng.standard_normal((12, 3))  # different number of rows
    with pytest.raises(ValueError, match='same shape'):
        alignment_score([a, b], metric='dispersion')

    c = rng.standard_normal((10, 4))  # different number of columns
    with pytest.raises(ValueError, match='same shape'):
        alignment_score([a, c], metric='dispersion')


def test_unknown_metric_raises_value_error():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((10, 3))
    with pytest.raises(ValueError, match='unknown alignment_score metric'):
        alignment_score([a, a.copy()], metric='not-a-metric')


def test_return_score_alone():
    rng = np.random.default_rng(6)
    datasets = _rotated_copies(rng, n=3, n_obs=20, n_features=4)
    result = align(datasets, model='HyperAlign', n_iter=5, return_score=True)
    assert len(result) == 2
    aligned, score = result
    assert isinstance(aligned, list) and len(aligned) == 3
    assert set(score) == {'before', 'after', 'metric'}


def test_return_score_combined_with_return_model_tuple_order():
    """return_model= and return_score= combine as (aligned, model, score) --
    model keeps the position it already had for return_model=True-only
    callers, with score appended last."""
    rng = np.random.default_rng(7)
    datasets = _rotated_copies(rng, n=3, n_obs=20, n_features=4)

    result = align(datasets, model='HyperAlign', n_iter=5,
                   return_model=True, return_score=True)
    assert len(result) == 3
    aligned, model, score = result
    assert isinstance(aligned, list) and len(aligned) == 3
    assert hasattr(model, 'transform')  # a fitted Aligner
    assert set(score) == {'before', 'after', 'metric'}

    # return_model=True alone is unaffected (still a 2-tuple)
    result_model_only = align(datasets, model='HyperAlign', n_iter=5,
                              return_model=True)
    assert len(result_model_only) == 2


def test_return_score_rejects_cross_module_pipeline():
    rng = np.random.default_rng(8)
    datasets = _rotated_copies(rng, n=3, n_obs=20, n_features=4)
    with pytest.raises(ValueError, match='multi-stage'):
        align(datasets, model='HyperAlign', normalize='ZScore',
              return_score=True)


def test_hyp_align_return_score_via_public_api():
    rng = np.random.default_rng(9)
    datasets = _rotated_copies(rng, n=4, n_obs=30, n_features=5, noise=0.1)
    aligned, score = hyp.align(datasets, model='HyperAlign', n_iter=10,
                               return_score=True)
    assert score['after'] <= score['before']


# -- 1.1 release review: ragged input scores what the aligner consumed ------

def test_return_score_works_on_ragged_input_that_align_trims():
    # hyp.align([a, b]) trims to the 40 common rows and succeeds, so
    # return_score=True must succeed too: the "before" score is taken on
    # the same row-trimmed data the aligner consumed.
    rng = np.random.default_rng(11)
    base = rng.standard_normal((50, 4))
    rot, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    a = base + 0.05 * rng.standard_normal((50, 4))
    b = (base @ rot + 0.05 * rng.standard_normal((50, 4)))[:40]
    with pytest.warns(UserWarning, match='common to all datasets'):
        aligned, score = hyp.align([a, b], model='HyperAlign', n_iter=10,
                                   return_score=True)
    assert [x.shape for x in aligned] == [(40, 4), (40, 4)]
    assert set(score) == {'before', 'after', 'metric'}
    assert score['metric'] == 'dispersion'
    assert score['after'] < score['before']
    # 'before' is exactly alignment_score of the manually trimmed arrays
    # (common rows in the FIRST dataset's order: a's first 40 rows)
    expected = alignment_score([a[:40], b], metric='dispersion')['before']
    assert score['before'] == pytest.approx(expected)
    assert score['after'] == pytest.approx(
        alignment_score([a[:40], b], aligned=aligned, metric='dispersion')['after'])

    # the plain (un-trimmed) originals are still rejected by the scorer
    # itself, so the trim is align's doing rather than a relaxed check
    with pytest.raises(ValueError, match='same shape'):
        alignment_score([a, b], metric='dispersion')


def test_return_score_ragged_input_isc_metric():
    rng = np.random.default_rng(12)
    base = rng.standard_normal((50, 4))
    rot, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    a = base + 0.05 * rng.standard_normal((50, 4))
    b = (base @ rot + 0.05 * rng.standard_normal((50, 4)))[:40]
    with pytest.warns(UserWarning, match='common to all datasets'):
        aligned, score = hyp.align([a, b], model='HyperAlign', n_iter=10,
                                   return_score=True, score_metric='isc')
    assert score['metric'] == 'isc'
    assert score['after'] > score['before']
    expected = alignment_score([a[:40], b], metric='isc')['before']
    assert score['before'] == pytest.approx(expected)


# --- degenerate and malformed input (release review 2026-09-07) -------------

@pytest.mark.parametrize('metric', ['dispersion', 'isc'])
def test_all_constant_datasets_raise_instead_of_nan(metric):
    """Two constant datasets have no cloud scale ('dispersion' divided 0 by 0
    and returned NaN with a RuntimeWarning) and no feature to correlate."""
    const = [np.ones((10, 3)), np.ones((10, 3))]
    with pytest.raises(ValueError, match='constant'):
        alignment_score(const, metric=metric)


@pytest.mark.parametrize('metric', ['dispersion', 'isc'])
def test_datasets_each_constant_at_their_own_value_raise(metric):
    """1.1 release review (2026-09-11): 'dispersion' raised only when EVERY
    observation of EVERY dataset was the same point. Datasets that are each
    constant at a DIFFERENT value have a cloud scale, so the score came out
    as exactly 1.0 -- the same number whatever the alignment -- while 'isc'
    raised. The docstring promises a raise for "every dataset constant"."""
    const = [np.zeros((10, 3)), np.ones((10, 3)), np.full((10, 3), 5.0)]
    with pytest.raises(ValueError, match='constant') as info:
        alignment_score(const, metric=metric)
    if metric == 'dispersion':
        assert 'dataset 0' in str(info.value)
        assert 'dataset 2' in str(info.value)
    rng = np.random.default_rng(3)
    fine = [rng.normal(size=(10, 3)) for _ in range(3)]
    with pytest.raises(ValueError, match='constant'):
        alignment_score(fine, aligned=const, metric=metric)


def test_dispersion_of_single_observation_datasets_raises_like_isc():
    """One observation per dataset is the degenerate constant case: the
    per-observation centroid IS the cloud's mean, so 'dispersion' is 1.0
    for any input ('isc' already raised: it needs two observations)."""
    rng = np.random.default_rng(4)
    single_rows = [rng.normal(size=(1, 3)) for _ in range(3)]
    for metric in ('dispersion', 'isc'):
        with pytest.raises(ValueError):
            alignment_score(single_rows, metric=metric)


def test_one_constant_dataset_among_varying_ones_still_scores():
    """Only the all-constant case is degenerate: a constant dataset beside
    varying ones has a well-defined dispersion (and 'isc' correlates the
    varying pairs)."""
    rng = np.random.default_rng(5)
    data = [rng.normal(size=(10, 3)), rng.normal(size=(10, 3)),
            np.ones((10, 3))]
    for metric in ('dispersion', 'isc'):
        score = alignment_score(data, metric=metric)['before']
        assert np.isfinite(score)
    assert alignment_score(data, metric='dispersion')['before'] != 1.0


@pytest.mark.parametrize('metric', ['dispersion', 'isc'])
def test_nan_input_raises_instead_of_nan_score(metric):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(10, 3)), rng.normal(size=(10, 3))
    y[2, 1] = np.nan
    with pytest.raises(ValueError, match=r'finite values; dataset 1 has 1 NaN'):
        alignment_score([x, y], metric=metric)
    with pytest.raises(ValueError, match='finite values'):
        alignment_score([x, x], aligned=[x, y], metric=metric)


@pytest.mark.parametrize('metric', ['dispersion', 'isc'])
def test_one_dimensional_input_raises_a_clear_error(metric):
    """1-D series raised numpy's own AxisError / unpack ValueError."""
    with pytest.raises(ValueError, match=r'2-D datasets .* dataset 0 has shape \(10,\)'):
        alignment_score([np.arange(10.0), np.arange(10.0)], metric=metric)
    with pytest.raises(ValueError, match='numeric'):
        alignment_score([np.array([['a', 'b']] * 3)] * 2, metric=metric)
