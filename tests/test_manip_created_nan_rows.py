"""Rows a `manip=` stage leaves with no values are the manip stage's doing
(1.1 release review, 2026-09-11).

A trailing ``Smooth(center=False)`` leaves its first ``kernel_width - 1``
rows NaN (pandas' rolling-window semantics) unless ``min_periods=1``. On a
finite input, ``hyp.plot(x, manip=<that smoother>)`` first emitted the
missing-data imputation warnings -- "Missing data: filling missing values
with PPCA ..." and "PPCA cannot fill 4 row(s) ... Use model='Kalman'",
which blame the input and point at the wrong remedy -- and only then raised
the right error; ``hyp.analyze``/``hyp.reduce``/``hyp.cluster``/
``hyp.align`` with ``manip=`` gave the same warnings followed by
scikit-learn's "Input X contains NaN" or numpy's "SVD did not converge".

The pipeline now stops right after the manip stage, before any later stage
tries to impute rows that have no observed feature at all, with the
manip-stage diagnosis and the ``min_periods=1`` hint. Real data, real
calls, no mocks.
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp
from hypertools.core.pipeline import build_pipeline
from hypertools.manip import Smooth
from hypertools.manip.common import Manipulator

TRAILING = {'model': 'Smooth',
            'kwargs': {'kernel': 'boxcar', 'center': False,
                       'kernel_width': 5}}
IMPUTATION_WARNINGS = ('Missing data', 'PPCA', 'Kalman')


def _walk(rows=40, cols=3, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


def _raises_manip_error_without_imputation_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError) as info:
            call()
    message = str(info.value)
    assert 'manip= stage' in message and 'Smooth' in message, message
    assert 'min_periods=1' in message, message
    assert 'no finite values' in message, message
    imputation = [str(w.message) for w in caught
                  if any(k in str(w.message) for k in IMPUTATION_WARNINGS)]
    assert imputation == [], imputation
    return message


def test_plot_reports_the_trailing_smoother_without_imputation_warnings():
    x = _walk()
    message = _raises_manip_error_without_imputation_warnings(
        lambda: hyp.plot(x, manip=TRAILING, show=False))
    assert '4 row(s) of dataset 0' in message
    assert 'rows 0-3' in message
    plt.close('all')


def test_plot_names_the_dataset_the_smoother_emptied():
    x = _walk()
    message = _raises_manip_error_without_imputation_warnings(
        lambda: hyp.plot([x, x[::-1]], manip=TRAILING, show=False))
    assert 'dataset 0' in message
    plt.close('all')


def test_a_scatter_plot_stops_at_the_manip_stage_too():
    """Markers do not need a finite line, but the reduce stage still
    received rows with no values -- the same misleading imputation
    warnings -- and drew a trajectory missing its first rows."""
    x = _walk()
    _raises_manip_error_without_imputation_warnings(
        lambda: hyp.plot(x, '.', manip=TRAILING, show=False))
    plt.close('all')


def test_an_instance_spec_is_named_too():
    x = _walk()
    _raises_manip_error_without_imputation_warnings(
        lambda: hyp.plot(x, manip=Smooth(kernel='boxcar', kernel_width=5,
                                         center=False), show=False))
    plt.close('all')


@pytest.mark.parametrize('entry', ['analyze', 'reduce', 'cluster', 'align'])
def test_every_manip_entry_point_raises_the_manip_error(entry):
    """These used to end in scikit-learn's 'Input X contains NaN' (reduce,
    cluster) or numpy's 'SVD did not converge' (align), after the same
    imputation warnings."""
    x, y = _walk(seed=1), _walk(seed=2)
    calls = {
        'analyze': lambda: hyp.analyze(x, manip=TRAILING, reduce='PCA',
                                       ndims=2),
        'reduce': lambda: hyp.reduce(x, manip=TRAILING, ndims=2),
        'cluster': lambda: hyp.cluster(x, manip=TRAILING, n_clusters=2),
        'align': lambda: hyp.align([x, y], manip=TRAILING),
    }
    _raises_manip_error_without_imputation_warnings(calls[entry])


def test_a_users_own_gap_is_imputed_and_the_emptied_rows_still_blamed():
    """`hyp.plot` fills the user's own missing entry at the format stage
    (warning about it, rightly), before the smoother runs; the rows the
    smoother then empties are still named as the manip stage's doing, and
    the imputation warning fires once -- for the user's gap only."""
    x = _walk()
    x[20, 1] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='manip= stage') as info:
            hyp.plot(x, manip=TRAILING, show=False)
    assert 'min_periods=1' in str(info.value)
    fills = [w for w in caught if 'Missing data' in str(w.message)]
    assert len(fills) == 1
    assert not [w for w in caught if 'PPCA cannot fill' in str(w.message)]
    plt.close('all')


def test_the_min_periods_remedy_works():
    x = _walk()
    spec = {'model': 'Smooth',
            'kwargs': {'kernel': 'boxcar', 'center': False,
                       'kernel_width': 5, 'min_periods': 1}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = hyp.analyze(x, manip=spec, reduce='PCA', ndims=2)
    out = np.asarray(out)
    assert out.shape == (40, 2) and np.isfinite(out).all()
    assert not [w for w in caught
                if any(k in str(w.message) for k in IMPUTATION_WARNINGS)]


def test_a_manip_only_pipeline_still_returns_the_rows():
    """With no stage after it, the manip result is the answer: its NaN rows
    are returned exactly as `hyp.manip` returns them (pandas semantics)."""
    x = _walk()
    out = np.asarray(hyp.analyze(x, manip=TRAILING))
    assert out.shape == x.shape
    assert np.isnan(out[:4]).all() and np.isfinite(out[4:]).all()


def test_partially_missing_rows_from_the_manip_stage_are_still_imputed():
    """Only rows with NO finite value are unrecoverable. A Delay embedding
    with drop_edges=False pads the lagged columns with NaN but keeps each
    row's own value, which the reduce stage's imputation can fill."""
    x = _walk(cols=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = hyp.analyze(x, manip={'model': 'Delay',
                                    'kwargs': {'tau': 1, 'dims': 2,
                                               'drop_edges': False}},
                          reduce='PCA', ndims=2)
    out = np.asarray(out)
    assert out.shape == (40, 2) and np.isfinite(out).all()


def _fit_range(data, **kwargs):
    frames = data if isinstance(data, list) else [data]
    return {'upper': np.max([np.nanmax(np.asarray(f, dtype=float))
                             for f in frames])}


def _blank_out_of_range(data, upper=None, **kwargs):
    """Blank every row with a value above the FIT-time maximum."""
    def one(frame):
        out = frame.astype(float).copy()
        out[(out > upper).any(axis=1)] = np.nan
        return out
    return [one(f) for f in data] if isinstance(data, list) else one(data)


class _OutOfRangeBlanker(Manipulator):
    """A real (if unusual) user manipulator: it learns the data's range at
    fit time and blanks rows of new data that leave it, so it empties rows
    only on `transform` -- the reuse path of a fitted pipeline."""

    def __init__(self):
        super().__init__(fitter=_fit_range, transformer=_blank_out_of_range,
                         required=['upper'])


def test_a_reused_pipeline_checks_the_manip_output_of_new_data():
    x = _walk(seed=6)
    pipe = build_pipeline(manip=_OutOfRangeBlanker(), reduce='PCA', ndims=2)
    fitted = np.asarray(pipe.fit_transform(x))
    assert np.isfinite(fitted).all()
    new = x.copy()
    new[7] = x.max() + 10.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='manip= stage') as info:
            pipe.transform(new)
    assert '1 row(s) of dataset 0' in str(info.value)
    assert 'row 7' in str(info.value)
    assert not [w for w in caught
                if any(k in str(w.message) for k in IMPUTATION_WARNINGS)]


def test_build_pipeline_fit_stops_at_the_manip_stage():
    """The check lives in the pipeline `build_pipeline` assembles, so a
    Pipeline built directly behaves like the dispatchers."""
    pipe = build_pipeline(manip=TRAILING, reduce='PCA', ndims=2)
    _raises_manip_error_without_imputation_warnings(
        lambda: pipe.fit_transform([_walk(seed=3), _walk(seed=4)]))
