"""Imputer comparison scoring for `hyp.impute(truth=)`.

Scores each imputer on the DAMAGED CELLS ONLY -- the entries that were NaN
in the input -- against a complete `truth` array, using the shared metric
core in `hypertools.predict.backtest` (GH #285 proposes lifting that core
into ``core/evaluate.py`` once a third caller appears; until then it lives
next to its first caller and is imported here rather than duplicated).

Off the default path: nothing here runs unless a caller passes ``truth=``
or a collection of imputer specs.

What it replaces: the hand-rolled masked per-axis RMSE of
``docs/tutorials/projectile_kalman.ipynb`` cell 9 and the
scattered-vs-occluded imputer comparison of its cell 13.
"""
import numpy as np
import pandas as pd

from ..predict.backtest import build_scores, resolve_metrics, score_pair


#: the always-present imputation baseline: fill each column with the mean of
#: its OBSERVED values (what `SimpleImputer` does by default, and what a
#: row-gap forces every cross-column imputer down to).
BASELINE = 'mean'


def _as_frame(x, like, what):
    """Coerce `truth`/`mask`-shaped input to a DataFrame matching `like`."""
    if isinstance(x, pd.DataFrame):
        frame = x
    else:
        values = np.asarray(x)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError(
                f'{what} must be 2-D (n_observations, n_features); got shape '
                f'{values.shape}')
        frame = pd.DataFrame(values, index=like.index, columns=like.columns)
    if frame.shape != like.shape:
        raise ValueError(
            f'{what} has shape {frame.shape} but the data being imputed has '
            f'shape {like.shape}; they must match cell for cell.')
    return frame


def _mean_fill(data):
    """The column-mean baseline fill for one damaged dataset.

    A column with NO observed values has no mean; it is filled with 0.0 --
    the same placeholder `PPCA`/`Kalman` fall back to (`hyp.impute` already
    warns about such columns).
    """
    values = np.asarray(data, dtype=float)
    observed = ~np.isnan(values)
    counts = observed.sum(axis=0)
    # np.divide with `where=` rather than np.nanmean: nanmean warns ("Mean of
    # empty slice") and returns NaN on an all-missing column, which would
    # then propagate into every metric for the baseline row.
    means = np.divide(np.where(observed, values, 0.0).sum(axis=0), counts,
                      out=np.zeros(values.shape[1], dtype=float),
                      where=counts > 0)
    filled = np.where(np.isnan(values), means[None, :], values)
    return pd.DataFrame(filled, index=data.index, columns=data.columns)


def imputer_collection(model, valid=(), caller='hyp.impute'):
    """`hypertools.predict.backtest.model_collection`, for imputers.

    Kept here rather than shared because the reserved-name set differs:
    `impute`'s baseline row is ``'mean'``, not ``'naive'``.
    """
    from ..predict.backtest import model_collection
    collection = model_collection(model, valid=valid, caller=caller)
    if collection is None:
        return None
    names, _specs = collection
    for name in names:
        if name.lower() == BASELINE:
            raise ValueError(
                f"model name {name!r} is reserved for the column-mean "
                'baseline row; rename it with the mapping form, e.g. '
                "model={'my mean imputer': <spec>}.")
    return collection


def score_imputations(datasets, impute_fn, names, specs, truth, mask=None,
                      metrics=None, per_column=False, return_imputed=False,
                      kwargs=None):
    """Score imputers on the damaged cells of `datasets` (see `hyp.impute`).

    `datasets` is a list of wrangled DataFrames (a single dataset is a
    one-element list); `impute_fn` is the public `impute`, injected to keep
    this module import-free of the dispatcher.
    """
    metrics = resolve_metrics(metrics)
    kwargs = dict(kwargs or {})
    single = len(datasets) == 1

    truths = truth if isinstance(truth, (list, tuple)) else [truth]
    if len(truths) != len(datasets):
        raise ValueError(
            f'truth has {len(truths)} dataset(s) but {len(datasets)} were '
            'passed to impute(); pass one complete dataset per input '
            'dataset.')
    truths = [_as_frame(x, d, 'truth') for x, d in zip(truths, datasets)]

    masks = []
    if mask is None:
        user_masks = [None] * len(datasets)
    elif isinstance(mask, (list, tuple)):
        user_masks = list(mask)
        if len(user_masks) != len(datasets):
            raise ValueError(
                f'mask has {len(user_masks)} dataset(s) but {len(datasets)} '
                'were passed to impute(); pass one mask per input dataset.')
    else:
        user_masks = [mask] * len(datasets)
    for d, m in zip(datasets, user_masks):
        missing = np.isnan(np.asarray(d, dtype=float))
        if m is not None:
            # a caller-supplied mask RESTRICTS scoring (e.g. "the occluded
            # band only"); observed cells are excluded regardless, since
            # every imputer passes those through untouched and scoring them
            # would flatter every model equally.
            missing = missing & _as_frame(m, d, 'mask').to_numpy().astype(bool)
        masks.append(missing)
    if not any(m.any() for m in masks):
        raise ValueError(
            'nothing to score: no missing (NaN) entries in the data' +
            ('' if mask is None else ' fall inside mask=') +
            '. Imputation scores compare the DAMAGED cells against truth.')

    imputed = {}
    for name, spec in zip(names, specs):
        results = impute_fn(datasets if not single else datasets[0],
                            model=spec, **kwargs)
        imputed[name] = results if isinstance(results, list) else [results]
    imputed[BASELINE] = [_mean_fill(d) for d in datasets]

    records = []
    for name in list(names) + [BASELINE]:
        for i, (filled, actual, missing) in enumerate(
                zip(imputed[name], truths, masks)):
            values = np.asarray(filled, dtype=float)
            if values.shape != missing.shape:
                raise ValueError(
                    f'model {name!r} returned data of shape {values.shape} '
                    f'but the input had shape {missing.shape}; imputation '
                    'must preserve shape to be scored.')
            actual_values = np.asarray(actual, dtype=float)
            for j, column in enumerate(actual.columns):
                cells = missing[:, j]
                record = {'model': name}
                if not single:
                    record['dataset'] = i
                record['column'] = column
                record.update(score_pair(values[cells, j],
                                         actual_values[cells, j], metrics))
                records.append(record)

    scores = build_scores(records, metrics, per_column=per_column,
                          baseline=BASELINE, kind='damaged cell')
    if not return_imputed:
        return scores
    out = {name: (f[0] if single else f) for name, f in imputed.items()}
    out['truth'] = truths[0] if single else truths
    return scores, out
