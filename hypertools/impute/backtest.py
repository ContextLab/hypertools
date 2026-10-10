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
import copy

import numpy as np
import pandas as pd

from .._shared.helpers import (as_pandas_dataframe, is_frame_dataset,
                               is_series_like)
from ..predict.backtest import build_scores, resolve_metrics, score_pair
from .common import Imputer


#: the always-present imputation baseline: fill each column with the mean of
#: its OBSERVED values (what `SimpleImputer` does by default, and what a
#: row-gap forces every cross-column imputer down to).
BASELINE = 'mean'


def label_flags(x):
    """Which axes of ONE dataset carry labels: ``(rows, columns)``.

    A pandas DataFrame (or anything with its API) labels both axes -- a
    default ``0..n-1`` index included, since a reordered frame carries
    those integers with it. A pandas Series labels its rows only (it is one
    column). Other dataframes (polars) name their columns but have no row
    labels. Arrays and everything else are unlabeled.
    """
    if is_frame_dataset(x):
        import datawrangler as dw
        return bool(dw.zoo.dataframe_like(x)), True
    if is_series_like(x):
        # a pandas Series carries a row index; a polars Series has none
        index = getattr(x, 'index', None)
        return index is not None and not callable(index), False
    return False, False


def data_label_flags(raw, n_datasets):
    """`label_flags` for each dataset the caller passed to `impute`.

    `raw` is the caller's ``data`` BEFORE wrangling (wrangling turns every
    dataset into a labeled pandas frame, erasing the difference between
    an array and a frame).
    """
    if isinstance(raw, (list, tuple)) and len(raw) == n_datasets:
        return [label_flags(d) for d in raw]
    if n_datasets == 1:
        return [label_flags(raw)]
    return [(False, False)] * n_datasets


def _show(labels, limit=8):
    labels = list(labels)
    shown = ', '.join(repr(v) for v in labels[:limit])
    more = f', ... ({len(labels)} in all)' if len(labels) > limit else ''
    return f'[{shown}{more}]'


def _align_axis(frame, target, axis, what):
    """Reorder `frame` along `axis` to `target`'s label order.

    Identical label sequences need nothing. The same SET of unique labels
    in another order is reindexed. Anything else cannot be aligned
    unambiguously and raises, naming the axis and the offending labels.
    """
    own = frame.axes[axis]
    if own.equals(target):
        return frame
    kind = 'column' if axis else 'row'
    hint = (f' Labeled {what} is aligned to the data by label; to compare '
            'by position instead, pass it as a bare array '
            f'(e.g. {what.split("[")[0]}.to_numpy()).')
    duplicated = []
    for index, owner in ((own, what), (target, 'the data')):
        if not index.is_unique:
            duplicated.append(
                f'duplicated in {owner}: '
                f'{_show(index[index.duplicated()].unique())}')
    if duplicated:
        raise ValueError(
            f'cannot align {what} to the data by {kind} labels: the labels '
            'differ and some are repeated, so the pairing is ambiguous '
            f'({"; ".join(duplicated)}).' + hint)
    missing = target.difference(own, sort=False)
    unexpected = own.difference(target, sort=False)
    if len(missing) or len(unexpected):
        parts = []
        if len(missing):
            parts.append(f'in the data but missing from {what}: '
                         f'{_show(missing)}')
        if len(unexpected):
            parts.append(f'in {what} but not in the data: '
                         f'{_show(unexpected)}')
        raise ValueError(
            f"{what} {kind} labels do not match the data's "
            f'({"; ".join(parts)}).' + hint)
    return frame.reindex(target, axis=axis)


def _as_frame(x, like, what, labelled=(False, False)):
    """Coerce `truth`/`mask`-shaped input to a DataFrame matching `like`.

    `labelled` says which axes of the DATA carry labels (`label_flags`).
    On an axis where both the data and `x` are labeled, `x` is aligned to
    the data BY LABEL (release red-team 2026-10-09: a truth frame with its
    columns or rows in another order used to be compared cell-by-position
    and silently mis-scored). Otherwise the comparison is positional.
    """
    rows, columns = label_flags(x)
    if rows and not columns:               # a row-labeled Series
        frame = x.to_frame()
    elif is_frame_dataset(x):
        frame = as_pandas_dataframe(x)     # any backend datawrangler knows
    else:
        values = np.asarray(x)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError(
                f'{what} must be 2-D (n_observations, n_features); got shape '
                f'{values.shape}')
        if values.shape != like.shape:
            # checked BEFORE borrowing the data's labels: pandas would
            # otherwise raise its own "indices imply" error first
            raise ValueError(
                f'{what} has shape {values.shape} but the data being imputed '
                f'has shape {like.shape}; they must match cell for cell.')
        frame = pd.DataFrame(values, index=like.index, columns=like.columns)
    if frame.shape != like.shape:
        raise ValueError(
            f'{what} has shape {frame.shape} but the data being imputed has '
            f'shape {like.shape}; they must match cell for cell.')
    if rows and labelled[0]:
        frame = _align_axis(frame, like.index, 0, what)
    if columns and labelled[1]:
        frame = _align_axis(frame, like.columns, 1, what)
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
                      kwargs=None, labelled=None):
    """Score imputers on the damaged cells of `datasets` (see `hyp.impute`).

    `datasets` is a list of wrangled DataFrames (a single dataset is a
    one-element list); `impute_fn` is the public `impute`, injected to keep
    this module import-free of the dispatcher. `labelled` is
    `data_label_flags` of the caller's un-wrangled data (None: treat every
    dataset as unlabeled, i.e. compare `truth`/`mask` by position).
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
    if labelled is None:
        labelled = [(False, False)] * len(datasets)

    def tag(what, i):
        """How error messages name dataset `i`'s truth/mask."""
        return what if single else f'{what}[{i}]'

    truths = [_as_frame(x, d, tag('truth', i), flags)
              for i, (x, d, flags) in enumerate(zip(truths, datasets,
                                                    labelled))]

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
    shared_mask = not isinstance(mask, (list, tuple))
    for i, (d, m, flags) in enumerate(zip(datasets, user_masks, labelled)):
        missing = np.isnan(np.asarray(d, dtype=float))
        if m is not None:
            # a caller-supplied mask RESTRICTS scoring (e.g. "the occluded
            # band only"); observed cells are excluded regardless, since
            # every imputer passes those through untouched and scoring them
            # would flatter every model equally.
            aligned = _as_frame(
                m, d, 'mask' if shared_mask else tag('mask', i), flags)
            missing = missing & aligned.to_numpy().astype(bool)
        masks.append(missing)
    if not any(m.any() for m in masks):
        raise ValueError(
            'nothing to score: no missing (NaN) entries in the data' +
            ('' if mask is None else ' fall inside mask=') +
            '. Imputation scores compare the DAMAGED cells against truth.')

    imputed = {}
    for name, spec in zip(names, specs):
        # GH #285 release review: score a fresh fit on damaged data, never
        # learned state that may already contain the hidden truth.
        candidate = spec.get('model') if isinstance(spec, dict) else spec
        if isinstance(candidate, Imputer) and candidate.is_fitted:
            raise ValueError(
                'truth= scoring requires an unfitted model so hidden values '
                'cannot leak into training; pass a model name, class, or '
                'unfitted instance instead.')
        results = impute_fn(datasets if not single else datasets[0],
                            model=copy.deepcopy(spec), **copy.deepcopy(kwargs))
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
