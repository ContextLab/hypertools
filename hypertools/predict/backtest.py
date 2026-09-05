"""Shared scoring core for `hyp.predict(holdout=)` and `hyp.impute(truth=)`.

This module holds the metric definitions, the model-name resolution used by
the LIST/dict ``model=`` forms, the tidy scores-frame builder, and the
hold-out backtest driver for `hypertools.predict.predict`. Its numeric core
(`METRIC_FUNCS`, `resolve_metrics`, `score_pair`, `build_scores`) is
deliberately hypertools-free -- `hypertools.impute.backtest` imports it
rather than duplicating it, and it is the piece GH #285 proposes to lift
into a shared ``core/evaluate.py`` once a third caller appears.

Everything here is off the default path: nothing in this module runs unless
a caller passes ``holdout=``/``truth=`` or a collection of model specs.

What it replaces (GH #285, "Backtest / model-comparison scoring"): the
hand-rolled hold-out split, the naive last-value baseline, the MAE/MAPE
rows, the model x ticker pivot and the best-vs-naive verdict of
``docs/tutorials/stock_forecasting.ipynb`` cells 6-9, and the
occluded-cells-only per-axis error tables of
``docs/tutorials/projectile_kalman.ipynb`` cells 9 and 13.
"""
import warnings

import numpy as np
import pandas as pd


#: metric keys accepted by ``metrics=``, in the default order. The FIRST
#: entry is the RANKING metric (see `build_scores`' ``attrs['best']``).
DEFAULT_METRICS = ('mae', 'rmse', 'mape')

#: metric key -> the scores-frame column label it produces.
METRIC_LABELS = {'mae': 'MAE', 'rmse': 'RMSE', 'mape': 'MAPE'}

#: names a user model may NOT resolve to: they label the ground truth and
#: the baseline in the returned forecast/imputation dicts, so a collision
#: would silently overwrite one of them.
RESERVED_NAMES = ('truth',)


def _finite_pair(pred, truth):
    """The (predicted, actual) pairs where BOTH values are finite.

    Missing entries in either array are dropped rather than propagated: a
    single NaN in a held-out row would otherwise make every metric NaN and
    hide the rest of the comparison.
    """
    pred = np.asarray(pred, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    if pred.shape != truth.shape:
        raise ValueError(
            f'cannot score a prediction of {pred.size} value(s) against '
            f'{truth.size} actual value(s); they must line up one-to-one.')
    ok = np.isfinite(pred) & np.isfinite(truth)
    return pred[ok], truth[ok]


def mae(pred, truth):
    """Mean absolute error (NaN if nothing is scoreable)."""
    p, a = _finite_pair(pred, truth)
    return float(np.mean(np.abs(p - a))) if p.size else np.nan


def rmse(pred, truth):
    """Root mean squared error (NaN if nothing is scoreable)."""
    p, a = _finite_pair(pred, truth)
    return float(np.sqrt(np.mean((p - a) ** 2))) if p.size else np.nan


def mape(pred, truth):
    """Mean absolute PERCENTAGE error, in percent, NaN-safe on zeros.

    MAPE divides by the actual value, so entries whose truth is exactly 0
    are UNDEFINED. They are dropped from the average (rather than returning
    ``inf`` for the whole column, which is what the plain formula does);
    if every actual value is 0, the result is NaN.
    """
    p, a = _finite_pair(pred, truth)
    nonzero = a != 0
    if not nonzero.any():
        return np.nan
    return float(np.mean(np.abs((p[nonzero] - a[nonzero]) / a[nonzero])) * 100)


#: metric key -> callable(pred, truth) -> float
METRIC_FUNCS = {'mae': mae, 'rmse': rmse, 'mape': mape}


def resolve_metrics(metrics):
    """Normalize a ``metrics=`` argument into a tuple of metric keys."""
    if metrics is None:
        return DEFAULT_METRICS
    if isinstance(metrics, str):
        metrics = (metrics,)
    try:
        metrics = tuple(metrics)
    except TypeError as e:
        raise ValueError(
            f'metrics must be a metric name or a sequence of them; got '
            f'{metrics!r}. Supported: {", ".join(METRIC_FUNCS)}.') from e
    if len(metrics) == 0:
        raise ValueError('metrics is empty; pass at least one of '
                         f'{", ".join(METRIC_FUNCS)}.')
    resolved = []
    for m in metrics:
        if not isinstance(m, str) or m.lower() not in METRIC_FUNCS:
            raise ValueError(
                f'unknown metric {m!r}; supported: '
                f'{", ".join(METRIC_FUNCS)}.')
        resolved.append(m.lower())
    return tuple(resolved)


def score_pair(pred, truth, metrics):
    """One record's worth of scores.

    ``{'MAE': ..., ..., 'n': <pairs scored>, 'unscored': <pairs the model
    left NaN>}``. `unscored` counts only entries whose TRUTH is known and
    whose PREDICTION is missing -- values the model failed to produce (PPCA
    leaves fully-missing rows NaN, GH #169). They are counted rather than
    silently dropped: a model scored on fewer, easier entries than its
    neighbours would otherwise look better than it is.
    """
    p, a = _finite_pair(pred, truth)
    record = {METRIC_LABELS[m]: METRIC_FUNCS[m](pred, truth) for m in metrics}
    known = int(np.isfinite(np.asarray(truth, dtype=float).ravel()).sum())
    record['n'] = int(p.size)
    record['unscored'] = max(known - int(p.size), 0)
    return record


def canonical_name(name, valid, aliases=None):
    """The registry spelling of a model name (alias- and case-insensitive).

    Unknown names pass through unchanged -- resolving the SPEC is the
    dispatcher's job, and it raises the message that lists what is
    supported. This only decides what to CALL the model in the scores
    frame.
    """
    aliases = dict(aliases or {})
    if name in valid:
        return name
    if name in aliases:
        return aliases[name]
    lowered = {v.lower(): v for v in valid}
    lowered.update({k.lower(): v for k, v in aliases.items()})
    return lowered.get(name.lower(), name)


def spec_name(spec, valid=(), aliases=None):
    """The display name for ONE model spec.

    A string names itself (canonicalized against the registry); a dict
    spec is named by its inner ``'model'``; a class by its ``__name__``;
    an instance by its type's ``__name__``.
    """
    if isinstance(spec, str):
        return canonical_name(spec, valid, aliases)
    if isinstance(spec, dict) and 'model' in spec:
        return spec_name(spec['model'], valid, aliases)
    if isinstance(spec, type):
        return spec.__name__
    return type(spec).__name__


def unique_names(names):
    """Disambiguate repeated names as ``X``, ``X (2)``, ``X (3)``, ...

    Two entries of the same model (say the same forecaster with different
    keyword arguments) are a legitimate comparison, so repeats are
    numbered rather than rejected.
    """
    counts, out = {}, []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        out.append(name if counts[name] == 1 else f'{name} ({counts[name]})')
    return out


def model_collection(model, valid=(), aliases=None, caller='predict'):
    """Split a ``model=`` argument into ``(names, specs)`` if it is a
    COLLECTION of specs; return None if it is a single spec.

    A collection is a list/tuple of specs (auto-named from the specs), or a
    dict MAPPING NAMES to specs. A dict is only read as a mapping when it
    carries none of the single-spec keys (``model``/``args``/``kwargs``/
    ``params``), so ``{'model': 'Kalman', 'kwargs': {...}}`` stays a single
    spec and a malformed dict still gets the dispatcher's "must include a
    'model' key" error.
    """
    if isinstance(model, dict):
        if not model or model.keys() & {'model', 'args', 'kwargs', 'params'}:
            return None
        names = [str(k) for k in model]
        specs = list(model.values())
    elif isinstance(model, (list, tuple)):
        specs = list(model)
        if len(specs) == 0:
            raise ValueError(
                f'model=[] is empty; pass at least one model spec to '
                f'{caller}(), or a single spec (not a list) for the '
                'single-model form.')
        names = unique_names([spec_name(s, valid, aliases) for s in specs])
    else:
        return None

    for name in names:
        if name.lower() in RESERVED_NAMES:
            raise ValueError(
                f'model name {name!r} is reserved (it labels the ground '
                f'truth in {caller}\'s returned data); rename it with the '
                "mapping form, e.g. model={'my model': <spec>}.")
    if len(set(names)) != len(names):
        raise ValueError(f'duplicate model name(s) in model={model!r}; '
                         'names must be unique.')
    return names, specs


def build_scores(records, metrics, per_column=False, baseline=None,
                 extra=None, kind='value'):
    """Assemble the tidy scores frame from per-(model, dataset, column)
    records.

    Parameters
    ----------
    records : list of dict
        Each carries ``'model'``, optionally ``'dataset'``, ``'column'``,
        one entry per metric LABEL, and ``'n'``. Order matters: rows keep
        the order the records were generated in (models in the order the
        caller listed them, baseline last).
    metrics : tuple of metric keys
        The first is the RANKING metric behind ``attrs['best']``.
    per_column : bool
        False (default) averages each metric over the scored columns (and
        datasets) of a model, giving ONE row per model. True keeps the
        per-column rows, indexed by (model[, dataset], column).
    baseline : str or None
        The row name of the always-present baseline, excluded from the
        "best" verdict.
    extra : dict or None
        Extra scalar columns (e.g. ``{'horizon': 30}``) appended to both
        forms.

    Returns
    -------
    DataFrame with ``attrs`` describing the verdict (see `hyp.predict`).
    """
    labels = [METRIC_LABELS[m] for m in metrics]
    extra = dict(extra or {})
    long = pd.DataFrame.from_records(records)
    levels = [c for c in ('model', 'dataset', 'column') if c in long.columns]

    grouped = long.groupby('model', sort=False)
    wide = grouped[labels].mean()
    wide['n'] = grouped['n'].sum()
    wide['unscored'] = grouped['unscored'].sum()
    for name, unscored in wide['unscored'].items():
        # a model that failed to produce some values is scored on FEWER
        # entries than the others, so its row is not directly comparable;
        # say so instead of letting the smaller `n` pass unnoticed.
        if unscored:
            warnings.warn(
                f'model {name!r} left {int(unscored)} of '
                f'{int(unscored + wide.loc[name, "n"])} scored {kind}(s) '
                'missing (NaN); its scores cover only the ones it produced, '
                'so they are not directly comparable to models that '
                'produced every value.')
    for key, value in extra.items():
        wide[key] = value

    primary = labels[0]
    candidates = wide.drop(index=baseline, errors='ignore')[primary].dropna()
    best = str(candidates.idxmin()) if len(candidates) else None
    baseline_score = (float(wide.loc[baseline, primary])
                      if baseline in wide.index else np.nan)
    best_score = float(wide.loc[best, primary]) if best is not None else np.nan
    attrs = {
        'metric': primary,
        'metrics': list(labels),
        'baseline': baseline,
        'baseline_score': baseline_score,
        'best': best,
        'best_score': best_score,
        'beats_baseline': bool(np.isfinite(best_score)
                               and np.isfinite(baseline_score)
                               and best_score < baseline_score),
    }
    attrs.update(extra)

    if per_column:
        out = long.set_index(levels)[labels + ['n', 'unscored']]
        for key, value in extra.items():
            out[key] = value
    else:
        out = wide
    out.attrs.update(attrs)
    return out


def resolve_holdout(holdout, n, t, caller='predict'):
    """Turn ``holdout=`` into a number of held-out rows.

    ``True`` holds out exactly `t` rows; an int is a row count; a float in
    (0, 1) is a fraction of `n` (rounded, at least 1). The remaining head
    must keep at least 2 rows -- every forecaster needs 2 observations.
    """
    if isinstance(holdout, (bool, np.bool_)):
        if not holdout:
            raise ValueError('holdout=False does nothing; omit holdout= for '
                             'a plain forecast.')
        if not isinstance(t, (int, np.integer)) or isinstance(t, (bool, np.bool_)):
            raise ValueError(
                'holdout=True holds out exactly t rows, so t must be an '
                f'integer number of steps; got {t!r}. Pass the number of '
                'rows (or a fraction) as holdout= instead.')
        k = int(t)
    elif isinstance(holdout, (int, np.integer)):
        k = int(holdout)
    elif isinstance(holdout, (float, np.floating)):
        if not 0 < holdout < 1:
            raise ValueError(
                f'a float holdout is a FRACTION of the data and must be '
                f'between 0 and 1 (exclusive); got {holdout}. Pass an '
                'integer to hold out that many rows.')
        k = max(1, int(round(holdout * n)))
    else:
        raise ValueError(
            f'holdout must be an int (rows), a float in (0, 1) (fraction), '
            f'or True (hold out t rows); got {holdout!r}')
    if k < 1:
        raise ValueError(f'holdout must be >= 1 row; got {holdout!r}')
    if n - k < 2:
        raise ValueError(
            f'holding out {k} of {n} row(s) leaves {max(n - k, 0)} to fit '
            f'on; {caller} needs at least 2 observations (rows) of history. '
            'Use a smaller holdout, or more data.')
    return k


def naive_forecast(train, index):
    """The last-value-carried-forward baseline: `train`'s last OBSERVED
    value per column, repeated over `index`.

    ``ffill`` first, so a trailing NaN in one column carries that column's
    last real observation rather than making the whole baseline NaN.
    """
    last = train.ffill().iloc[-1]
    return pd.DataFrame(
        np.repeat(last.to_numpy(dtype=float)[None, :], len(index), axis=0),
        index=index, columns=train.columns)


def backtest_predict(datasets, predict_fn, t, holdout, names, specs,
                     metrics=None, per_column=False, return_forecasts=False,
                     kwargs=None):
    """Hold-out backtest for `hypertools.predict.predict` (see its docs).

    `datasets` is a list of wrangled DataFrames (a single dataset is a
    one-element list, flagged by `single` in the caller); `predict_fn` is
    the public `predict`, injected to keep this module import-free of the
    dispatcher.
    """
    metrics = resolve_metrics(metrics)
    kwargs = dict(kwargs or {})
    single = len(datasets) == 1

    baseline = 'naive'
    if any(name.lower() == baseline for name in names):
        raise ValueError(
            "model name 'naive' is reserved for the last-value baseline "
            "row; rename the model with the mapping form, e.g. "
            "model={'my naive model': <spec>}.")

    splits, horizons = [], []
    for d in datasets:
        k = resolve_holdout(holdout, len(d), t)
        splits.append((d.iloc[:-k], d.iloc[-k:]))
        horizons.append(k)

    forecasts = {}
    for name, spec in zip(names, specs):
        per_dataset = [predict_fn(train, model=spec, t=k, **kwargs)
                       for (train, _), k in zip(splits, horizons)]
        forecasts[name] = per_dataset
    forecasts[baseline] = [naive_forecast(train, held.index)
                           for (train, held) in splits]

    records = []
    for name in list(names) + [baseline]:
        for i, (forecast, (_, held)) in enumerate(zip(forecasts[name], splits)):
            predicted = np.asarray(forecast, dtype=float)
            actual = np.asarray(held, dtype=float)
            if predicted.shape != actual.shape:
                raise ValueError(
                    f'model {name!r} forecast {predicted.shape} rows/columns '
                    f'but {actual.shape} were held out; they must match to '
                    'be scored.')
            for j, column in enumerate(held.columns):
                record = {'model': name}
                if not single:
                    record['dataset'] = i
                record['column'] = column
                record.update(score_pair(predicted[:, j], actual[:, j],
                                         metrics))
                records.append(record)

    horizon = horizons[0] if len(set(horizons)) == 1 else float(
        np.mean(horizons))
    scores = build_scores(records, metrics, per_column=per_column,
                          baseline=baseline, extra={'horizon': horizon},
                          kind='forecast value')
    if not return_forecasts:
        return scores
    out = {name: (f[0] if single else f) for name, f in forecasts.items()}
    out['truth'] = splits[0][1] if single else [held for _, held in splits]
    return scores, out
