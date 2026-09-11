"""Alignment quality scoring (GH #285): `alignment_score` computes a single
number summarizing how well a list of equal-shape datasets agree with each
other, before/after alignment.

`metric='dispersion'` reproduces, EXACTLY, the `dispersion()` helper in
`examples/plot_story_trajectories.py` (mean distance of each dataset to the
cross-dataset centroid at each timepoint, divided by the overall cloud
scale) -- so the example can be rewritten to call this function directly and
print the identical numbers. `metric='isc'` computes the classic
hyperalignment diagnostic: mean pairwise inter-subject correlation, per
feature, averaged over features (and, for >2 datasets, over the
n-choose-2 pairs).
"""
import numpy as np

__all__ = ['alignment_score']


def _stack_equal_shape(datasets, fname):
    """Convert `datasets` (a list of array-likes) to one `(n_datasets, n_obs,
    n_features)` numpy stack, raising a clear `ValueError` if the datasets do
    not all share the same shape (alignment scoring requires the SAME
    observations/features across datasets -- comparing a centroid or a
    per-feature correlation across ragged datasets is meaningless)."""
    arrays = [np.asarray(d) for d in datasets]
    if len(arrays) == 0:
        raise ValueError(f'{fname} requires at least one dataset; got an empty list.')
    for i, a in enumerate(arrays):
        if a.ndim != 2:
            raise ValueError(
                f'{fname} requires 2-D datasets of shape (n_observations, '
                f'n_features); dataset {i} has shape {a.shape}. Reshape a '
                '1-D series to a single column (x[:, None]) first.')
        if not np.issubdtype(a.dtype, np.number):
            raise ValueError(
                f'{fname} requires numeric datasets; dataset {i} has dtype '
                f'{a.dtype}.')
        if not np.all(np.isfinite(a)):
            raise ValueError(
                f'{fname} requires finite values; dataset {i} has '
                f'{int((~np.isfinite(a)).sum())} NaN/inf entries. Impute or '
                'drop them (e.g. hyp.impute) before scoring.')
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        raise ValueError(
            f'{fname} requires all datasets to have the same shape (they '
            'must already be aligned/trimmed to common observations and '
            f'features); got shapes {[a.shape for a in arrays]}. Run '
            "hyp.align(...) (or trim/pad the datasets yourself) so every "
            'dataset has the same number of rows and columns before '
            'scoring.')
    return np.stack(arrays)  # (n_datasets, n_obs, n_features)


def dispersion(trajectories):
    """Mean distance of the datasets to their shared centroid, averaged over
    observations and divided by the overall cloud scale (so it is
    comparable before and after alignment).

    Reproduces `examples/plot_story_trajectories.py`'s `dispersion()`
    helper EXACTLY (same computation, same result for the same input) --
    kept as the library implementation of that example's inline function.

    Raises `ValueError` when every dataset is constant (each one's
    observations all at one point, a single observation included), naming
    the datasets. Then the per-observation centroid is the same point as
    the cloud's own mean, so the score is exactly 1.0 whatever the datasets
    are (or, when they all sit at the SAME point, 0/0) -- it measures
    nothing, matching `metric='isc'`, which has no correlation to compute
    there either.
    """
    stack = np.stack([np.asarray(t) for t in trajectories])   # (subj, t, d)
    # a dataset is constant when its observations never move (np.ptp is 0
    # along the observation axis for every feature)
    constant = np.all(np.ptp(stack, axis=1) == 0, axis=1)     # (subj,)
    if constant.all():
        # release review 2026-09-07 caught the all-at-one-point case (NaN
        # with a RuntimeWarning); 1.1 release review 2026-09-11 the
        # each-at-its-own-point case, which returned a meaningless 1.0
        # while 'isc' raised on the same input
        n_datasets, n_obs = stack.shape[0], stack.shape[1]
        which = (f'dataset {n_datasets - 1}' if n_datasets == 1
                 else f'datasets 0-{n_datasets - 1}' if n_datasets > 3
                 else ', '.join(f'dataset {i}' for i in range(n_datasets)))
        single = (' (a dataset with a single observation is constant)'
                  if n_obs == 1 else '')
        raise ValueError(
            "alignment_score(metric='dispersion') is undefined when every "
            f'dataset is constant: {which} each keep every observation at '
            f'one point{single}. The per-observation centroid is then the '
            "cloud's own mean, so the score would be 1.0 (0/0 when the "
            'datasets share one point) whatever the alignment. Score '
            'datasets whose observations vary.')
    centroid = stack.mean(axis=0, keepdims=True)
    spread = np.linalg.norm(stack - centroid, axis=2).mean()
    scale = np.linalg.norm(stack - stack.mean(axis=(0, 1)), axis=2).mean()
    return spread / scale


def _isc(datasets):
    """Mean pairwise inter-subject correlation, per feature, averaged over
    features (and over dataset pairs when there are more than two datasets)
    -- the classic hyperalignment diagnostic: for each feature (column),
    correlate every pair of datasets' timecourses for that feature across
    observations, then average all the (feature, pair) correlations into a
    single number in [-1, 1]."""
    stack = _stack_equal_shape(datasets, 'alignment_score')  # (n, obs, feat)
    n_datasets, n_obs, n_features = stack.shape
    if n_datasets < 2:
        raise ValueError(
            "alignment_score(metric='isc') requires at least 2 datasets to "
            f'compute pairwise correlations; got {n_datasets}.')
    if n_obs < 2:
        raise ValueError(
            "alignment_score(metric='isc') requires at least 2 observations "
            f'(rows) per dataset to compute a correlation; got {n_obs}.')
    correlations = []
    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            for f in range(n_features):
                x = stack[i, :, f]
                y = stack[j, :, f]
                if np.std(x) == 0 or np.std(y) == 0:
                    # a constant feature has an undefined correlation;
                    # exclude it rather than injecting a NaN into the mean
                    continue
                correlations.append(np.corrcoef(x, y)[0, 1])
    if not correlations:
        raise ValueError(
            "alignment_score(metric='isc') could not compute any pairwise "
            'correlation: every feature was constant across observations '
            'in at least one dataset.')
    return float(np.mean(correlations))


_METRICS = {
    'dispersion': dispersion,
    'isc': _isc,
}


def alignment_score(datasets, aligned=None, metric='dispersion'):
    """Score how well a list of equal-shape datasets agree with each other,
    optionally comparing before vs. after alignment.

    Parameters
    ----------
    datasets : list of array-likes
        The (pre-alignment) datasets, all sharing the same shape
        `(n_observations, n_features)`.
    aligned : list of array-likes, or None
        The same datasets after alignment (e.g. `hyp.align(datasets)`'s
        result), also all sharing one common shape (not necessarily the
        same shape as `datasets`, e.g. after zero-padding to a different
        common column count). If `None` (default), only the `'before'`
        score is computed and `'after'` is `None`.
    metric : {'dispersion', 'isc'}
        Which score to compute:

        - `'dispersion'`: mean distance of the datasets to their shared
          centroid at each observation, averaged over observations and
          divided by the overall cloud scale (lower means the datasets
          agree more). Reproduces
          `examples/plot_story_trajectories.py`'s `dispersion()` helper
          exactly.
        - `'isc'`: mean pairwise inter-subject correlation, per feature,
          averaged over features and dataset pairs (higher means the
          datasets agree more); always in `[-1, 1]`.

        (default: `'dispersion'`).

    Returns
    -------
    score : dict
        `{'before': float, 'after': float or None, 'metric': str}`.

    Raises
    ------
    ValueError
        If `datasets` (or `aligned`) is empty, if any dataset is not a 2-D
        numeric array of finite values, if the datasets in either list do
        not all share the same shape (ragged input), if `metric` is not one
        of the supported names, or if the input is degenerate for the
        metric (every dataset constant for `'dispersion'`; no non-constant
        feature to correlate for `'isc'`).
    """
    if metric not in _METRICS:
        raise ValueError(
            f'unknown alignment_score metric {metric!r}; supported: '
            f"{', '.join(sorted(_METRICS))}.")
    scorer = _METRICS[metric]

    _stack_equal_shape(datasets, 'alignment_score')  # validate shape (before)
    before = scorer(datasets)

    after = None
    if aligned is not None:
        _stack_equal_shape(aligned, 'alignment_score')  # validate shape (after)
        after = scorer(aligned)

    return {'before': before, 'after': after, 'metric': metric}
