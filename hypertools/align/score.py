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
    """
    stack = np.stack([np.asarray(t) for t in trajectories])   # (subj, t, d)
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
        If `datasets` (or `aligned`) is empty, if the datasets in either
        list do not all share the same shape (ragged input), or if
        `metric` is not one of the supported names.
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
