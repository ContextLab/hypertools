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


# --- fix-pass: list-case precedence (task-6 review, Important finding) ----
#
# The two tests above only ever passed a *scalar* alpha=0.9, so the
# precedence path was never exercised with a *list* -- which is exactly how
# the regression below escaped review: `_validate_alpha` ran eagerly at the
# early write site, BEFORE these branches decide whether they will
# override alpha, so a wrong-length or non-numeric list RAISED instead of
# being silently superseded (with a warning) like a scalar already was.

def test_multiindex_level_fading_wins_with_list_alpha_and_says_so():
    """List form of test_multiindex_level_fading_wins_and_says_so: a
    same-length, individually-valid list must still lose to MultiIndex
    fading and warn, exactly like the scalar case."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(_multiindex_frame(), '-',
                       alpha=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9], show=False)
    assert [w for w in caught
            if 'alpha' in str(w.message) and 'MultiIndex' in str(w.message)]
    alphas = [a for a in _alphas(fig) if a is not None]
    assert not all(a == pytest.approx(0.9) for a in alphas)


def test_multiindex_level_fading_wins_over_wrong_length_alpha_without_raising():
    """THE confirmed regression (task-6 review, Important finding):
    `hyp.plot(multiindex_df, '-', alpha=[0.1, 0.2, 0.3])` (3 values, 6
    leaves) succeeded silently at db02c64e (pre-task-6: alpha was a plain
    **kwargs passthrough, dropped by the internal-wins-over-extra-kwarg
    rule) and raised ValueError('alpha has 3 entries but there are 6
    datasets to plot') at 76c4f27f. It must warn and be ignored, not
    raise: whether alpha will be used has to be decided before it is
    validated, since MultiIndex fading overrides it unconditionally."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(_multiindex_frame(), '-',
                       alpha=[0.1, 0.2, 0.3], show=False)
    assert [w for w in caught
            if 'alpha' in str(w.message) and 'MultiIndex' in str(w.message)]
    alphas = [a for a in _alphas(fig) if a is not None]
    assert alphas, 'expected drawn lines with a resolved (internal) alpha'
    assert all(0.0 <= a <= 1.0 for a in alphas)


def test_multiindex_level_fading_wins_over_non_numeric_alpha_without_raising():
    """Second half of the same confirmed regression:
    `alpha=['a', 'b', 'c']` also succeeded silently at db02c64e and raised
    ValueError('alpha must be a number...') at 76c4f27f. Must warn and be
    ignored, not raise -- and the *internal* (numeric) alpha must be what
    reaches matplotlib, not the raw user strings (which would themselves
    crash matplotlib if they ever leaked through)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(_multiindex_frame(), '-',
                       alpha=['a', 'b', 'c'], show=False)
    assert [w for w in caught
            if 'alpha' in str(w.message) and 'MultiIndex' in str(w.message)]
    alphas = [a for a in _alphas(fig) if a is not None]
    assert alphas, 'expected drawn lines with a resolved (internal) alpha'
    assert all(0.0 <= a <= 1.0 for a in alphas)


def test_nested_list_depth_fading_wins_and_says_so():
    """plot.py:3629 writes a depth-derived alpha list for nested inputs."""
    rng = np.random.default_rng(0)
    nested = [[rng.normal(size=(10, 4)).cumsum(axis=0) for _ in range(2)],
              rng.normal(size=(10, 4)).cumsum(axis=0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(nested, '-', alpha=0.9, show=False)
    assert [w for w in caught if 'alpha' in str(w.message)]


def test_nested_list_depth_fading_wins_over_invalid_list_alpha_without_raising():
    """List-shaped analog of test_nested_list_depth_fading_wins_and_says_so
    -- the nested-input half of the task-6 review's Important finding
    ('Same for nested varying-depth input'): a wrong-length alpha list
    must warn and be ignored, not raise, once depth fading is known to
    override it (mirrors the two MultiIndex regression tests above)."""
    rng = np.random.default_rng(0)
    nested = [[rng.normal(size=(10, 4)).cumsum(axis=0) for _ in range(2)],
              rng.normal(size=(10, 4)).cumsum(axis=0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(nested, '-', alpha=[0.1, 0.2, 0.3, 0.4, 0.5],
                       show=False)
    assert [w for w in caught if 'alpha' in str(w.message)]
    alphas = [a for a in _alphas(fig) if a is not None]
    assert alphas, 'expected drawn lines with a resolved (internal) alpha'
    assert all(0.0 <= a <= 1.0 for a in alphas)


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


def _clusterable_datasets(n=2, rows=20, dims=4, seed=0, shift=12.0):
    """Two datasets, each with a sharp mid-trajectory jump so a 2-cluster
    KMeans fit assigns cluster labels that are contiguous within each
    dataset (first half one cluster, second half the other): the discrete
    (non-mixture) clustering + line-format branch (plot.py, the
    `elif _fmt_draws_line(fmt):` sibling of the `_mixture_name(model) in
    mixture_models` check) segments each dataset into contiguous
    same-cluster runs, same as hue= does above."""
    rng = np.random.default_rng(seed)
    half = rows // 2
    out = []
    for _ in range(n):
        first = rng.normal(loc=0.0, scale=0.5, size=(half, dims))
        second = rng.normal(loc=shift, scale=0.5, size=(rows - half, dims))
        out.append(np.vstack([first, second]))
    return out


def test_alpha_survives_cluster_run_segmentation():
    """cluster= analog of test_alpha_survives_contiguous_run_segmentation
    (task-6 review, Minor finding: the suite covered hue= run-expansion
    but not cluster=, even though both go through the same
    `_expand_styles_to_runs` call at plot.py ~3597). The reviewer manually
    verified this works (KMeans, 2 datasets -> 3 runs, alphas {0.15,
    0.85}); this is that same check made automatic."""
    ds = _clusterable_datasets(n=2, rows=20)
    fig = hyp.plot(ds, '-', cluster='KMeans', n_clusters=2,
                   alpha=[0.15, 0.85], show=False)
    alphas = [a for a in (ln.get_alpha() for ln in _ax(fig).lines)
              if a is not None]
    assert set(np.round(alphas, 6)) <= {0.15, 0.85}
    assert len(alphas) > 2, 'expected more runs than datasets'


# --- fix-pass 2: the lookahead read `hue` before animate='morph' nulled it
# (task-6 second review, NEW ISSUE) -----------------------------------------
#
# `_alpha_overridden_internally`'s nested-list arm snapshotted `hue` at its
# (former) write site, but `animate='morph'` (or a per-dataset list
# `animate=`) nulls `hue` LATER, immediately before the
# MultiIndex/cluster/hue/nested_groups chain picks its branch. A lookahead
# evaluated before that null judges the nested-list arm against the
# PRE-null `hue` (non-None -> arm False -> eager validation), disagreeing
# with the chain's actual POST-null choice (hue None -> nested_groups arm
# fires -> depth fading overrides alpha). The fix moved the lookahead (and
# the validate/write it guards) to right before the chain, after the
# hue-drop, so it always sees the FINAL `hue`.

def _varying_depth_nested(seed=0):
    """Same fixture as test_nested_list_depth_fading_wins_and_says_so /
    test_nested_list_depth_fading_wins_over_invalid_list_alpha_without_raising
    above: 3 leaves (2 at depth 2 under one outer group, 1 at depth 1 under
    another), so nested_groups is set and depths vary."""
    rng = np.random.default_rng(seed)
    return [[rng.normal(size=(10, 4)).cumsum(axis=0) for _ in range(2)],
            rng.normal(size=(10, 4)).cumsum(axis=0)]


def test_nested_list_depth_fading_wins_with_hue_and_morph_over_invalid_alpha_without_raising():
    """THE confirmed regression (task-6 second review, NEW ISSUE):
    nested_groups + hue=<array> + animate='morph' + a bad-length alpha
    list RAISED ValueError('alpha has 5 entries but there are 3 datasets
    to plot') at 8d089c23 (the lookahead read `hue` before the
    animate='morph' hue-drop nulled it, so it wrongly believed the
    nested-list branch would not fire) but SUCCEEDED SILENTLY at db02c64e
    (pre-task-6: alpha was an unvalidated, unconditionally-overwritten
    **kwargs entry) -- confirmed live in a scratch worktree at db02c64e.
    Must warn (for both the dropped hue and the overridden alpha) and
    apply the depth-derived alpha, not raise."""
    nested = _varying_depth_nested()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(nested, '-', hue=np.array([0.2, 0.5, 0.9]),
                            animate='morph',
                            alpha=[0.1, 0.2, 0.3, 0.4, 0.5],
                            duration=1, show=False)
    messages = [str(w.message) for w in caught]
    assert any("hue is not supported with animate='morph'" in m
               for m in messages), messages
    assert any('nested list with varying nesting depth' in m
               for m in messages), messages
    alphas = [a for a in (ln.get_alpha() for ln in _ax(fig).lines)
              if a is not None]
    assert alphas, 'expected drawn lines with a resolved (internal) alpha'
    assert all(0.0 <= a <= 1.0 for a in alphas)


def test_nested_list_depth_fading_wins_with_hue_and_morph_over_non_numeric_alpha_without_raising():
    """Non-numeric sibling of the test above (mirrors the MultiIndex
    branch's own scalar/list-non-numeric pairing)."""
    nested = _varying_depth_nested()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(nested, '-', hue=np.array([0.2, 0.5, 0.9]),
                            animate='morph', alpha=['a', 'b', 'c'],
                            duration=1, show=False)
    messages = [str(w.message) for w in caught]
    assert any("hue is not supported with animate='morph'" in m
               for m in messages), messages
    assert any('nested list with varying nesting depth' in m
               for m in messages), messages
    alphas = [a for a in (ln.get_alpha() for ln in _ax(fig).lines)
              if a is not None]
    assert alphas, 'expected drawn lines with a resolved (internal) alpha'
    assert all(0.0 <= a <= 1.0 for a in alphas)


def test_nested_list_depth_fading_wins_with_hue_and_valid_alpha_under_morph():
    """Positive-control sibling: a VALID (correct-length, in-range) list
    alpha must still lose to depth fading under morph, exactly like an
    invalid one does above -- mirrors
    test_multiindex_level_fading_wins_with_list_alpha_and_says_so for the
    MultiIndex branch. Guards against a fix that only special-cases
    invalid alpha instead of the general override."""
    nested = _varying_depth_nested()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(nested, '-', hue=np.array([0.2, 0.5, 0.9]),
                            animate='morph', alpha=[0.9, 0.9, 0.9],
                            duration=1, show=False)
    assert [w for w in caught if 'nested list with varying nesting depth'
            in str(w.message)]
    alphas = [a for a in (ln.get_alpha() for ln in _ax(fig).lines)
              if a is not None]
    assert alphas
    assert not all(a == pytest.approx(0.9) for a in alphas)


def test_nested_list_depth_fading_wins_with_hue_and_list_animate_without_raising():
    """Sibling of the primary regression test exercising the OTHER
    disjunct of the hue-drop's own gate (`(animate == 'morph') or
    isinstance(animate, list)`, plot.py): a per-dataset list `animate=`
    also nulls hue LATER than the (now-relocated) lookahead reads it, so
    the fix must generalise rather than special-case the literal string
    'morph'."""
    nested = _varying_depth_nested()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(nested, '-', hue=np.array([0.2, 0.5, 0.9]),
                            animate=['morph', None, 'morph'],
                            alpha=[0.1, 0.2, 0.3, 0.4, 0.5],
                            duration=1, show=False)
    messages = [str(w.message) for w in caught]
    assert any("hue is not supported with animate='morph'" in m
               for m in messages), messages
    assert any('nested list with varying nesting depth' in m
               for m in messages), messages
    alphas = [a for a in (ln.get_alpha() for ln in _ax(fig).lines)
              if a is not None]
    assert alphas
    assert all(0.0 <= a <= 1.0 for a in alphas)


def test_cluster_still_wins_over_depth_fading_and_bad_length_alpha_raises():
    """Over-correction guard (task-6 second review, fix-pass instructions):
    the fix must not make `cluster=` cases look internally-overridden.
    `_alpha_overridden_internally`'s nested-list arm requires `cluster is
    None and n_clusters is None`, so cluster= must still win the elif
    chain over nested-list depth fading (even though nested_groups is
    set) and a bad-length alpha must still raise fast, exactly as the
    reviewer manually verified for the non-nested case in the original
    task-6 review."""
    nested = _varying_depth_nested()
    with pytest.raises(ValueError, match='alpha has 5 entries'):
        hyp.plot(nested, '-', cluster='KMeans', n_clusters=2,
                 alpha=[0.1, 0.2, 0.3, 0.4, 0.5], show=False)


def test_cluster_still_wins_over_depth_fading_and_non_numeric_alpha_raises():
    """Non-numeric sibling of the test above."""
    nested = _varying_depth_nested()
    with pytest.raises(ValueError, match='alpha must be a number'):
        hyp.plot(nested, '-', cluster='KMeans', n_clusters=2,
                 alpha=['a', 'b', 'c'], show=False)


def test_cluster_still_wins_over_depth_fading_with_valid_alpha():
    """Positive control: cluster= is NOT an override branch (only
    MultiIndex and nested-list depth fading are), so a VALID list alpha
    must actually apply (widened to run length by
    `_expand_styles_to_runs`, like test_alpha_survives_cluster_run_
    segmentation above), not be silently dropped."""
    nested = _varying_depth_nested()
    fig = hyp.plot(nested, '-', cluster='KMeans', n_clusters=2,
                   alpha=[0.2, 0.4, 0.6], show=False)
    alphas = [a for a in (ln.get_alpha() for ln in _ax(fig).lines)
              if a is not None]
    assert alphas
    assert set(np.round(alphas, 6)) <= {0.2, 0.4, 0.6}
