# -*- coding: utf-8 -*-
"""Regression tests for reported GitHub issues."""

import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp


def test_issue_264_fresh_results_in_loops():
    """GH #264: plots inside a for loop all looked like the first plot.

    Root cause was the memoize cache: its str()-based keys truncated numpy
    arrays, so new data hashed to the same key and returned the first
    call's (stale) transformed data. With the cache removed, each
    iteration must reflect its own data.
    """
    drawn = []
    for seed in range(3):
        data = np.cumsum(
            np.random.default_rng(seed).standard_normal((80, 5)), axis=0)
        geo = hyp.plot(data, show=False)
        line = geo.ax.get_lines()[0]
        drawn.append(np.column_stack(line.get_data_3d()))
        plt.close('all')

    # every iteration must produce different rendered coordinates
    assert not np.allclose(drawn[0], drawn[1])
    assert not np.allclose(drawn[1], drawn[2])


def test_issue_264_reduce_fresh_in_loops():
    """The same staleness affected hyp.reduce directly."""
    results = []
    for seed in range(3):
        data = np.random.default_rng(seed).standard_normal((60, 10))
        results.append(hyp.reduce(data, ndims=3))
    assert not np.allclose(results[0], results[1])
    assert not np.allclose(results[1], results[2])


def test_issue_265_animate_numpy2():
    """GH #265: animate=True raised `np.string_ was removed in NumPy 2.0`.

    Reproduces the exact array construction from the issue report and
    asserts the animation is created under numpy >= 2.
    """
    assert int(np.__version__.split('.')[0]) >= 2, \
        'regression test requires numpy >= 2'
    arr = np.array([[math.sin(3 * i / 100), math.cos(3 * i / 100),
                     (i / 100) ** 2, (i / 100) ** 3, 1 / (1 + i / 100)]
                    for i in range(0, 300)])
    geo = hyp.plot(arr, animate=True, show=False)
    assert geo.line_ani is not None
    plt.close('all')


def test_issue_259_rcparams_untouched():
    """GH #259: plotting permanently mutated global matplotlib rcParams."""
    before = {k: str(v) for k, v in plt.rcParams.items()}
    hyp.plot(np.random.default_rng(0).standard_normal((50, 4)), show=False)
    plt.close('all')
    changed = {k for k in before if str(plt.rcParams[k]) != before[k]}
    assert not changed, f'plot() mutated rcParams: {changed}'


def test_corrupt_dataset_cache_recovers():
    """CI failure 2026-07-02 (macos/py3.11): Google Drive rate-limiting
    returned an HTML page that was cached as the dataset, poisoning every
    subsequent text-data test with UnpicklingError. load() must detect the
    corrupt cache, delete it, and re-download.
    """
    from hypertools.tools.load import DATA_DIR
    from hypertools._shared.exceptions import HypertoolsIOError

    target = DATA_DIR / 'spiral'
    DATA_DIR.mkdir(exist_ok=True)
    # what Google Drive actually returns when rate-limited
    target.write_bytes(b'<!DOCTYPE html><html>quota exceeded</html>')

    try:
        geo = hyp.load('spiral')
    except HypertoolsIOError:
        # the re-download itself was rate-limited: the essential guarantee
        # is that the poisoned cache was removed so a later attempt can
        # succeed instead of failing forever
        assert (not target.is_file()
                or target.read_bytes()[:1] != b'<')
    else:
        assert geo is not None
        assert target.read_bytes()[:1] != b'<'  # cache healed
