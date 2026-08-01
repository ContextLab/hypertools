import matplotlib
matplotlib.use("Agg")

import time

import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.plot import MORPH_SAMPLES_REQUIRED_ABOVE


def _clouds(n_points, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n_points, 3)) + off for off in (0.0, 5.0)]


def test_threshold_constant_is_2000():
    """Matches the ~2000-point recommendation already in plot()'s
    morph_samples docstring (plot.py:1519-1522)."""
    assert MORPH_SAMPLES_REQUIRED_ABOVE == 2000


def test_simplify_false_over_the_threshold_raises_naming_simplify():
    """Would otherwise be a >20-minute pytest-timeout kill, not an error.

    The message must name the escape hatch, not just the problem: the
    maintainer asked for "an informative message with a suggestion to set
    simplify=True", carried BY the exception (see the plan's Task 3 prose).
    """
    start = time.monotonic()
    with pytest.raises(ValueError, match='simplify=True'):
        hyp.plot(_clouds(12000), '.', animate='morph', simplify=False,
                 duration=1, frame_rate=2, show=False)
    assert time.monotonic() - start < 60, 'the guard must fire before matching'


def test_the_error_reports_the_actual_cloud_size_and_names_morph_samples():
    with pytest.raises(ValueError, match='12000'):
        hyp.plot(_clouds(12000), '.', animate='morph', simplify=False,
                 duration=1, frame_rate=2, show=False)
    with pytest.raises(ValueError, match='morph_samples'):
        hyp.plot(_clouds(12000), '.', animate='morph', simplify=False,
                 duration=1, frame_rate=2, show=False)


def test_default_simplify_downsamples_silently_above_the_threshold():
    """The DEFAULT path above the cap: it renders, it is capped at
    MORPH_SAMPLES_REQUIRED_ABOVE, and it says NOTHING.

    The maintainer was explicit that simplify=True must not warn. Any
    warnings.warn or print added here is a contract violation, so assert the
    absence of BOTH -- and assert the plot actually drew, so a silent
    no-render cannot pass this test.
    """
    import warnings
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(_clouds(12000), '.', animate='morph',
                            duration=1, frame_rate=2, show=False)
    assert caught == [], f'simplify=True must be silent; got {caught}'
    assert time.monotonic() - start < 60, 'the cap must apply before matching'
    ani._func(0, *ani._args)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    drawn = max(len(ln.get_data_3d()[0]) for ln in ax.lines)
    assert 0 < drawn <= MORPH_SAMPLES_REQUIRED_ABOVE


def test_default_simplify_prints_nothing_above_the_threshold(capsys):
    """"Silently" also means nothing on stdout/stderr: the informative
    message belongs to the simplify=False raise, not to this path."""
    hyp.plot(_clouds(12000), '.', animate='morph',
             duration=1, frame_rate=2, show=False)
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err == ''


def test_simplify_is_a_no_op_below_the_threshold():
    """Below the cap, `simplify` has NO effect whatsoever -- this pins the
    ordinary-morph default path as untouched by this task.

    All three spellings must draw the same point counts, and none may warn.
    """
    import warnings
    drawn = {}
    for label, kwargs in (('default', {}),
                          ('true', {'simplify': True}),
                          ('false', {'simplify': False})):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig, ani = hyp.plot(_clouds(300), '.', animate='morph',
                                duration=1, frame_rate=2, show=False,
                                **kwargs)
        assert caught == [], f'{label} warned below the cap: {caught}'
        ani._func(0, *ani._args)
        ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
        assert len(ax.lines) >= 1
        drawn[label] = sorted(len(ln.get_data_3d()[0]) for ln in ax.lines)
    assert drawn['default'] == drawn['true'] == drawn['false']


def test_clouds_at_or_below_the_threshold_keep_every_point():
    """The morph.py:17-24 full-sample guarantee holds unconditionally below
    the bar: no cap, no warning, and every one of the 300 points kept."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(_clouds(300), '.', animate='morph',
                            duration=1, frame_rate=2, show=False)
    assert not [w for w in caught if 'morph_samples' in str(w.message)]
    ani._func(0, *ani._args)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    assert len(ax.lines) >= 1
    assert max(len(ln.get_data_3d()[0]) for ln in ax.lines) == 300


def test_non_boolean_simplify_is_rejected_fail_fast():
    """Validated at plot.py:2231 with the other raw-argument checks
    (Contract 8), so it fires before the analyze/reduce pipeline."""
    with pytest.raises(TypeError, match='simplify'):
        hyp.plot(_clouds(50), '.', animate='morph', simplify='yes',
                 duration=1, frame_rate=2, show=False)


def test_explicit_morph_samples_is_respected_above_the_threshold():
    """The explicit opt-in still works and still draws. `simplify` never
    engages when the caller has already chosen a cap, so this holds for
    BOTH values of the flag."""
    for kwargs in ({}, {'simplify': False}):
        fig, ani = hyp.plot(_clouds(12000), '.', animate='morph',
                            morph_samples=400, duration=1, frame_rate=2,
                            show=False, **kwargs)
        ani._func(0, *ani._args)
        ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
        drawn = max(len(ln.get_data_3d()[0]) for ln in ax.lines)
        assert 0 < drawn <= 400


def test_only_morph_tagged_datasets_are_measured():
    """A per-dataset animate list (plot.py:480-505) morphs only the tagged
    datasets, so a huge UNTAGGED backdrop must not trip the guard.

    Driven with simplify=False so the guard is in its RAISING mode: under the
    default it would not raise regardless, which would make this vacuous.
    """
    rng = np.random.default_rng(0)
    small = [rng.normal(size=(200, 3)) + off for off in (0.0, 5.0)]
    big_backdrop = rng.normal(size=(12000, 3)) + 10.0
    hyp.plot(small + [big_backdrop], '.',
             animate=['morph', 'morph', None], simplify=False,
             duration=1, frame_rate=2, show=False)


def test_plotly_backend_applies_the_same_guard():
    """Backend parity: the check lives in plot.py, above both dispatches
    (plot.py:4239 plotly / plot.py:4324 matplotlib), so both the raise and
    the silent cap behave identically."""
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        with pytest.raises(ValueError, match='simplify=True'):
            hyp.plot(_clouds(12000), '.', animate='morph', simplify=False,
                     duration=1, frame_rate=2, show=False)
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            hyp.plot(_clouds(12000), '.', animate='morph',
                     duration=1, frame_rate=2, show=False)
        assert caught == []
    finally:
        hyp.set_interactive_backend('matplotlib')
