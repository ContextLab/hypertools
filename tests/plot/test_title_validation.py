import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pytest

import hypertools as hyp


def _datasets(n=3, rows=10, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def test_title_string_is_rendered():
    fig = hyp.plot(_datasets(), '-', title='My Title', show=False)
    assert _ax(fig).get_title() == 'My Title'


def test_title_none_leaves_axes_untitled():
    fig = hyp.plot(_datasets(), '-', title=None, show=False)
    assert _ax(fig).get_title() == ''


@pytest.mark.parametrize('bad', [['a', 'b', 'c'], ('a', 'b', 'c'), 3, {'a': 1}])
def test_non_string_title_raises_rather_than_stringifying(bad):
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(_datasets(), '-', title=bad, show=False)


def test_title_error_names_the_alternatives():
    with pytest.raises(TypeError, match='names='):
        hyp.plot(_datasets(), '-', title=['a', 'b', 'c'], show=False)


def test_title_is_rejected_before_the_analyze_pipeline_runs():
    """Fail-fast (plot.py:423-430): the title error must beat the reduce error.

    `reduce='NoSuchReducer'` raises `ValueError: unknown reduce model ...`
    from inside analyze(), which `plot()` calls at plot.py:2804 -- far after
    the validation anchor at plot.py:2231. If validation were placed after
    _resolve_animate_mode (plot.py:3653) this test would see the ValueError.
    """
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(_datasets(), '-', title=['a', 'b', 'c'],
                 reduce='NoSuchReducer', show=False)


def test_stream_input_also_rejects_a_list_title():
    """plot_stream returns at plot.py:2582 and forwards `title` verbatim
    (plot.py:2555), so validation placed after that line never sees it.
    Measured before this task: renders the title "['a', 'b']", no warning."""
    rng = np.random.default_rng(0)
    stream = (rng.normal(size=4) for _ in range(40))
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(stream, '-', title=['a', 'b'], stream_max=20, show=False)


# --- animate='spin'/'window' + order='serial' + a title list: minor finding
# (whole-branch review): this combination used to clear the fail-fast check
# above (raw order='serial' alone satisfied it, regardless of whether the
# STYLE could ever honor a serial ordering), run the whole analyze/reduce
# pipeline, emit `_resolve_animate_mode`'s own "animate='spin' has no
# serial ordering ...; ignoring order='serial'" warning, and only THEN
# raise TypeError -- whose generic message advised "order='serial' ...",
# which is exactly what the caller had already passed. The error must both
# name the REAL reason and fire fail-fast, before any warning or pipeline
# work, exactly like every other title validation in this file.

def test_spin_order_serial_title_list_names_the_real_reason():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(TypeError,
                           match="animate='spin' has no serial ordering"):
            hyp.plot(_datasets(), '-', animate='spin', order='serial',
                     title=['a', 'b', 'c'], show=False)
    assert not caught, (
        "must raise fail-fast, before _resolve_animate_mode's own "
        "'ignoring order=serial' warning fires; saw "
        f"{[str(w.message) for w in caught]}")


def test_spin_order_serial_title_list_is_rejected_before_the_analyze_pipeline_runs():
    """Fail-fast companion, mirroring
    test_title_is_rejected_before_the_analyze_pipeline_runs above: an
    otherwise-fatal reduce= must never be reached."""
    with pytest.raises(TypeError,
                       match="animate='spin' has no serial ordering"):
        hyp.plot(_datasets(), '-', animate='spin', order='serial',
                 title=['a', 'b', 'c'], reduce='NoSuchReducer', show=False)


def test_window_order_serial_title_list_also_names_the_real_reason():
    """'window' is the other order='serial'-incapable style; same fix."""
    with pytest.raises(TypeError,
                       match="animate='window' has no serial ordering"):
        hyp.plot(_datasets(), '-', animate='window', order='serial',
                 title=['a', 'b', 'c'], show=False)


def test_order_serial_without_animation_and_title_list_keeps_its_own_error():
    """Regression guard: an UNANIMATED plot (animate=False, the default)
    with order='serial' and a title list must keep raising the PRE-EXISTING,
    more specific ValueError ("order='serial' requires an animated plot"),
    not get swept into the spin/window-specific TypeError branch above --
    that branch only applies to a TRUTHY (animated) style that cannot honor
    a serial ordering."""
    with pytest.raises(ValueError, match="order='serial' requires an "
                                         "animated plot"):
        hyp.plot(_datasets(), '-', order='serial', title=['a', 'b', 'c'],
                 show=False)
