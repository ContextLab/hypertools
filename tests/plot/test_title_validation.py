import matplotlib
matplotlib.use("Agg")

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
