"""A static 3-D plot whose cube needs no fitting is drawn exactly as
matplotlib lays it out (1.2.1 gallery review).

The cube fit of GH #297 (`plot._keep_cube_clear`) applied its starting
zoom, 1, to every static 3-D plot through `Axes3D.set_box_aspect`, which
rescales the aspect it is handed. An axes given its own aspect back came
out one unit in the last place away from it, and three gallery figures
(`plot_missing_data` twice, `plot_multiindex`) differed from 1.2.0 in 2
to 7 antialiased pixels, each by one level of one channel.
"""
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import hypertools as hyp  # noqa: E402


def _multiindex_frame():
    """The data of `examples/plot_multiindex.py`."""
    rng = np.random.default_rng(3)
    n_time = 60
    t = np.linspace(0, 4 * np.pi, n_time)
    tuples, rows = [], []
    for ci, cond in enumerate(['baseline', 'treatment']):
        for si in range(4):
            noise = rng.standard_normal((n_time, 3)) * 0.15
            helix = np.column_stack([np.cos(t), np.sin(t), t / 4]) + noise
            helix[:, 0] += ci * 3.0
            rows.append(helix)
            tuples.extend([(cond, f'subj{si}')] * n_time)
    index = pd.MultiIndex.from_tuples(tuples, names=['condition', 'subject'])
    return pd.DataFrame(np.vstack(rows), columns=['x', 'y', 'z'], index=index)


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _default_box_aspect():
    """The box aspect matplotlib gives a new 3-D axes."""
    fig = plt.figure()
    aspect = np.asarray(fig.add_subplot(projection='3d').get_box_aspect())
    plt.close(fig)
    return aspect


@pytest.mark.parametrize('kwargs', [
    {},
    {'legend': ['a', 'b']},
    {'legend': ['a', 'b'], 'title': 'A title'},
    {'legend': ['a', 'b'], 'colorbar': True},
])
def test_static_cube_that_fits_keeps_matplotlibs_box_aspect(kwargs):
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((40, 3)).cumsum(axis=0) for _ in range(2)]
    fig = hyp.plot(data, show=False, **kwargs)
    aspect = np.asarray(fig.axes[0].get_box_aspect())
    # exact: one unit in the last place is the whole defect
    assert aspect.tolist() == _default_box_aspect().tolist()


def test_static_cube_that_fits_is_drawn_as_matplotlib_lays_it_out():
    """The pixels of `examples/plot_multiindex.py`'s figure are those of
    the same figure with matplotlib's own default box aspect."""
    fig = hyp.plot(_multiindex_frame(), legend=True, show=False)
    drawn = _pixels(fig)
    fig.axes[0].set_box_aspect(None)
    reference = _pixels(fig)
    differing = int((drawn != reference).any(axis=-1).sum())
    assert differing == 0, f'{differing} pixels differ'


def test_cube_under_a_corner_legend_is_still_zoomed_out():
    """The fit still acts where it is needed: a legend in a corner of the
    axes has the cube drawn smaller."""
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((40, 3)).cumsum(axis=0) for _ in range(10)]
    fig = hyp.plot(data, legend=[f'dataset {i}' for i in range(10)],
                   legend_kwargs={'loc': 'upper left'}, show=False)
    aspect = np.asarray(fig.axes[0].get_box_aspect())
    assert (aspect < _default_box_aspect() * 0.999).all()
