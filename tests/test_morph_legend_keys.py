"""An ``animate='morph'`` legend shows its keys (1.2.0 review, round 3;
also wrong in 1.1.0): the morphing datasets' own artists are hidden for
the whole animation, and matplotlib copied that onto the legend keys, so
the legend was a column of labels with nothing beside them.

Real animations; the keys are read from the real legend and looked for in
the rendered pixels.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

import hypertools as hyp

DATA = [np.random.default_rng(i).normal(size=(60, 5)) + i for i in range(3)]

CASES = {
    'single-3d': {},
    'single-2d': dict(ndims=2),
    'grid': dict(panels=(1, 3),
                 reduce=['PCA', 'FastICA', 'IncrementalPCA']),
}


@pytest.fixture(autouse=True)
def _close_figures():
    """Leave no figure behind."""
    yield
    plt.close('all')


def _legend(fig):
    """The figure's one legend (figure-level for a grid)."""
    found = list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                 if ax.get_legend() is not None]
    assert len(found) == 1
    return found[0]


def _pixels_near(fig, legend, color):
    """How many pixels inside `legend`'s box are within 0.08 of `color`
    in the figure as rendered."""
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255.0
    box = legend.get_window_extent()
    height = image.shape[0]
    crop = image[int(height - box.y1):int(height - box.y0),
                 int(box.x0):int(box.x1)]
    return int((np.abs(crop - np.asarray(color)).max(axis=2) < 0.08).sum())


@pytest.mark.parametrize('case', list(CASES))
@pytest.mark.parametrize('frame', [0, 3, -1])
def test_morph_legend_keys_show_the_marker_in_its_color(case, frame):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        anim = hyp.plot(DATA, '.', animate='morph', legend=True, duration=2,
                        frame_rate=5, show=False, **CASES[case])
        reference = hyp.plot(DATA, '.', legend=True, show=False,
                             **{k: v for k, v in CASES[case].items()
                                if k == 'ndims'})
    anim.draw_frame(frame % anim.n_frames)
    fig = anim.figure
    legend = _legend(fig)
    assert [t.get_text() for t in legend.get_texts()] == ['1', '2', '3']
    # the static plot of the same datasets keys them in the same colors
    wanted = [to_rgb(h.get_color())
              for h in _legend(reference).legend_handles]
    assert len(set(wanted)) == 3
    for handle, color in zip(legend.legend_handles, wanted):
        assert handle.get_visible()
        assert handle.get_marker() == '.'
        assert handle.get_markersize() > 0
        assert to_rgb(handle.get_color()) == color
        assert _pixels_near(fig, legend, color) >= 4, (
            f'no key of color {color} is drawn in the legend')
