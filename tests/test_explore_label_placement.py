"""explore=True hover labels stay inside the figure near every edge.

The label used to sit up and to the LEFT of every point, so hovering a point
near the left edge drew a label cut off by the window (seen in the native Qt
screencast of the feature tour's GUI-native case, 2026-10-08). Hover is
exercised through matplotlib's own event dispatch with a real MouseEvent.
"""
import warnings

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.mplot3d import proj3d

import hypertools as hyp


def _explore_figure():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # non-interactive backend notice
        fig = hyp.plot(hyp.load('helix'), explore=True, show=False)
    fig.canvas.draw()
    return fig


def _screen_points(ax):
    x, y, z = ax.lines[0].get_data_3d()
    xs, ys, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    return ax.transData.transform(np.column_stack([xs, ys]))


@pytest.mark.parametrize('edge', ['left', 'right', 'bottom', 'top'])
def test_hover_label_stays_inside_figure(edge):
    fig = _explore_figure()
    ax = fig.axes[0]
    pts = _screen_points(ax)
    pick = {'left': np.argmin(pts[:, 0]), 'right': np.argmax(pts[:, 0]),
            'bottom': np.argmin(pts[:, 1]), 'top': np.argmax(pts[:, 1])}[edge]
    # move somewhere else first so the edge point is a NEW closest point
    other = pts[len(pts) // 2]
    for target in (other, pts[pick]):
        event = MouseEvent('motion_notify_event', fig.canvas, *target)
        fig.canvas.callbacks.process('motion_notify_event', event)
    labels = [t for t in ax.texts if t.get_visible() and t.get_text()]
    assert labels, 'hovering a data point drew no label'
    fig.canvas.draw()
    box = labels[-1].get_window_extent(fig.canvas.get_renderer())
    fig_box = fig.bbox
    assert box.x0 >= fig_box.x0 and box.x1 <= fig_box.x1, (edge, box, fig_box)
    assert box.y0 >= fig_box.y0 and box.y1 <= fig_box.y1, (edge, box, fig_box)
