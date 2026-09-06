"""``panels=True`` picks its grid from the figure's aspect ratio and prefers
grids without spare cells (1.1 feature-tour report: three panels came out
2x2 with a hole, and in a 9x3.2 in figure each square 3-D axes shrank to
the short cell height). Real figures, measured axes positions."""

import matplotlib.pyplot as plt
import pytest

import hypertools as hyp
from hypertools.plot.plot import _resolve_panel_grid


@pytest.mark.parametrize('n, size, expected', [
    (3, (9.0, 3.2), (1, 3)),     # wide figure: a row
    (3, (6.4, 4.8), (1, 3)),     # default figure: a row beats 2x2-with-a-hole
    (3, (3.0, 9.0), (3, 1)),     # tall figure: a column
    (2, (6.4, 4.8), (1, 2)),
    (4, (6.4, 4.8), (2, 2)),
    (6, (6.4, 4.8), (2, 3)),
    (5, (6.4, 4.8), (2, 3)),     # no hole-free grid is close enough: 2x3
    (8, (9.0, 3.2), (2, 4)),
])
def test_auto_grid_follows_the_figure_aspect(n, size, expected):
    assert _resolve_panel_grid(True, n, size=size) == expected
    assert _resolve_panel_grid('auto', n, size=size) == expected


def test_explicit_grids_are_untouched_by_size():
    assert _resolve_panel_grid((2, 2), 3, size=(9.0, 3.2)) == (2, 2)
    assert _resolve_panel_grid(2, 3, size=(9.0, 3.2)) == (2, 2)


def _walks():
    return [hyp.load('random_walk', n_samples=60, n_features=8, random_state=s)
            for s in range(3)]


def test_three_panels_in_a_wide_figure_fill_one_row():
    fig = hyp.plot(_walks(), '.', panels=True, size=[9, 3.2],
                   title=['walk 0', 'walk 1', 'walk 2'], show=False)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3 and len(fig.axes) == 3       # no hidden spare
    boxes = [ax.get_position() for ax in visible]
    assert len({round(b.y0, 2) for b in boxes}) == 1      # one row
    assert min(b.width for b in boxes) > 0.25             # each ~a third wide
    assert min(b.height for b in boxes) > 0.7
    plt.close(fig)


def test_three_panels_by_default_fill_one_row_too():
    fig = hyp.plot(_walks(), '.', panels=True, show=False)
    assert len(fig.axes) == 3
    assert len({round(ax.get_position().y0, 2) for ax in fig.axes}) == 1
    plt.close(fig)
