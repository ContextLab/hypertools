"""The plotly `panels=` grid is laid out like the matplotlib one (1.1
release review, feature-tour section 9.8): square, centred 3-D cells with
`tight_layout`-sized gaps, room above each titled row, a camera that is
backed off only in a cell narrower than it is tall -- and a cube that fills
its cell within a few pixels of the matplotlib panel's.

The parity test renders both backends for real (matplotlib Agg +
plotly/kaleido, the `tests/test_marker_parity.py` pattern) and measures
the ink in every cell. No mocks.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.plotly_backend import (PANEL_AXIS_GAP_PX, PANEL_GAP_PX,
                                            PANEL_TITLE_PX,
                                            SCENE_CUBE_WIDTH_PER_HEIGHT)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walk(seed=0):
    return hyp.load('random_walk', n_samples=60, n_features=8,
                    random_state=seed)


def _cells_px(fig):
    """Every cell's pixel box ``(x0, y0, x1, y1)`` (y down), in layout
    order, from its scene/xaxis+yaxis domains."""
    layout = fig.layout
    mg = layout.margin
    plot_w = layout.width - mg.l - mg.r
    plot_h = layout.height - mg.t - mg.b
    boxes = []
    # a 2-D figure's layout still iterates an (empty) 'scene'
    keys = sorted((k for k in layout if k.startswith('scene')
                   and layout[k].domain.x is not None),
                  key=lambda k: int(k[5:] or 1))
    if keys:
        for key in keys:
            d = layout[key].domain
            boxes.append((d.x[0], d.y[0], d.x[1], d.y[1]))
    else:
        xkeys = sorted((k for k in layout if k.startswith('xaxis')),
                       key=lambda k: int(k[5:] or 1))
        for xk in xkeys:
            yk = 'yaxis' + xk[5:]
            dx, dy = layout[xk].domain, layout[yk].domain
            boxes.append((dx[0], dy[0], dx[1], dy[1]))
    return [(mg.l + x0 * plot_w, mg.t + (1 - y1) * plot_h,
             mg.l + x1 * plot_w, mg.t + (1 - y0) * plot_h)
            for x0, y0, x1, y1 in boxes]


def _eye_distance(scene):
    eye = scene.camera.eye
    return float(np.sqrt(eye.x ** 2 + eye.y ** 2 + eye.z ** 2))


def test_3d_cells_are_square_with_tight_gaps_and_no_back_off():
    fig = hyp.plot([_walk(i) for i in range(3)], panels=True, show=False,
                   backend='plotly')
    cells = _cells_px(fig)
    assert len(cells) == 3
    for x0, y0, x1, y1 in cells:
        assert abs((x1 - x0) - (y1 - y0)) <= 2.0
    for left, right in zip(cells, cells[1:]):
        assert right[0] - left[2] == pytest.approx(PANEL_GAP_PX, abs=1.5)
    # centred: the same slack above and below
    top = cells[0][1]
    bottom = fig.layout.height - cells[0][3]
    assert abs(top - bottom) <= 2.0
    # a square cell needs no camera back-off: the single figure's own view
    single = hyp.plot(_walk(0), show=False, backend='plotly')
    for key in ('scene', 'scene2', 'scene3'):
        assert _eye_distance(fig.layout[key]) == pytest.approx(
            _eye_distance(single.layout.scene), rel=1e-6)


def test_titled_rows_reserve_a_title_line():
    fig = hyp.plot([_walk(i) for i in range(4)], panels=True, show=False,
                   title=['a', 'b', 'c', 'd'], backend='plotly')
    cells = _cells_px(fig)
    # row 1 starts a gap plus one title line below row 0
    assert cells[2][1] - cells[0][3] == pytest.approx(
        PANEL_GAP_PX + PANEL_TITLE_PX, abs=1.5)
    assert fig.layout.margin.t >= 40
    titles = [a for a in fig.layout.annotations
              if a.name and a.name.startswith('hyp-cell-title-')]
    assert [a.text for a in titles] == ['a', 'b', 'c', 'd']


def test_2d_cells_fill_the_figure_with_axis_gaps():
    fig = hyp.plot([_walk(i) for i in range(4)], ndims=2, panels=True,
                   show=False, backend='plotly')
    cells = _cells_px(fig)
    assert cells[0][0] == pytest.approx(fig.layout.margin.l, abs=1.0)
    assert cells[1][2] == pytest.approx(
        fig.layout.width - fig.layout.margin.r, abs=1.0)
    assert cells[1][0] - cells[0][2] == pytest.approx(PANEL_AXIS_GAP_PX,
                                                     abs=1.5)
    assert cells[2][1] - cells[0][3] == pytest.approx(PANEL_AXIS_GAP_PX,
                                                     abs=1.5)


def test_a_cell_narrower_than_tall_backs_the_camera_off():
    fig, cells = hyp.subplots(1, 2, backend='plotly',
                              column_widths=[0.25, 0.75])
    hyp.plot(_walk(0), ax=cells[0], show=False, backend='plotly')
    hyp.plot(_walk(1), ax=cells[1], show=False, backend='plotly')
    narrow, wide = _cells_px(fig)
    narrow_w, narrow_h = narrow[2] - narrow[0], narrow[3] - narrow[1]
    assert narrow_w < narrow_h
    single = hyp.plot(_walk(0), show=False, backend='plotly')
    base = _eye_distance(single.layout.scene)
    expected = SCENE_CUBE_WIDTH_PER_HEIGHT * narrow_h / narrow_w
    assert _eye_distance(fig.layout.scene) == pytest.approx(base * expected,
                                                            rel=1e-3)
    assert _eye_distance(fig.layout.scene2) == pytest.approx(base, rel=1e-6)


def _ink_box(gray, box):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    sub = gray[y0:y1, x0:x1]
    ys, xs = np.where(sub < 200)
    assert len(xs), 'an empty cell'
    return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


@pytest.mark.parametrize('n,kw', [
    pytest.param(3, {}, id='1x3'),
    pytest.param(4, {}, id='2x2'),
    pytest.param(3, dict(size=[9, 3.2], title=['a', 'b', 'c']),
                 id='1x3-wide-titled'),
])
def test_cube_fills_its_cell_like_the_matplotlib_panel(n, kw, tmp_path):
    """Real renders of the same grid on both backends: in every cell the
    drawn cube (the ink's bounding box) is the matplotlib panel's width
    within 6 %, so the plotly grid neither shrinks its cubes nor spaces
    them out (the pre-review grid drew them ~35 % narrower)."""
    from PIL import Image
    data = [_walk(i) for i in range(n)]
    mpl_fig = hyp.plot(data, panels=True, show=False, backend='matplotlib',
                       **kw)
    mpl_path = tmp_path / 'mpl.png'
    mpl_fig.savefig(mpl_path, dpi=100)
    width, height = (mpl_fig.get_size_inches() * 100).astype(int)
    mpl_gray = np.asarray(Image.open(mpl_path).convert('L'))
    mpl_widths = []
    for ax in mpl_fig.axes:
        x0, y0, w, h = ax.get_position().bounds
        mpl_widths.append(_ink_box(
            mpl_gray, (x0 * width, (1 - y0 - h) * height,
                       (x0 + w) * width, (1 - y0) * height))[0])

    pl_fig = hyp.plot(data, panels=True, show=False, backend='plotly', **kw)
    pl_path = tmp_path / 'plotly.png'
    pl_fig.write_image(str(pl_path))
    pl_gray = np.asarray(Image.open(pl_path).convert('L'))
    assert pl_gray.shape[::-1] == (pl_fig.layout.width, pl_fig.layout.height)
    pl_widths = [_ink_box(pl_gray, box)[0] for box in _cells_px(pl_fig)]

    assert len(pl_widths) == len(mpl_widths) == n
    for pl_w, mpl_w in zip(pl_widths, mpl_widths):
        assert abs(pl_w - mpl_w) / mpl_w <= 0.06, (pl_widths, mpl_widths)
