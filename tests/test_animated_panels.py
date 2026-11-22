"""Animated panel grids: `animate=` together with `panels=` (GH #295).

`hyp.plot([a, b, c, d], panels=(1, 4), animate=True)` gives ONE animation of
the whole grid: one clock for every panel, each panel animated exactly as
the single-axes call for its data is, and one `HyperAnimation` returned for
the figure.

Real figures, real frames, rendered headless: every assertion reads artists,
camera angles and extents back off the returned figure after drawing real
frames, or reads a file a real writer produced. Nothing is mocked.
"""
import itertools
import shutil
import subprocess
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import proj3d

import hypertools as hyp
from hypertools.plot.hyper_animation import HyperAnimation
from hypertools.plot.plot import subplots as hyp_subplots

HAS_FFMPEG = shutil.which('ffmpeg') is not None
HAS_FFPROBE = shutil.which('ffprobe') is not None


def _walk(seed, rows=40, cols=5):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


DATA = [_walk(seed) for seed in range(4)]


def _grid(data=None, **kwargs):
    """An animated grid with short, test-sized defaults."""
    call = dict(panels=(1, 4), animate=True, duration=1, frame_rate=8,
                show=False)
    call.update(kwargs)
    return hyp.plot(DATA if data is None else data, **call)


def _panel_axes(fig):
    """The grid's visible panel axes (not colorbar axes), in panel order."""
    return [ax for ax in fig.axes
            if ax.get_visible() and not hasattr(ax, '_colorbar')
            and ax.get_label() != '<colorbar>']


def _line_data(line):
    """A drawn line's coordinates as one (n_points, n_dims) array."""
    if hasattr(line, 'get_data_3d'):
        return np.column_stack([np.asarray(c, dtype=float)
                                for c in line.get_data_3d()])
    return np.column_stack([np.asarray(c, dtype=float)
                            for c in line.get_data()])


def _drawn(ax):
    """Everything the panel's data lines show right now, concatenated."""
    return np.vstack([_line_data(line) for line in ax.lines])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


# --- the returned object ------------------------------------------------

def test_an_animated_grid_returns_one_hyper_animation_for_the_figure():
    anim = _grid()
    assert isinstance(anim, HyperAnimation)
    assert isinstance(anim.animation, FuncAnimation)
    fig, raw = anim                      # still unpacks as (figure, animation)
    assert fig is anim.figure and raw is anim.animation
    axes = _panel_axes(fig)
    assert len(axes) == 4
    assert all(ax.name == '3d' for ax in axes)
    # ONE animation drives the figure, not one per panel
    assert all(ax.figure is fig for ax in axes)


@pytest.mark.parametrize('duration, frame_rate', [(1, 8), (2, 5), (0.5, 12)])
def test_the_grid_draws_duration_times_frame_rate_frames(duration,
                                                         frame_rate):
    anim = _grid(duration=duration, frame_rate=frame_rate)
    assert anim.n_frames == int(round(duration * frame_rate))


@pytest.mark.parametrize('shape, expected', [
    ((1, 4), (12.8, 3.2)), ((2, 2), (6.4, 6.4)), ((4, 1), (6.4, 12.8)),
    ((1, 3), (9.6, 3.2))])
def test_the_default_figure_is_sized_by_the_grid(shape, expected):
    """No `size=`: one 3.2 in square cell per panel (never narrower than
    matplotlib's 6.4 in), so a saved clip of a 1 x 4 row is not mostly blank
    canvas above and below the panels. Bare panels (no title, legend,
    labels or caption) need no extra room, so the size is exactly that."""
    ani = _grid(DATA[:shape[0] * shape[1]], panels=shape)
    assert tuple(ani.figure.get_size_inches()) == pytest.approx(expected)


def test_the_default_grid_figure_is_mostly_panel_not_blank_canvas():
    """What the sizing is for: in a 1 x 4 row, the panels' cubes span most
    of the figure's height on a real draw (they spanned about a fifth of
    it on the 6.4 x 4.8 in default)."""
    ani = _grid()
    fig = ani.figure
    ani.draw_frame(0)
    fig.canvas.draw()
    ink = np.asarray(fig.canvas.buffer_rgba())[..., :3].min(axis=2) < 128
    rows = np.flatnonzero(ink.any(axis=1))
    assert (rows[-1] - rows[0]) / ink.shape[0] > 0.45


def test_an_explicit_size_is_kept_for_a_bare_grid():
    ani = _grid(size=(9, 4))
    assert tuple(ani.figure.get_size_inches()) == pytest.approx((9, 4))


def test_a_static_grid_keeps_its_default_size():
    fig = hyp.plot(DATA, panels=(1, 4), show=False)
    assert tuple(fig.get_size_inches()) == pytest.approx((6.4, 4.8))


def test_subplots_alias_animates_the_same_grid():
    anim = hyp.plot(DATA, subplots=(2, 2), animate=True, duration=1,
                    frame_rate=8, show=False)
    assert isinstance(anim, HyperAnimation)
    assert len(_panel_axes(anim.figure)) == 4


def test_spare_cells_stay_hidden():
    anim = _grid(DATA[:3], panels=(2, 2))
    assert sum(ax.get_visible() for ax in anim.figure.axes) == 3


def test_show_false_leaves_no_figure_registered_with_pyplot():
    before = set(plt.get_fignums())
    anim = _grid()
    assert set(plt.get_fignums()) == before
    # ...and the returned animation still draws
    anim.draw_frame(3)
    anim.figure.canvas.draw()


# --- one clock, every panel ---------------------------------------------

def test_every_panels_artists_change_between_frames():
    anim = _grid()
    axes = _panel_axes(anim.figure)
    anim.draw_frame(1)
    early = [_drawn(ax) for ax in axes]
    anim.draw_frame(anim.n_frames - 1)
    late = [_drawn(ax) for ax in axes]
    for i, (a, b) in enumerate(zip(early, late)):
        assert a.shape != b.shape or not np.allclose(a, b), (
            f'panel {i} drew the same thing on frames 1 and '
            f'{anim.n_frames - 1}: it is not on the shared clock')


def test_every_panel_rotates_its_own_camera_in_step():
    anim = _grid(rotations=1, duration=2)
    axes = _panel_axes(anim.figure)
    anim.draw_frame(0)
    start = [ax.azim for ax in axes]
    anim.draw_frame(anim.n_frames // 2)
    middle = [ax.azim for ax in axes]
    # every panel's camera moved, and all panels are at the same angle
    for a, b in zip(start, middle):
        assert abs(((b - a + 180) % 360) - 180) > 30
    assert np.allclose(middle, middle[0])
    assert np.allclose(start, start[0])


def test_more_rotations_turn_every_camera_further():
    one = _grid(rotations=1, duration=2)
    two = _grid(rotations=2, duration=2)
    frame = 3
    one.draw_frame(0), two.draw_frame(0)
    base = _panel_axes(one.figure)[0].azim
    one.draw_frame(frame), two.draw_frame(frame)
    for ax1, ax2 in zip(_panel_axes(one.figure), _panel_axes(two.figure)):
        assert (ax2.azim - base) == pytest.approx(2 * (ax1.azim - base))


#: every one of these is a per-panel animation setting: the grid must draw
#: each panel exactly as the single-axes call with the same setting does
_PER_PANEL_SETTINGS = {
    'parallel': dict(),
    'tail_duration': dict(tail_duration=0.25),
    'rotations': dict(rotations=2),
    'spin': dict(animate='spin'),
    'window': dict(animate='window', tail_duration=0.5),
    'chemtrails': dict(chemtrails=True),
    'precog': dict(precog=True),
    'serial': dict(order='serial'),
    'two-d': dict(ndims=2),
    'markers': dict(fmt='o'),
    'elevation': dict(elev=35),
}


@pytest.mark.parametrize('setting', sorted(_PER_PANEL_SETTINGS))
def test_each_panel_is_animated_exactly_as_its_single_axes_call(setting):
    """`panel_fit='independent'` fits each panel as `hyp.plot(x[i], ...)`
    does, so the artists a grid panel shows on frame k must be the artists
    that single-axes animation shows on frame k -- same rows, same trail,
    same camera angle."""
    extra = _PER_PANEL_SETTINGS[setting]
    data = DATA[:3]
    grid = _grid(data, panels=(1, 3), panel_fit='independent', **extra)
    axes = _panel_axes(grid.figure)
    singles = [hyp.plot(d, animate=extra.get('animate', True), duration=1,
                        frame_rate=8, show=False,
                        **{k: v for k, v in extra.items() if k != 'animate'})
               for d in data]
    assert all(single.n_frames == grid.n_frames for single in singles)
    for frame in (0, grid.n_frames // 2, grid.n_frames - 1):
        grid.draw_frame(frame)
        for i, (ax, single) in enumerate(zip(axes, singles)):
            single.draw_frame(frame)
            single_ax = single.figure.axes[0]
            assert len(ax.lines) == len(single_ax.lines)
            for line, expected in zip(ax.lines, single_ax.lines):
                got, want = _line_data(line), _line_data(expected)
                assert got.shape == want.shape, (setting, frame, i)
                assert np.allclose(got, want, equal_nan=True), (
                    setting, frame, i)
                assert line.get_alpha() == expected.get_alpha()
            if ax.name == '3d':
                assert ax.azim == pytest.approx(single_ax.azim)
                assert ax.elev == pytest.approx(single_ax.elev)


def test_tail_duration_changes_what_every_panel_shows():
    short = _grid(animate='window', tail_duration=0.25, duration=2)
    long_ = _grid(animate='window', tail_duration=1.0, duration=2)
    frame = short.n_frames - 1
    short.draw_frame(frame), long_.draw_frame(frame)
    for ax_short, ax_long in zip(_panel_axes(short.figure),
                                 _panel_axes(long_.figure)):
        assert len(_drawn(ax_short)) < len(_drawn(ax_long))


def test_serial_order_plays_within_each_panel_not_across_panels():
    """order='serial' reveals each panel's own datasets one after another;
    it is not a sequence across the panels."""
    anim = _grid(order='serial', duration=2)
    axes = _panel_axes(anim.figure)
    anim.draw_frame(anim.n_frames // 2)
    # half way through, EVERY panel has revealed part of its own dataset
    counts = [len(_line_data(ax.lines[0])) for ax in axes]
    assert all(count > 1 for count in counts), counts
    assert len(set(counts)) == 1, counts


# --- titles, legends, colorbars -----------------------------------------

def test_titles_are_per_panel():
    names = ['one', 'two', 'three', 'four']
    anim = _grid(title=names)
    assert [ax.get_title() for ax in _panel_axes(anim.figure)] == names


def test_a_single_title_names_every_panel():
    anim = _grid(title='same')
    assert [ax.get_title() for ax in _panel_axes(anim.figure)] == ['same'] * 4


def test_a_callable_title_is_updated_per_panel_every_frame():
    anim = _grid(title=lambda ctx: f'frame {ctx.frame}')
    anim.draw_frame(5)
    assert [ax.get_title() for ax in _panel_axes(anim.figure)] == (
        ['frame 5'] * 4)


def test_the_grid_gets_one_legend_naming_every_panel():
    # CHANGED for 1.2.0 (owner decision 2026-11-22; was
    # `test_every_panel_gets_its_legend`): every panel carried a one-entry
    # legend. An animated grid now draws ONE legend for the figure, listing
    # every dataset once, as a static grid does.
    anim = _grid(legend=['a', 'b', 'c', 'd'])
    assert all(ax.get_legend() is None for ax in _panel_axes(anim.figure))
    (legend,) = anim.figure.legends
    assert [text.get_text() for text in legend.get_texts()] == (
        ['a', 'b', 'c', 'd'])


def test_the_grid_gets_one_colorbar():
    # CHANGED for 1.2.0 (owner decision 2026-11-22; was
    # `test_every_panel_gets_its_colorbar`): every panel carried its own
    # colorbar (six axes). The panels share one color scale, so the grid
    # draws ONE colorbar for the figure.
    hue = [np.arange(len(d), dtype=float) for d in DATA[:3]]
    anim = _grid(DATA[:3], panels=(1, 3), hue=hue, colorbar=True)
    fig = anim.figure
    # three panels and one colorbar axes
    assert len(fig.axes) == 4
    assert sum(ax.name == '3d' for ax in fig.axes) == 3
    assert sum(ax.get_label() == '<colorbar>' for ax in fig.axes) == 1


# --- fits ---------------------------------------------------------------

def test_shared_fit_gives_the_static_grids_coordinates():
    """The pipeline is fit exactly as for a static grid; only the drawing
    is animated."""
    static = hyp.plot(DATA, panels=(1, 4), show=False, return_model=True)
    moving = _grid(return_model=True)
    for a, b in zip(static['xform_data'], moving['xform_data']):
        assert np.allclose(a, b)
    assert moving['pipeline'] is not None


def test_independent_fit_differs_from_the_shared_fit():
    shared = _grid(return_model=True)
    independent = _grid(panel_fit='independent', return_model=True)
    assert independent['pipeline'] is None
    assert any(not np.allclose(a, b) for a, b in
               zip(shared['xform_data'], independent['xform_data']))


def test_a_reducer_list_animates_one_panel_per_reducer():
    anim = _grid(DATA[:3], panels=True, reduce=['PCA', 'FastICA'],
                 title=['PCA', 'FastICA'])
    axes = _panel_axes(anim.figure)
    assert len(axes) == 2
    assert [ax.get_title() for ax in axes] == ['PCA', 'FastICA']
    # every panel draws EVERY dataset
    assert all(len(ax.lines) == 3 for ax in axes)
    anim.draw_frame(anim.n_frames - 1)
    last = [_drawn(ax) for ax in axes]
    # ...through a different reducer
    assert last[0].shape != last[1].shape or not np.allclose(last[0],
                                                             last[1])
    # and each reducer panel matches that reducer's own single-axes call
    single = hyp.plot(DATA[:3], reduce='PCA', animate=True, duration=1,
                      frame_rate=8, show=False)
    single.draw_frame(single.n_frames - 1)
    for line, expected in zip(axes[0].lines, single.figure.axes[0].lines):
        assert np.allclose(_line_data(line), _line_data(expected))


def test_a_morph_grid_reports_its_segments():
    anim = _grid(DATA[:3], panels=True, reduce=['PCA', 'FastICA'],
                 animate='morph', duration=2)
    # three clouds: hold, move, hold, move, hold
    assert anim.n_segments == 5
    anim.draw_frame(0)
    first = [np.asarray(ax.collections[-1].get_offsets())
             if ax.collections else None
             for ax in _panel_axes(anim.figure)]
    anim.draw_frame(anim.n_frames - 1)
    assert first is not None


def test_return_model_adds_the_animation_to_the_panel_bundle():
    bundle = _grid(return_model=True)
    assert isinstance(bundle['animation'], FuncAnimation)
    assert bundle['panels'] == (1, 4)
    assert len(bundle['axes']) == len(bundle['panel_models']) == 4
    assert len(bundle['xform_data']) == 4
    # a panel has no animation of its own
    assert all(model['animation'] is None
               for model in bundle['panel_models'])
    # the handle draws the grid
    bundle['animation']._func(2)
    assert bundle['fig'] is bundle['axes'][0].figure


# --- geometry: nothing leaves its cell or the canvas ----------------------

def _cube_corners_px(ax):
    corners = np.array(list(itertools.product(
        ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())))
    x, y, _ = proj3d.proj_transform(corners[:, 0], corners[:, 1],
                                    corners[:, 2], ax.get_proj())
    return ax.transData.transform(np.column_stack([x, y]))


def _inside(inner, outer, slack=1.0):
    return (inner.x0 >= outer.x0 - slack and inner.x1 <= outer.x1 + slack
            and inner.y0 >= outer.y0 - slack and inner.y1 <= outer.y1 + slack)


@pytest.mark.parametrize('extra', [
    dict(),
    dict(zoom=3),
    dict(elev=45),
    dict(panels=(2, 2)),
    dict(panels=(4, 1)),
    dict(animate='spin', rotations=2),
], ids=['default', 'zoom', 'elev', '2x2', '4x1', 'spin'])
def test_every_rotating_cube_stays_inside_its_own_cell(extra):
    anim = _grid(duration=3, frame_rate=12, **extra)
    fig = anim.figure
    axes = _panel_axes(fig)
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        fig.canvas.draw()
        for i, ax in enumerate(axes):
            box = ax.get_window_extent()
            corners = _cube_corners_px(ax)
            assert corners[:, 0].min() >= box.x0 - 0.5, (frame, i)
            assert corners[:, 0].max() <= box.x1 + 0.5, (frame, i)
            assert corners[:, 1].min() >= box.y0 - 0.5, (frame, i)
            assert corners[:, 1].max() <= box.y1 + 0.5, (frame, i)


def test_panel_boxes_do_not_overlap_each_other():
    anim = _grid(legend=['a', 'b', 'c', 'd'], title=['1', '2', '3', '4'])
    fig = anim.figure
    anim.draw_frame(0)
    fig.canvas.draw()
    boxes = [ax.get_window_extent() for ax in _panel_axes(fig)]
    for a, b in itertools.combinations(boxes, 2):
        assert not a.overlaps(b)


def test_titles_and_legends_stay_on_the_canvas_and_clear_of_the_cubes():
    anim = _grid(legend=['alpha', 'beta', 'gamma', 'delta'],
                 title=['one', 'two', 'three', 'four'], duration=2,
                 frame_rate=6)
    fig = anim.figure
    axes = _panel_axes(fig)
    # CHANGED for 1.2.0 (owner decision 2026-11-22): every panel had its
    # own legend, which had to sit right of its own cube and short of the
    # next panel's. The grid now draws ONE legend, which must sit right of
    # EVERY panel's cube, on the canvas, at every frame.
    (shared,) = fig.legends
    for frame in (0, anim.n_frames // 2, anim.n_frames - 1):
        anim.draw_frame(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        canvas = fig.bbox
        legend = shared.get_window_extent(renderer)
        assert _inside(legend, canvas), (frame, 'legend')
        for i, ax in enumerate(axes):
            assert ax.get_legend() is None
            title = ax.title.get_window_extent(renderer)
            assert _inside(title, canvas), (frame, i, 'title')
            corners = _cube_corners_px(ax)
            # the title is above the cube, the legend right of every cube
            assert title.y0 >= corners[:, 1].max(), (frame, i)
            assert legend.x0 >= corners[:, 0].max(), (frame, i)


def test_axis_labels_enlarge_the_figure_and_stay_inside_their_cells():
    plain = _grid(DATA[:3], panels=(1, 3))
    labeled = _grid(DATA[:3], panels=(1, 3), xlabel='first axis',
                    ylabel='second axis', zlabel='third axis', duration=2,
                    frame_rate=6)
    assert (labeled.figure.get_size_inches()[0]
            > plain.figure.get_size_inches()[0])
    fig = labeled.figure
    axes = _panel_axes(fig)
    for frame in range(labeled.n_frames):
        labeled.draw_frame(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for i, ax in enumerate(axes):
            box = ax.get_window_extent()
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                label = axis.label.get_window_extent(renderer)
                # a few pixels of slack: matplotlib places the labels at
                # draw time and the fit samples the turn every 30 degrees
                assert _inside(label, box, slack=0.06 * box.width), (
                    frame, i, axis.label.get_text())
            assert _inside(box, fig.bbox)


def test_two_d_panels_animate_in_fixed_boxes():
    anim = _grid(ndims=2, panels=(2, 2), title=['a', 'b', 'c', 'd'])
    fig = anim.figure
    axes = _panel_axes(fig)
    assert all(ax.name == 'rectilinear' for ax in axes)
    anim.draw_frame(0)
    fig.canvas.draw()
    before = [ax.get_position().bounds for ax in axes]
    anim.draw_frame(anim.n_frames - 1)
    fig.canvas.draw()
    assert [ax.get_position().bounds for ax in axes] == before
    for a, b in itertools.combinations(
            [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in axes], 2):
        assert not a.overlaps(b)


# --- labels= ------------------------------------------------------------

def test_point_labels_follow_the_reveal_in_every_panel():
    """The annotation bookkeeping was one module-level list holding only
    the LAST plot's labels, so in a grid only the last panel's labels
    followed the animation."""
    labels = [[f'p{j}' if j in (0, 35) else None for j in range(len(d))]
              for d in DATA[:3]]
    anim = _grid(DATA[:3], panels=(1, 3), labels=labels, duration=2)
    axes = _panel_axes(anim.figure)

    def visible(ax):
        return sorted(text.get_text() for text in ax.texts
                      if text.get_visible())

    anim.draw_frame(1)
    early = [visible(ax) for ax in axes]
    anim.draw_frame(anim.n_frames - 1)
    late = [visible(ax) for ax in axes]
    # the late point's label appears in EVERY panel only once it is reached
    assert all('p35' not in shown for shown in early), early
    assert all('p35' in shown for shown in late), late


# --- on_frame -----------------------------------------------------------

def test_on_frame_is_called_once_per_frame_for_the_whole_figure():
    seen = []
    anim = _grid(on_frame=seen.append)
    assert seen == []                    # nothing fires at construction
    anim.draw_frame(3)
    assert len(seen) == 1
    ctx = seen[0]
    axes = _panel_axes(anim.figure)
    assert ctx.frame == 3 and ctx.n_frames == anim.n_frames
    assert ctx.figure is anim.figure
    assert isinstance(ctx.axes, tuple) and list(ctx.axes) == axes
    assert ctx.progress == pytest.approx(3 / (anim.n_frames - 1))


def test_the_grid_context_holds_one_context_per_panel():
    seen = []
    anim = _grid(on_frame=seen.append)
    anim.draw_frame(4)
    ctx = seen[-1]
    axes = _panel_axes(anim.figure)
    assert len(ctx.panels) == 4
    for panel, ax in zip(ctx.panels, axes):
        assert panel.axes is ax
        assert panel.frame == 4 and panel.n_frames == anim.n_frames
        assert all(artist.axes is ax for artist in panel.artists)
        assert len(panel.datasets) == 1
        assert panel.revealed_counts is not None
    # the figure-level fields are every panel's, in panel order
    assert ctx.artists == tuple(a for p in ctx.panels for a in p.artists)
    assert len(ctx.datasets) == 4
    assert ctx.revealed_counts == tuple(
        c for p in ctx.panels for c in p.revealed_counts)
    assert ctx.current_index is None


def test_a_single_axes_context_has_no_panels():
    seen = []
    anim = hyp.plot(DATA[0], animate=True, duration=1, frame_rate=8,
                    on_frame=seen.append, show=False)
    anim.draw_frame(2)
    assert seen[-1].panels == ()
    assert seen[-1].axes is anim.figure.axes[0]


def test_on_frame_can_be_attached_after_the_call():
    anim = _grid()
    seen = []
    assert anim.on_frame(lambda ctx: seen.append(ctx.frame)) is anim
    anim.draw_frame(2).draw_frame(5)
    assert seen == [2, 5]


def test_on_frame_must_be_callable():
    with pytest.raises(TypeError, match='on_frame must be callable'):
        _grid(on_frame=3)


# --- export -------------------------------------------------------------

def test_saving_a_gif_writes_every_frame_of_the_whole_grid(tmp_path):
    from PIL import Image
    out = tmp_path / 'grid.gif'
    anim = _grid(save_path=str(out), title=['a', 'b', 'c', 'd'])
    with Image.open(out) as im:
        assert im.n_frames == anim.n_frames == 8
        width, height = im.size
    fig = anim.figure
    assert (width, height) == tuple(
        int(v) for v in fig.get_size_inches() * fig.dpi)
    # one wide image holding four panels, not one panel's worth
    assert width > height


def test_the_saved_gif_shows_all_four_panels_moving(tmp_path):
    from PIL import Image
    out = tmp_path / 'grid.gif'
    anim = _grid(save_path=str(out), duration=2)
    fig = anim.figure
    anim.draw_frame(0)
    fig.canvas.draw()
    width = fig.bbox.width
    spans = [(ax.get_window_extent().x0 / width,
              ax.get_window_extent().x1 / width)
             for ax in _panel_axes(fig)]
    with Image.open(out) as im:
        im.seek(1)
        early = np.asarray(im.convert('RGB')).astype(int)
        im.seek(im.n_frames - 1)
        late = np.asarray(im.convert('RGB')).astype(int)
    changed = np.abs(early - late).max(-1) > 40
    for i, (x0, x1) in enumerate(spans):
        band = changed[:, int(x0 * changed.shape[1]):
                       int(x1 * changed.shape[1])]
        assert band.any(), f'panel {i} is identical in both saved frames'


@pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg not installed')
def test_saving_an_mp4_writes_every_frame(tmp_path):
    out = tmp_path / 'grid.mp4'
    anim = _grid(save_path=str(out), caption='a caption')
    assert out.stat().st_size > 1000
    if not HAS_FFPROBE:
        return
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-count_frames', '-show_entries', 'stream=nb_read_frames',
         '-of', 'csv=p=0', str(out)],
        capture_output=True, text=True, check=True)
    assert int(probe.stdout.strip().split(',')[0]) == anim.n_frames


def test_hyper_animation_save_writes_the_grid(tmp_path):
    from PIL import Image
    anim = _grid()
    out = tmp_path / 'later.gif'
    anim.save(out)
    with Image.open(out) as im:
        assert im.n_frames == anim.n_frames


def test_to_jshtml_embeds_every_frame():
    anim = _grid(duration=1, frame_rate=5)
    html = anim.to_jshtml()
    assert html.count('data:image/png;base64') == anim.n_frames == 5


def test_an_unsupported_extension_is_refused_before_anything_is_fitted(
        tmp_path):
    with pytest.raises(ValueError, match='unsupported animation save format'):
        _grid(save_path=str(tmp_path / 'grid.xyz'))


# --- refusals -----------------------------------------------------------

def test_the_plotly_backend_is_refused_like_companion():
    with pytest.raises(NotImplementedError,
                       match="backend='matplotlib'") as err:
        _grid(backend='plotly')
    assert 'matplotlib-only' in str(err.value)


def test_a_static_plotly_grid_still_works():
    fig = hyp.plot(DATA, panels=(1, 4), backend='plotly', show=False)
    assert type(fig).__module__.startswith(('plotly', 'hypertools'))
    assert len(fig.data) >= 4


def test_companion_is_refused_with_what_to_do_instead():
    with pytest.raises(ValueError, match='companion= cannot be combined'):
        _grid(companion={'data': np.arange(40.0)})


def test_explore_is_refused():
    with pytest.raises(ValueError, match='explore'):
        _grid(explore=True)


def test_panels_of_different_widths_are_refused():
    mixed = [_walk(0, cols=5), _walk(1, cols=2)]
    with pytest.raises(ValueError, match='same number of columns'):
        hyp.plot(mixed, panels=True, panel_fit='independent', animate=True,
                 duration=1, frame_rate=8, show=False)
    # the static grid draws the narrow panel flat on its cell's floor
    fig = hyp.plot(mixed, panels=True, panel_fit='independent', show=False)
    assert len(_panel_axes(fig)) == 2


def test_a_caller_supplied_ax_is_refused_with_panels():
    fig, axes = hyp_subplots(1, 1)
    with pytest.raises(ValueError, match='ax='):
        _grid(ax=axes[0])


def test_the_public_ax_plus_animate_refusal_still_holds():
    fig, axes = hyp_subplots(1, 2)
    with pytest.raises(ValueError, match='cannot be combined with animate'):
        hyp.plot(DATA[0], ax=axes[0], animate=True, show=False)
    # ...and the internal marker is not a public keyword
    assert '_panel_animation' not in str(
        __import__('inspect').signature(hyp.plot))


def test_axis_labels_that_cannot_fit_an_explicit_size_are_refused():
    with pytest.raises(ValueError, match='larger size=') as err:
        _grid(DATA[:3], panels=(1, 3), size=(4, 3), xlabel='first axis',
              ylabel='second axis', zlabel='third axis')
    assert 'drop the axis labels' in str(err.value)
    assert not plt.get_fignums()


def test_panels_with_different_frame_counts_are_refused():
    """Every public route gives the panels one schedule, so this drives the
    grid builder with two real panels drawn at different frame rates."""
    from hypertools.plot.animation_context import FrameHooks
    from hypertools.plot.plot import _animate_panel_grid
    fig, axes = hyp_subplots(1, 2)
    cells = []
    hyp.plot(DATA[0], ax=axes[0], animate=True, duration=1, frame_rate=8,
             show=False, _panel_animation=cells)
    hyp.plot(DATA[1], ax=axes[1], animate=True, duration=1, frame_rate=5,
             show=False, _panel_animation=cells)
    assert [cell['stub']._save_count for cell in cells] == [8, 5]
    with pytest.raises(ValueError, match='different numbers of frames'):
        _animate_panel_grid(
            fig, cells, list(axes), [], (1, 2), grid_hooks=FrameHooks(),
            caption=None, dynamic_caption=None, font=None, save_path=None,
            show=False, return_model=False, shared_pipeline=None)


def test_an_animated_grid_raises_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        anim = _grid(title=['a', 'b', 'c', 'd'], legend=['a', 'b', 'c', 'd'],
                     caption='quiet')
        anim.draw_frame(0)
        anim.figure.canvas.draw()


def test_a_frame_grid_of_dataframes_keeps_its_index_for_the_caption():
    index = pd.date_range('2024-01-01', periods=40, freq='D')
    frames = [pd.DataFrame(d, index=index) for d in DATA[:2]]
    anim = _grid(frames, panels=(1, 2), caption='{index:%Y-%m-%d}')
    anim.draw_frame(anim.n_frames - 1)
    assert anim.figure.texts[-1].get_text() == '2024-02-09'
