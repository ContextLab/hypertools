"""`hyp.plot` stays quiet and keeps its hands off a figure the caller built.

Three reports from executing the timecorr tutorial notebooks against
hypertools 1.1.0 (2026-11-21):

- GH #290: the import-time backend warning was repeated on every
  `hyp.plot` call, including calls that display nothing.
- GH #291: `hyp.plot(..., ax=ax)` called `plt.tight_layout()`, which moved
  every axes of the caller's figure (or of whichever figure was current).
- GH #292: matplotlib >= 3.11 logged ``findfont: Failed to find font weight
  normal for Hiragino Kaku Gothic ProN`` several times per figure on macOS.
"""

import logging
import warnings

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot import backend as B
from hypertools.plot import fonts

matplotlib.use('Agg')


def _walk():
    return np.random.default_rng(0).normal(size=(50, 3)).cumsum(0)


def _positions(fig):
    return np.round([a.get_position().bounds for a in fig.axes], 6).tolist()


# ---------------------------------------------------------------------------
# GH #290: the backend warning
# ---------------------------------------------------------------------------

MESSAGE = 'example backend warning (GH #290 test)'


@pytest.fixture
def backend_warning():
    """Set `BACKEND_WARNING` as a failed import-time backend switch would,
    with a clean record of what this session has already been told."""
    saved_warning = B.BACKEND_WARNING
    saved_issued = set(B._BACKEND_WARNINGS_ISSUED)
    B.BACKEND_WARNING = MESSAGE
    B._BACKEND_WARNINGS_ISSUED.clear()
    yield MESSAGE
    B.BACKEND_WARNING = saved_warning
    B._BACKEND_WARNINGS_ISSUED.clear()
    B._BACKEND_WARNINGS_ISSUED.update(saved_issued)
    plt.close('all')


def _backend_warnings(caught, message=MESSAGE):
    return [str(w.message) for w in caught if str(w.message) == message]


def _record(**plot_kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(_walk(), **plot_kwargs)
    return caught


def test_backend_warning_is_silent_for_a_plot_that_displays_nothing(
        backend_warning):
    """The issue's own reproduction: `show=False` into a caller's axes."""
    for _ in range(2):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        assert _backend_warnings(_record(ax=ax, show=False)) == []


def test_backend_warning_is_silent_for_static_plots_even_when_shown(
        backend_warning):
    """A static figure needs no interactive backend: any backend shows it."""
    assert _backend_warnings(_record(show=True)) == []
    assert _backend_warnings(_record(show=False)) == []


def test_backend_warning_is_silent_for_an_animation_that_is_not_displayed(
        backend_warning, tmp_path):
    assert _backend_warnings(_record(animate=True, show=False)) == []
    out = tmp_path / 'walk.gif'
    assert _backend_warnings(
        _record(animate=True, save_path=str(out), duration=1)) == []
    assert out.stat().st_size > 0


def test_backend_warning_is_issued_once_for_displayed_animations(
        backend_warning):
    """The one kind of call the warning is about hears it -- once per
    session, not once per call. `mpl_backend='disable'` keeps this test
    from switching the suite's backend; the warning does not depend on it."""
    first = _record(animate=True, show=True, mpl_backend='disable')
    assert _backend_warnings(first) == [MESSAGE]
    assert all(w.category is UserWarning for w in first
               if str(w.message) == MESSAGE)
    second = _record(animate=True, show=True, mpl_backend='disable')
    assert _backend_warnings(second) == []


def test_a_new_backend_warning_is_issued_in_its_turn(backend_warning):
    assert _backend_warnings(
        _record(animate=True, show=True, mpl_backend='disable')) == [MESSAGE]
    other = 'another backend warning (GH #290 test)'
    B.BACKEND_WARNING = other
    caught = _record(animate=True, show=True, mpl_backend='disable')
    assert _backend_warnings(caught, other) == [other]
    assert _backend_warnings(caught) == []


def test_backend_warning_is_silent_when_plotly_renders_the_animation(
        backend_warning):
    """Plotly embeds the frames in its own figure; matplotlib's backend
    plays no part, so its warning would be wrong as well as noisy."""
    pytest.importorskip('plotly')
    caught = _record(animate=True, show=True, backend='plotly')
    assert _backend_warnings(caught) == []


def test_backend_warning_inside_a_set_interactive_backend_context(
        backend_warning):
    """Inside the context the wrapper skips its own backend switch, but the
    warning follows the same rule: displayed animations only, once."""
    with hyp.set_interactive_backend(B.HYPERTOOLS_BACKEND.normalize()):
        assert B.BACKEND_WARNING == MESSAGE      # same backend: kept
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        assert _backend_warnings(_record(ax=ax, show=False)) == []
        assert _backend_warnings(_record(animate=True, show=True)) == [
            MESSAGE]
        assert _backend_warnings(_record(animate=True, show=True)) == []


def test_backend_warning_is_silent_when_ax_is_a_plotly_figure(
        backend_warning):
    """A plotly Figure passed as `ax=` makes plotly the renderer even with
    the default ``backend='auto'``."""
    go = pytest.importorskip('plotly.graph_objects')
    caught = _record(ax=go.Figure(), interactive=True, show=True,
                     mpl_backend='disable')
    assert _backend_warnings(caught) == []


def test_a_rejected_argument_inside_the_context_restores_rcparams(
        backend_warning):
    """Inspecting the call for the warning happens under the same
    `finally` that puts the caller's rcParams back."""
    before = (matplotlib.rcParams['pdf.fonttype'],
              matplotlib.rcParams['ps.fonttype'])
    assert before != (42, 42), 'this test needs the stock fonttype'
    for in_context in (True, False):
        with pytest.raises(ValueError, match='backend must be one of'):
            if in_context:
                with hyp.set_interactive_backend(
                        B.HYPERTOOLS_BACKEND.normalize()):
                    hyp.plot(_walk(), animate=True, backend='invalid')
            else:
                hyp.plot(_walk(), animate=True, backend='invalid')
        assert (matplotlib.rcParams['pdf.fonttype'],
                matplotlib.rcParams['ps.fonttype']) == before


def test_an_array_valued_animate_gets_plots_own_error(backend_warning):
    """Deciding whether the warning applies must not trip over an argument
    with no single truth value: `plot` rejects it with its own message, in
    and out of a `set_interactive_backend` context."""
    def _error():
        with pytest.raises(Exception) as raised:
            hyp.plot(_walk(), animate=np.array([1, 0]), show=False)
        return str(raised.value)

    outside = _error()
    assert 'truth value' not in outside
    with hyp.set_interactive_backend(B.HYPERTOOLS_BACKEND.normalize()):
        assert _error() == outside


# ---------------------------------------------------------------------------
# GH #291: the caller's layout
# ---------------------------------------------------------------------------

def test_plot_into_caller_axes_leaves_every_axes_where_it_was():
    """The issue's own reproduction."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4),
                            subplot_kw={'projection': '3d'})
    fig.subplots_adjust(left=0.2, right=0.7, wspace=0.5)
    before = _positions(fig)
    assert before == [[0.2, 0.295, 0.2, 0.4], [0.5, 0.295, 0.2, 0.4]]
    hyp.plot(_walk(), ax=axs[0], show=False)
    assert _positions(fig) == before
    assert fig.get_size_inches().tolist() == [8.0, 4.0]
    plt.close(fig)


@pytest.mark.parametrize('ndims', [2, 3])
def test_plot_into_add_axes_axes_does_not_warn_about_tight_layout(ndims):
    fig = plt.figure(figsize=(6, 4))
    kwargs = {'projection': '3d'} if ndims == 3 else {}
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.5], **kwargs)
    inset = fig.add_axes([0.7, 0.7, 0.2, 0.2])
    before = _positions(fig)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(_walk(), ax=ax, ndims=ndims, show=False)
    assert [str(w.message) for w in caught
            if 'tight_layout' in str(w.message)] == []
    assert _positions(fig) == before
    assert inset in fig.axes
    plt.close(fig)


def test_plot_into_caller_axes_leaves_the_current_figure_alone():
    """`plt.tight_layout()` acts on pyplot's CURRENT figure, which need not
    be the one that owns `ax`."""
    target, ax = plt.subplots(subplot_kw={'projection': '3d'})
    bystander, others = plt.subplots(1, 2, figsize=(8, 4))
    bystander.subplots_adjust(left=0.3, right=0.6, wspace=0.9)
    assert plt.gcf() is bystander
    before_target, before_bystander = _positions(target), _positions(bystander)
    hyp.plot(_walk(), ax=ax, show=False)
    assert _positions(bystander) == before_bystander
    assert _positions(target) == before_target
    plt.close(target)
    plt.close(bystander)


@pytest.mark.parametrize('plot_kwargs', [
    {},
    {'legend': ['a long legend label for the walk']},
    {'title': 'a title'},
], ids=['bare', 'legend', 'title'])
def test_plot_into_caller_axes_keeps_the_callers_layout_engine(plot_kwargs):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout='constrained',
                            subplot_kw={'projection': '3d'})
    engine = fig.get_layout_engine()
    hyp.plot(_walk(), ax=axs[0], show=False, **plot_kwargs)
    assert fig.get_layout_engine() is engine
    assert fig.get_size_inches().tolist() == [8.0, 4.0]
    fig.canvas.draw()                     # the caller's engine still runs
    assert fig.get_layout_engine() is engine
    plt.close(fig)


def _legend_report(fig, axs):
    """Per axes: (legend runs off the canvas, legend covers another axes)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    report = []
    for ax in axs:
        box = ax.get_legend().get_window_extent(renderer)
        covers = any(box.overlaps(other.get_window_extent(renderer))
                     for other in axs if other is not ax)
        report.append((bool(box.x1 > fig.bbox.x1 + 1), covers))
    return report


def _two_walks():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(50, 3)).cumsum(0) for _ in range(2)]


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('n_panels', [1, 2])
def test_legends_on_caller_axes_leave_the_figure_alone_and_lay_out_cleanly(
        ndims, n_panels):
    """A legend beside a caller's axes changes nothing about the caller's
    figure (1.1.0 widened it, stretching its other axes), and the caller's
    own `fig.tight_layout()` then places every legend on the canvas, clear
    of the other panels. (On a figure hypertools had widened, that same
    call left the last legend clipped.)"""
    fig, axs = plt.subplots(
        1, n_panels, figsize=(4 * n_panels, 4), squeeze=False,
        subplot_kw={'projection': '3d'} if ndims == 3 else {})
    axs = list(axs.ravel())
    before = _positions(fig)
    for ax in axs:
        hyp.plot(_two_walks(), ax=ax, ndims=ndims, show=False,
                 legend=['walk A', 'walk B'])
    assert _positions(fig) == before
    assert fig.get_size_inches().tolist() == [4.0 * n_panels, 4.0]
    assert fig.get_layout_engine() is None
    fig.tight_layout()
    assert _legend_report(fig, axs) == [(False, False)] * n_panels
    plt.close(fig)


@pytest.mark.parametrize('ndims', [2, 3])
def test_legends_on_caller_axes_fit_under_the_callers_constrained_layout(
        ndims):
    fig, axs = plt.subplots(
        1, 2, figsize=(8, 4), layout='constrained',
        subplot_kw={'projection': '3d'} if ndims == 3 else {})
    engine = fig.get_layout_engine()
    for ax in axs:
        hyp.plot(_two_walks(), ax=ax, ndims=ndims, show=False,
                 legend=['walk A', 'walk B'])
    assert _legend_report(fig, list(axs)) == [(False, False)] * 2
    assert fig.get_layout_engine() is engine
    assert fig.get_size_inches().tolist() == [8.0, 4.0]
    plt.close(fig)


def test_plot_with_a_legend_into_subfigure_axes():
    """An axes that lives in a SubFigure: the legend fit crashed on it
    (`'SubFigure' object has no attribute 'get_size_inches'`), in 1.1.0
    too."""
    fig = plt.figure(figsize=(8, 4))
    left, right = fig.subfigures(1, 2)
    ax = left.add_subplot(projection='3d')
    other = right.add_subplot()
    before = _positions(fig)
    hyp.plot(_two_walks(), ax=ax, show=False, legend=['walk A', 'walk B'])
    assert ax.get_legend() is not None
    assert _positions(fig) == before and other in fig.axes
    fig.canvas.draw()
    plt.close(fig)


# -- the legend and title hypertools adds are fitted without being asked ----

LONG_LABELS = ['first group', 'second group', 'third long group name']


def _three_clouds(ndims):
    rng = np.random.default_rng(0)
    return [rng.normal(size=(40, ndims)) + i for i in range(3)]


def _decoration_report(fig, axs=None):
    """Sorted problems with what hypertools added, from a real draw: a
    legend past the canvas or over another axes, a title past the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    for i, ax in enumerate(fig.axes if axs is None else axs):
        legend = ax.get_legend()
        if legend is not None:
            box = legend.get_window_extent(renderer)
            if box.x1 > fig.bbox.x1 + 0.5 or box.x0 < fig.bbox.x0 - 0.5:
                problems.append(f'legend {i} off canvas')
            for j, other in enumerate(fig.axes):
                if other is not ax and box.overlaps(
                        other.get_window_extent(renderer)):
                    problems.append(f'legend {i} over axes {j}')
        if ax.get_title():
            if ax.title.get_window_extent(renderer).y1 > fig.bbox.y1 + 0.5:
                problems.append(f'title {i} off canvas')
    return problems


def _caller_figure(shape, ndims, figsize, **fig_kwargs):
    fig, axs = plt.subplots(
        *shape, figsize=figsize, squeeze=False,
        subplot_kw={'projection': '3d'} if ndims == 3 else {}, **fig_kwargs)
    return fig, list(axs.ravel())


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('shape, figsize', [
    ((1, 1), (5, 4)), ((1, 2), (9, 4)), ((2, 2), (9, 7)), ((1, 3), (15, 4))])
def test_a_legend_on_caller_axes_fits_without_any_call_from_the_caller(
        ndims, shape, figsize):
    """`plt.subplots()` then `hyp.plot(..., ax=ax, legend=...)` and nothing
    else: every legend is on the canvas and clear of the other panels (it
    ran off the canvas, or over the next panel, until the caller laid the
    figure out), with the figure's size, layout engine and every OTHER
    axes exactly as the caller made them."""
    fig, axs = _caller_figure(shape, ndims, figsize)
    slots = [ax.get_position(original=True).bounds for ax in axs]
    for ax in axs:
        hyp.plot(_three_clouds(ndims), 'o', ax=ax, legend=LONG_LABELS,
                 title='A title', show=False)
    assert _decoration_report(fig) == []
    assert fig.get_size_inches().tolist() == list(figsize)
    assert fig.get_layout_engine() is None
    assert len(fig.axes) == len(axs)
    for ax, slot in zip(axs, slots):
        # the room the caller's layout gave the axes is not rewritten...
        assert ax.get_position(original=True).bounds == slot
        # ...the drawn axes only ever sits inside it, anchored bottom-left
        drawn = ax.get_position()
        assert drawn.x0 >= slot[0] - 1e-9 and drawn.y0 >= slot[1] - 1e-9
        assert drawn.x1 <= slot[0] + slot[2] + 1e-9
        assert drawn.y1 <= slot[1] + slot[3] + 1e-9
        assert ax.get_legend().get_bbox_to_anchor() is not None
    plt.close(fig)


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('later', ['tight_layout', 'subplots_adjust',
                                   'resize', 'none'])
def test_the_fit_holds_whatever_the_caller_does_to_the_figure_afterwards(
        ndims, later):
    """The fit is worked out when the figure is drawn, so it follows the
    caller's own later layout calls (a 3-D legend was still clipped after
    `fig.tight_layout()`, which reserves no room for one)."""
    fig, axs = _caller_figure((2, 2), ndims, (9, 7))
    for ax in axs:
        hyp.plot(_three_clouds(ndims), 'o', ax=ax, legend=LONG_LABELS,
                 show=False)
    fig.canvas.draw()
    with warnings.catch_warnings():
        warnings.simplefilter('error')    # e.g. "not compatible with tight_layout"
        if later == 'tight_layout':
            fig.tight_layout()
        elif later == 'subplots_adjust':
            fig.subplots_adjust(left=0.05, right=0.97, wspace=0.6)
        elif later == 'resize':
            fig.set_size_inches(12, 6)
        assert _decoration_report(fig) == []
    plt.close(fig)


@pytest.mark.parametrize('ndims', [2, 3])
def test_the_fit_holds_in_a_saved_file_at_another_dpi(ndims, tmp_path):
    from PIL import Image
    fig, axs = _caller_figure((1, 2), ndims, (9, 4))
    for ax in axs:
        hyp.plot(_three_clouds(ndims), 'o', ax=ax, legend=LONG_LABELS,
                 show=False)
    path = tmp_path / 'saved.png'
    fig.savefig(path, dpi=180, facecolor='white')
    pixels = np.asarray(Image.open(path).convert('L'))
    assert pixels.shape == (4 * 180, 9 * 180)
    # nothing is cut by the right edge: its last columns are background
    assert (pixels[:, -6:] == 255).all()
    assert (pixels < 250).any()
    plt.close(fig)


def test_a_legend_that_already_fits_moves_nothing():
    """Room made by the caller is respected: nothing is narrowed when the
    legend is already inside the canvas and clear of the next panel."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(left=0.05, right=0.8, wspace=0.9)
    slots = [ax.get_position(original=True).bounds for ax in axs]
    for ax in axs:
        hyp.plot(_three_clouds(2), 'o', ax=ax, legend=['a', 'b', 'c'],
                 show=False)
    assert _decoration_report(fig) == []
    np.testing.assert_allclose([ax.get_position().bounds for ax in axs],
                               slots, atol=1e-9)
    plt.close(fig)


def test_a_legend_inside_the_axes_moves_nothing():
    fig, ax = plt.subplots(figsize=(5, 4))
    slot = ax.get_position(original=True).bounds
    hyp.plot(_three_clouds(2), 'o', ax=ax, legend=LONG_LABELS, show=False,
             legend_kwargs={'loc': 'upper left'})
    assert _decoration_report(fig) == []
    np.testing.assert_allclose(ax.get_position().bounds, slot, atol=1e-9)
    plt.close(fig)


def test_a_plot_without_legend_or_title_gets_no_locator():
    """Nothing to fit, nothing installed: `ax.get_position()` keeps
    answering the way it does for any plain axes."""
    fig, ax = plt.subplots()
    hyp.plot(_walk(), ax=ax, ndims=2, show=False)
    assert ax.get_axes_locator() is None
    plt.close(fig)


def test_a_legend_with_no_room_beside_the_axes_is_drawn_inside_it():
    """Three panels across: the first panel's legend cannot sit between it
    and the second panel however far the axes is narrowed (down to its
    floor), but it does fit in the axes. It is drawn inside, framed so it
    reads over the data, instead of covering the next panel or squeezing
    the plot to nothing; the last panel has the margin and keeps its legend
    outside."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    fig.subplots_adjust(wspace=0.05)
    for ax in axs:
        hyp.plot(_three_clouds(2), 'o', ax=ax, show=False, legend=LONG_LABELS)
    assert _decoration_report(fig) == []
    renderer = fig.canvas.get_renderer()
    first, last = axs[0], axs[-1]
    box = first.get_legend().get_window_extent(renderer)
    axes_box = first.get_window_extent(renderer)
    assert first.get_legend().get_frame_on()
    assert axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1
    np.testing.assert_allclose(first.get_position().bounds,
                               first.get_position(original=True).bounds)
    outside = last.get_legend().get_window_extent(renderer)
    assert outside.x0 > last.get_window_extent(renderer).x1
    assert last.get_position().width < last.get_position(original=True).width
    plt.close(fig)


def test_a_legend_larger_than_its_panel_is_reported_once_and_left():
    """Six panels across a narrow figure with long labels: the legend is
    wider than the panel, so there is no placement that is clearly right.
    hypertools says so, once, and changes nothing."""
    fig, axs = plt.subplots(1, 6, figsize=(9, 2.5))
    for ax in axs[:1]:
        hyp.plot(_three_clouds(2), 'o', ax=ax, show=False,
                 legend=['a rather long group label number one',
                         'a rather long group label number two', 'three'])
    with pytest.warns(UserWarning, match='larger than the axes') as caught:
        fig.canvas.draw()
        fig.canvas.draw()
        fig.savefig(__import__('io').BytesIO(), format='png')
    assert len([w for w in caught
                if 'larger than the axes' in str(w.message)]) == 1
    np.testing.assert_allclose(axs[0].get_position().bounds,
                               axs[0].get_position(original=True).bounds)
    assert axs[0].get_legend().get_bbox_to_anchor() is not None
    plt.close(fig)


def test_the_fit_works_inside_a_locator_the_caller_already_set():
    """An axes whose place is decided by the caller's own locator: that
    locator keeps deciding, and the legend is fitted inside its answer."""
    from matplotlib.transforms import Bbox
    calls = []

    def callers_locator(ax, renderer):
        calls.append(1)
        return Bbox.from_bounds(0.1, 0.2, 0.7, 0.6)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axes_locator(callers_locator)
    hyp.plot(_three_clouds(2), 'o', ax=ax, legend=LONG_LABELS, show=False)
    assert _decoration_report(fig) == []
    drawn = ax.get_position()
    assert calls and abs(drawn.x0 - 0.1) < 1e-9 and abs(drawn.y0 - 0.2) < 1e-9
    assert drawn.width < 0.7 and abs(drawn.height - 0.6) < 1e-9
    plt.close(fig)


@pytest.mark.parametrize('ndims', [2, 3])
def test_a_colorbar_and_a_legend_on_caller_axes_stay_on_the_canvas(ndims):
    fig, axs = _caller_figure((1, 2), ndims, (11, 4))
    for ax in axs:
        hyp.plot(_three_clouds(ndims), 'o', ax=ax, legend=LONG_LABELS,
                 hue=list(np.repeat(['a', 'b', 'c'], 40)), colorbar=True,
                 show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axs:
        box = ax.get_legend().get_window_extent(renderer)
        assert box.x1 <= fig.bbox.x1 + 0.5
    plt.close(fig)


def test_a_legend_in_a_subfigure_stays_inside_the_subfigure():
    fig = plt.figure(figsize=(10, 4))
    left, right = fig.subfigures(1, 2)
    ax = left.add_subplot()
    other = right.add_subplot()
    hyp.plot(_three_clouds(2), 'o', ax=ax, legend=LONG_LABELS, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_legend().get_window_extent(renderer)
    assert box.x1 <= left.bbox.x1 + 0.5
    assert not box.overlaps(other.get_window_extent(renderer))
    plt.close(fig)


def test_a_fitted_caller_figure_pickles():
    import pickle
    fig, ax = plt.subplots(figsize=(5, 4))
    hyp.plot(_three_clouds(2), 'o', ax=ax, legend=LONG_LABELS, show=False)
    fig.canvas.draw()
    clone = pickle.loads(pickle.dumps(fig))
    assert _decoration_report(clone) == []
    np.testing.assert_allclose(clone.axes[0].get_position().bounds,
                               ax.get_position().bounds, atol=1e-9)
    plt.close(fig)
    plt.close(clone)


def test_hyp_subplots_grids_are_still_laid_out_by_each_plot_call():
    """A `hyp.subplots` grid is hypertools' own figure: the documented
    ``fig, axes = hyp.subplots(...); hyp.plot(d, ax=axes[i])`` loop keeps
    tightening it, exactly as in 1.1.0."""
    fig, axes = hyp.subplots(1, 2, size=[8, 4], backend='matplotlib')
    loose = _positions(fig)
    for ax in axes:
        hyp.plot(_walk(), ax=ax, show=False, backend='matplotlib')
    tightened = _positions(fig)
    assert tightened != loose
    # ...to exactly the layout a caller now gets by asking for it: the same
    # grid built with pyplot is left alone by `hyp.plot`, and one
    # `tight_layout()` per panel reproduces what 1.1.0 did unasked
    own, own_axes = plt.subplots(1, 2, figsize=(8, 4),
                                 subplot_kw={'projection': '3d'})
    for ax in own_axes:
        before = _positions(own)
        hyp.plot(_walk(), ax=ax, show=False, backend='matplotlib')
        assert _positions(own) == before
        own.tight_layout()
    # (to within a few pixels: the caller's `tight_layout()` measures the
    # tick labels outside hypertools' font context)
    np.testing.assert_allclose(_positions(own), tightened, atol=5e-3)
    assert not np.allclose(loose, tightened, atol=5e-2)
    plt.close(fig)
    plt.close(own)


def test_hypertools_own_figure_is_still_tightened():
    """Without `ax=` the figure is hypertools' own and is laid out as
    before."""
    fig = hyp.plot(_walk(), show=False, backend='matplotlib')
    tightened = fig.axes[0].get_position().bounds
    default, default_ax = plt.subplots(subplot_kw={'projection': '3d'})
    assert tightened != default_ax.get_position().bounds
    plt.close(fig)
    plt.close(default)


def test_a_second_call_into_hypertools_own_figure_still_fits_its_legend():
    """`fig = hyp.plot(x); hyp.plot(more, ax=fig.axes[0], legend=...)`: the
    figure is hypertools' own, so the second call lays it out and widens
    it for the legend exactly as 1.1.0 did."""
    fig = hyp.plot(_walk(), show=False, backend='matplotlib')
    width = fig.get_size_inches()[0]
    hyp.plot(_walk() + 1, ax=fig.axes[0], show=False,
             legend=['the second walk, overlaid'])
    assert _legend_report(fig, [fig.axes[0]]) == [(False, False)]
    assert fig.get_size_inches()[0] > width
    plt.close(fig)


def test_a_hyp_subplots_grid_is_tightened_whatever_figure_is_current():
    """The layout goes to the figure that owns the axes, not to pyplot's
    current figure: with another figure made current BEFORE the call, the
    grid is tightened and the bystander is not touched."""
    grid, axes = hyp.subplots(1, 2, size=[8, 4], backend='matplotlib')
    loose = _positions(grid)
    bystander, _ = plt.subplots(1, 2, figsize=(8, 4))
    bystander.subplots_adjust(left=0.3, right=0.6, wspace=0.9)
    before = _positions(bystander)
    assert plt.gcf() is bystander
    hyp.plot(_walk(), ax=axes[0], show=False, backend='matplotlib')
    assert _positions(bystander) == before
    assert not np.allclose(_positions(grid), loose, atol=1e-2)
    plt.close(grid)
    plt.close(bystander)


def test_backend_warning_is_raised_again_under_an_error_filter(
        backend_warning):
    """A warning turned into an error has not been delivered: it must not
    be recorded as shown and then pass silently the next time."""
    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(UserWarning, match='GH #290 test'):
                hyp.plot(_walk(), animate=True, show=True,
                         mpl_backend='disable')
        plt.close('all')


# ---------------------------------------------------------------------------
# GH #292: findfont log lines
# ---------------------------------------------------------------------------

class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.messages.append(record.getMessage())


@pytest.fixture
def findfont_log():
    """Every WARNING+ record matplotlib's font manager logs, with its
    `findfont` cache cleared first so earlier tests cannot hide a line."""
    handler = _Records()
    logger = logging.getLogger('matplotlib.font_manager')
    logger.addHandler(handler)
    font_manager.fontManager._findfont_cached.cache_clear()
    yield handler.messages
    logger.removeHandler(handler)
    plt.close('all')


@pytest.mark.parametrize('ndims', [2, 3])
def test_drawing_a_plot_logs_no_findfont_lines(findfont_log, ndims):
    """The issue's own reproduction, on whatever fonts this machine has.
    (On macOS with matplotlib >= 3.11 this logged the Hiragino line four
    times before the fix.)"""
    fig, ax = plt.subplots(
        subplot_kw={'projection': '3d'} if ndims == 3 else {})
    hyp.plot(_walk(), ax=ax, ndims=ndims, show=False, title='t')
    fig.canvas.draw()
    own = hyp.plot(_walk(), ndims=ndims, show=False, title='t',
                   legend=['walk'], backend='matplotlib')
    own.canvas.draw()
    assert findfont_log == []


def test_drawing_multibyte_text_logs_no_findfont_lines(findfont_log):
    """CJK text walks the whole fallback stack, one family per glyph."""
    fig = hyp.plot(_walk(), show=False, title='日本語のタイトル',
                   legend=['軌跡'], backend='matplotlib')
    fig.canvas.draw()
    assert [m for m in findfont_log if 'font weight' in m] == []


def test_every_default_stack_family_resolves_at_regular_weight(findfont_log):
    """What the stack promises matplotlib: each family hypertools adds on
    its own has a face matplotlib picks at weight 400 without comment."""
    for family in fonts.sans_serif_stack():
        font_manager.findfont(
            font_manager.FontProperties(family=[family]),
            fallback_to_default=False)
    assert findfont_log == []


@pytest.fixture
def light_only_family():
    """A family with no regular face, on any machine: the bundled Noto
    Sans Bold file (a real font file) registered a second time with the
    real font manager, under a family name of its own, as that family's
    300- and 600-weight faces -- the shape matplotlib >= 3.11 reports for
    macOS's 'Hiragino Kaku Gothic ProN'. The weights are stated for the
    test, not read from the file: Linux and Windows runners have no
    installed family of this shape. On macOS with matplotlib >= 3.11 the
    tests above exercise the real one."""
    name = 'Hypertools Test No Regular Face'
    bold = [p for p in fonts.bundled_font_files() if 'Bold' in p][0]
    fonts.register_bundled_fonts()
    manager = font_manager.fontManager
    entries = [font_manager.FontEntry(fname=bold, name=name, style='normal',
                                      variant='normal', weight=weight,
                                      stretch='normal', size='scalable')
               for weight in (300, 600)]
    manager.ttflist.extend(entries)
    manager._findfont_cached.cache_clear()
    fonts._PREFERRED_FAMILIES.insert(0, name)
    yield name
    fonts._PREFERRED_FAMILIES.remove(name)
    for entry in entries:
        manager.ttflist.remove(entry)
    manager._findfont_cached.cache_clear()


def test_a_family_without_a_regular_face_logs_when_named(
        light_only_family, findfont_log):
    """Control: this is the line the stack must not provoke. (matplotlib
    logs it from 3.11 on; earlier versions pick the face silently.)"""
    font_manager.findfont(
        font_manager.FontProperties(family=[light_only_family]),
        fallback_to_default=False)
    mpl_version = tuple(int(p) for p in matplotlib.__version__.split('.')[:2])
    if mpl_version >= (3, 11):
        assert findfont_log == [
            'findfont: Failed to find font weight normal for '
            f'{light_only_family}, now using 300.']


def test_stack_leaves_out_a_fallback_family_without_a_regular_face(
        light_only_family, findfont_log):
    assert light_only_family in fonts.installed_families()
    assert fonts._families_with_a_regular_face(
        [light_only_family, 'Noto Sans', 'DejaVu Sans']) == {
            'Noto Sans', 'DejaVu Sans'}
    stack = fonts.sans_serif_stack()
    assert light_only_family not in stack
    assert stack[0] == 'Noto Sans' and stack[-1] == 'DejaVu Sans'

    fig = hyp.plot(_walk(), show=False, title='t', backend='matplotlib')
    fig.canvas.draw()
    assert findfont_log == []


def test_stack_keeps_such_a_family_when_it_is_asked_for_or_needed(
        light_only_family):
    """`first` is the caller's explicit `font=`; `extra` is the only font
    found that covers the text. Neither is hypertools' to drop."""
    assert fonts.sans_serif_stack(first=light_only_family)[0] == (
        light_only_family)
    needed = fonts.sans_serif_stack(extra=light_only_family)
    assert light_only_family in needed
    assert needed[0] == 'Noto Sans' and needed[-1] == 'DejaVu Sans'


def test_regular_face_check_follows_matplotlibs_own_choice_of_face():
    """A family whose only 400-weight face is italic resolves, for upright
    text, to another weight -- matplotlib weighs style above weight -- so it
    is left out too; one with an upright regular face among others stays."""
    name = 'Hypertools Test Italic Regular Only'
    regular = [p for p in fonts.bundled_font_files() if 'Regular' in p][0]
    manager = font_manager.fontManager
    entries = [
        font_manager.FontEntry(fname=regular, name=name, style='italic',
                               variant='normal', weight=400,
                               stretch='normal', size='scalable'),
        font_manager.FontEntry(fname=regular, name=name, style='normal',
                               variant='normal', weight=700,
                               stretch='normal', size='scalable'),
    ]
    manager.ttflist.extend(entries)
    try:
        assert fonts._families_with_a_regular_face([name]) == set()
        upright = font_manager.FontEntry(
            fname=regular, name=name, style='normal', variant='normal',
            weight='regular', stretch='normal', size='scalable')
        manager.ttflist.append(upright)
        entries.append(upright)
        assert fonts._families_with_a_regular_face([name]) == {name}
    finally:
        for entry in entries:
            manager.ttflist.remove(entry)
        manager._findfont_cached.cache_clear()
