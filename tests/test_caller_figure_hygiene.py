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


def test_a_legend_on_caller_axes_without_an_engine_is_still_fitted():
    """Unchanged by GH #291: a right-side legend that would overflow the
    canvas still widens the figure, so it is not clipped on save."""
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': '3d'})
    hyp.plot(_walk(), ax=ax, show=False,
             legend=['a very long legend label that cannot fit beside it'])
    fig.canvas.draw()
    legend_box = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    assert legend_box.x1 <= fig.bbox.x1 + 1
    assert fig.get_size_inches()[0] > 4.0
    plt.close(fig)


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


def test_hypertools_own_figure_is_still_tightened_whatever_is_current():
    """Without `ax=` the figure is hypertools' own and is laid out as
    before -- on that figure, not on whichever one pyplot calls current."""
    fig = hyp.plot(_walk(), show=False, backend='matplotlib')
    ax = fig.axes[0]
    tightened = ax.get_position().bounds
    plt.figure(fig)
    default, default_ax = plt.subplots(subplot_kw={'projection': '3d'})
    assert tightened != default_ax.get_position().bounds
    plt.close(fig)
    plt.close(default)


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
    """A REAL installed family with no regular face, on any machine: the
    bundled Noto Sans Bold file registered a second time under its own
    family name, as the 300- and 600-weight faces of one family -- the shape
    matplotlib >= 3.11 reports for macOS's 'Hiragino Kaku Gothic ProN'."""
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
