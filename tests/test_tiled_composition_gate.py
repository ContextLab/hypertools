# -*- coding: utf-8 -*-
"""Does the tiled-composition gate DISCRIMINATE?

`test_examples_are_native.py` gained `_assert_tiled_composition`, which
encodes the defining properties of Plan v5 criterion 3(b): panels laid out
in the data, one call, one animation, one pooled scaling, and a hierarchy
whose parents are real means. That gate cannot run against the Market
example until Task 2 rewrites it, so without this file it would sit
unexercised until then -- and the same gate's own docstring records what
happened last time an unexercised gate shipped: `_save_count >= 1` and
`'morph' in 'morph'` were both tautologies that could not fail.

So the gate is checked here against a composition built to satisfy it, and
against four mutations that each break exactly one of its claims.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
import pytest                                                  # noqa: E402

import hypertools as hyp                                       # noqa: E402
# `tests/` IS a package (it has an `__init__.py`), so the gate module is
# imported by its dotted name rather than as a bare top-level module.
from tests.test_examples_are_native import (                   # noqa: E402
    _assert_tiled_composition, _path_xy, _tiled_panels)

PANELS = ['Alpha', 'Beta', 'Gamma']
LEAVES = 2
COLOURS = ['#c1272d', '#1b7f4f', '#2d5fa8']
STEP = 2.6
SPEC = dict(panels=len(PANELS), leaves=LEAVES)


def _tiled_frame(step=STEP, interleaved=False):
    """Three panel groups, each translated into its own region of one box.

    `interleaved=True` orders the columns leaf-major instead of panel-major
    -- a real way for a rewrite to go wrong, and the one the gate's
    leaf-attribution assumption exists to catch.
    """
    rng = np.random.default_rng(0)
    walks = {(panel, leaf): np.cumsum(rng.standard_normal((12, 2)) * 0.2,
                                      axis=0)
             + np.array([(index % 3) * step, -(index // 3) * step])
             for index, panel in enumerate(PANELS) for leaf in range(LEAVES)}
    order = ([(panel, leaf) for leaf in range(LEAVES) for panel in PANELS]
             if interleaved else
             [(panel, leaf) for panel in PANELS for leaf in range(LEAVES)])
    columns, blocks = [], []
    for panel, leaf in order:
        blocks.append(walks[(panel, leaf)])
        columns += [(panel, f'leaf{leaf}', 'x'), (panel, f'leaf{leaf}', 'y')]
    return pd.DataFrame(np.hstack(blocks), columns=pd.MultiIndex.from_tuples(
        columns, names=['Panel', 'Leaf', 'Measure']))


def _compose(frame, step=STEP, box_scale=1.0, overlap=False,
             uneven_boxes=False):
    """One `hyp.plot` call, then the panel rectangles drawn as annotations.

    The cells are ONE size -- the largest panel's content, used for every
    panel -- because that is the claim the gate checks: panels differ in
    what they hold, not in how much room they get.
    """
    anim = hyp.plot(frame, '-', palette=COLOURS, reduce=None, ndims=2,
                    normalize=None, animate='parallel', duration=2,
                    frame_rate=45, colorbar=False, show=False,
                    frame_kwargs={'visible': False},
                    title='three panels and their means')
    ax = anim.figure.axes[0]
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    anim.animation._func(anim.n_frames - 1, *anim.animation._args)
    panels = _tiled_panels(anim.figure, len(PANELS), LEAVES)
    pad = 0.02
    content = {name: np.vstack([_path_xy(a) for a in (*leaves, parent)])
               for name, (leaves, parent) in panels.items()}
    cell = (max(np.ptp(pts[:, 0]) for pts in content.values()) + 2 * pad,
            max(np.ptp(pts[:, 1]) for pts in content.values()) + 2 * pad)
    for name in PANELS:
        pts = content[name]
        if uneven_boxes:
            size = (np.ptp(pts[:, 0]) + 2 * pad, np.ptp(pts[:, 1]) + 2 * pad)
        else:
            size = cell
        width, height = size[0] * box_scale, size[1] * box_scale
        centre = ((pts[:, 0].min() + pts[:, 0].max()) / 2,
                  (pts[:, 1].min() + pts[:, 1].max()) / 2)
        if overlap:
            centre = (0.0, 0.0)
        ax.add_patch(plt.Rectangle(
            (centre[0] - width / 2, centre[1] - height / 2), width, height,
            fill=False, ec='#cccccc', lw=0.8))
    limits = (ax.get_xlim(), ax.get_ylim())

    def hold(_context):
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])

    anim.on_frame(hold)
    return anim


@pytest.fixture
def clean_figures():
    yield
    plt.close('all')


def test_a_composition_built_to_the_contract_PASSES(clean_figures):
    """The positive control. Without it, every failure below could be the
    gate rejecting everything rather than rejecting the right things."""
    _assert_tiled_composition('probe', _compose(_tiled_frame()), SPEC)


def test_it_catches_PANELS_GIVEN_DIFFERENT_AMOUNTS_OF_ROOM(clean_figures):
    """Cells sized to each panel's own content instead of to one shared
    cell: a reader comparing two panels would be comparing two scales.

    Note what this canNOT catch, which the gate says out loud: a per-panel
    RESCALE of the source data renders identically to a panel whose data
    legitimately spans that much. Nothing in the figure separates them, so
    pooled scaling is held by construction rather than by this gate."""
    with pytest.raises(AssertionError, match='differ in size'):
        _assert_tiled_composition(
            'probe', _compose(_tiled_frame(), uneven_boxes=True), SPEC)


def test_it_catches_OVERLAPPING_panels(clean_figures):
    """Translations too small to separate the panels make a neighbour's path
    read as this panel's."""
    with pytest.raises(AssertionError, match='overlap|outside every panel'):
        _assert_tiled_composition(
            'probe', _compose(_tiled_frame(), overlap=True), SPEC)


def test_it_catches_a_path_ESCAPING_its_box(clean_figures):
    """A box drawn smaller than the paths it is supposed to contain."""
    with pytest.raises(AssertionError, match='outside every panel box'):
        _assert_tiled_composition(
            'probe', _compose(_tiled_frame(), box_scale=0.5), SPEC)


def test_it_catches_LEAVES_THAT_BELONG_TO_ANOTHER_PANEL(clean_figures):
    """The hierarchy claim, and the same check that proves the leaves were
    attributed to the right panels.

    The mutation is a REAL one rather than a nudged artist: a nudge would
    be wiped by the next frame the gate drives, since the animated backend
    rebuilds every path it draws. Ordering the frame's columns leaf-major
    instead of panel-major is how a rewrite would actually break this --
    the parents stay correct, the leaves stop being contiguous per panel,
    and each panel is then handed a mixture of three panels' leaves.
    """
    with pytest.raises(AssertionError,
                       match='does not start and end at|mixing traces'):
        _assert_tiled_composition(
            'probe', _compose(_tiled_frame(interleaved=True)), SPEC)


def test_it_catches_FORECAST_prose_the_plan_retired(clean_figures):
    anim = _compose(_tiled_frame())
    anim.figure.axes[0].set_title('sector forecast accuracy')
    with pytest.raises(AssertionError, match='still says'):
        _assert_tiled_composition('probe', anim, SPEC)


def test_it_catches_a_HOOK_THAT_MOVES_THE_VIEW(clean_figures):
    """The panel view must be the same at every frame as the one the
    example left behind, or the boxes and the paths stop lining up.

    The mutation is an `on_frame` callback that moves the view -- NOT a
    claim that the backend resets styling by itself. An earlier version of
    this file asserted that, and it was wrong: measured, axis limits, spine
    visibility and patch visibility all survive both `draw_frame` and
    `save`. A test for a reset that does not happen would be exactly the
    tautology the gate's own docstring warns about, so this checks the
    property that genuinely can break instead.
    """
    anim = _compose(_tiled_frame())
    ax = anim.figure.axes[0]

    def wander(_context):
        ax.set_xlim(-0.5, 0.5)

    anim.on_frame(wander)
    with pytest.raises(AssertionError, match='the view moved'):
        _assert_tiled_composition('probe', anim, SPEC)
