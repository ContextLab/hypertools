import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.morph import segment_frame_counts, frame_to_segment


def _datasets(n=3, rows=20, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _clouds(n=3, pts=120, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(pts, 3)) + off for off in np.arange(n) * 4.0]


def _titles_over(ani, fig, n):
    threed = [a for a in fig.axes if hasattr(a, 'zaxis')]
    ax = threed[0] if threed else fig.axes[0]
    seen = []
    for f in range(n):
        ani._func(f, *ani._args)
        seen.append(ax.get_title())
    return seen


# --- serial reveal ----------------------------------------------------------

def test_title_list_tracks_the_revealed_dataset():
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        title=['first', 'second', 'third'],
                        duration=4, frame_rate=4, show=False)
    seen = _titles_over(ani, fig, 16)
    assert seen[0] == 'first', 'frame 0 must title the FIRST dataset'
    assert set(seen) <= {'first', 'second', 'third'}
    assert seen.index('second') < seen.index('third')


def test_title_list_matches_the_published_current_index():
    """The title must be driven by the same schedule on_frame publishes."""
    names = ['first', 'second', 'third']
    seen_ctx = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        title=names, duration=4, frame_rate=4,
                        on_frame=seen_ctx.append, show=False)
    titles = _titles_over(ani, fig, 16)
    assert [names[c.current_index] for c in seen_ctx] == titles


def test_title_list_length_must_match_dataset_count():
    with pytest.raises(ValueError, match='title has 2 entries'):
        hyp.plot(_datasets(), '-', animate=True, order='serial',
                 title=['a', 'b'], duration=2, frame_rate=2, show=False)


def test_scalar_title_is_constant_across_a_serial_animation():
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        title='constant', duration=2, frame_rate=4,
                        show=False)
    assert set(_titles_over(ani, fig, 8)) == {'constant'}


def test_title_list_still_rejected_for_parallel_animations():
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(_datasets(), '-', animate=True, title=['a', 'b', 'c'],
                 duration=2, frame_rate=2, show=False)


def test_title_list_still_rejected_for_static_plots():
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(_datasets(), '-', title=['a', 'b', 'c'], show=False)


def test_title_list_works_for_2d_serial_animations():
    fig, ani = hyp.plot(_datasets(), '-', ndims=2, animate=True,
                        order='serial', title=['first', 'second', 'third'],
                        duration=4, frame_rate=4, show=False)
    seen = _titles_over(ani, fig, 16)
    assert seen[0] == 'first' and 'third' in seen


# --- morph: holds named, transitions blank ---------------------------------

def test_morph_titles_follow_the_hold_transition_schedule_exactly():
    """C9: derived from frame_to_segment, so it CANNOT pass under v1's
    fraction rule (which blanked 12 of 15 hold frames and named 4 transition
    frames while still landing at blank_fraction == 0.5)."""
    names = ['alpha', 'beta', 'gamma']
    fig, ani = hyp.plot(_clouds(), '.', animate='morph', title=names,
                        morph_samples=120, duration=6, frame_rate=4,
                        show=False)
    counts = segment_frame_counts(3, 24)
    assert counts == [5, 5, 5, 5, 4]
    seen = _titles_over(ani, fig, sum(counts))
    for frame, title in enumerate(seen):
        seg, step, n_steps = frame_to_segment(counts, frame)
        expected = names[seg // 2] if seg % 2 == 0 else ''
        assert title == expected, (frame, seg, step, n_steps, title)


def test_every_interior_transition_frame_is_blank():
    """The weaker property stated on its own, so intent stays legible."""
    fig, ani = hyp.plot(_clouds(), '.', animate='morph',
                        title=['alpha', 'beta', 'gamma'], morph_samples=120,
                        duration=6, frame_rate=4, show=False)
    counts = segment_frame_counts(3, 24)
    seen = _titles_over(ani, fig, sum(counts))
    interiors = [f for f in range(len(seen))
                 if frame_to_segment(counts, f)[0] % 2 == 1
                 and 0 < frame_to_segment(counts, f)[1]
                 < frame_to_segment(counts, f)[2] - 1]
    assert interiors, 'the schedule must contain interior transition frames'
    assert all(seen[f] == '' for f in interiors)


def test_every_hold_frame_is_named():
    fig, ani = hyp.plot(_clouds(), '.', animate='morph',
                        title=['alpha', 'beta', 'gamma'], morph_samples=120,
                        duration=6, frame_rate=4, show=False)
    counts = segment_frame_counts(3, 24)
    seen = _titles_over(ani, fig, sum(counts))
    holds = [f for f in range(len(seen))
             if frame_to_segment(counts, f)[0] % 2 == 0]
    assert len(holds) == 14
    assert all(seen[f] != '' for f in holds)


# --- partial-tag morph: titles must follow morph_tags, not segment position
#
# whole-branch review, Important finding 1: `plot([a, b, c],
# animate=[None, 'morph', 'morph'], title=['a', 'b', 'c'])` morphs datasets
# 1 and 2 -- but the title updater indexed by SEGMENT POSITION
# (`seg_idx // 2`), never through `morph_tags`, so the hold titles came out
# 'a' then 'b' (dataset 0 -- untagged, never shown -- and dataset 1) while
# 'c' (the actual second hold) was unreachable. Every test above uses a
# scalar animate='morph' (every dataset tagged), where segment position and
# final dataset index coincide by construction, so none of them exercise
# this gap.

def test_partial_tag_morph_titles_name_the_actual_dataset():
    names = ['a', 'b', 'c']
    fig, ani = hyp.plot(_clouds(n=3), '.', animate=[None, 'morph', 'morph'],
                        title=names, morph_samples=120, duration=8,
                        frame_rate=1, show=False)
    counts = segment_frame_counts(2, 8)   # 2 TAGGED datasets -> 3 segments
    seen = _titles_over(ani, fig, sum(counts))
    hold_titles = {t for t in seen if t != ''}
    assert hold_titles == {'b', 'c'}, (
        "a partial-tag morph must only ever title the TAGGED datasets "
        f"('b', 'c'); dataset 'a' is untagged and never shown -- saw "
        f"{hold_titles}")
    assert seen[0] == 'b', 'the first hold must name the FIRST TAGGED dataset'


def test_partial_tag_morph_titles_match_across_backends():
    """Parity companion to the matplotlib-only test above: the same bug
    (segment position instead of morph_tags) was independently present in
    the plotly backend's own per-frame title lookup."""
    pytest.importorskip('plotly')
    names = ['a', 'b', 'c']
    clouds = _clouds(n=3)
    fig, ani = hyp.plot(clouds, '.', animate=[None, 'morph', 'morph'],
                        title=names, morph_samples=120, duration=8,
                        frame_rate=1, show=False)
    mpl_titles = _titles_over(ani, fig, sum(segment_frame_counts(2, 8)))

    hyp.set_interactive_backend('plotly')
    try:
        pfig = hyp.plot(clouds, '.', animate=[None, 'morph', 'morph'],
                        title=names, morph_samples=120, duration=8,
                        frame_rate=1, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    ply_titles = [f.layout.title.text for f in pfig.frames]

    assert ply_titles == mpl_titles
    assert {t for t in ply_titles if t} == {'b', 'c'}


# --- backend parity ---------------------------------------------------------

def test_serial_titles_render_on_plotly_frames():
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=True, order='serial',
                       title=['first', 'second', 'third'],
                       duration=4, frame_rate=4, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    titles = [f.layout.title.text for f in fig.frames]
    assert titles[0] == 'first'
    assert set(titles) <= {'first', 'second', 'third'}
    assert titles.index('second') < titles.index('third')


def test_morph_titles_match_across_backends():
    """The same call must produce the same per-frame title sequence."""
    pytest.importorskip('plotly')
    names = ['alpha', 'beta', 'gamma']
    fig, ani = hyp.plot(_clouds(), '.', animate='morph', title=names,
                        morph_samples=120, duration=6, frame_rate=4,
                        show=False)
    mpl_titles = _titles_over(ani, fig, 24)

    hyp.set_interactive_backend('plotly')
    try:
        pfig = hyp.plot(_clouds(), '.', animate='morph', title=names,
                        morph_samples=120, duration=6, frame_rate=4,
                        show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    ply_titles = [f.layout.title.text for f in pfig.frames]
    assert ply_titles == mpl_titles


def test_serial_titles_compose_with_chemtrails():
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        chemtrails=True, title=['first', 'second', 'third'],
                        duration=4, frame_rate=4, show=False)
    seen = _titles_over(ani, fig, 16)
    assert seen[0] == 'first' and 'third' in seen


# --- plotly top margin (task-8 review, "MARGIN CONCERN: FUNCTIONAL DEFECT")-
#
# `plotly_backend.py`'s figure-level top margin used to key off the STATIC
# `title` alone (`t=40 if title else 10`). `plot.py` nulls `title` for
# segment-titled serial/morph animations -- the title is drawn PER FRAME
# instead (see the 'morph'/'serial' branches of `_add_animation`) -- so
# every one of these figures fell back to the "no title" t=10 margin even
# though a title renders on most frames. A real kaleido render proved that
# clips the title text at the canvas top edge (see the pixel-level test
# below); these layout-margin checks would all FAIL under the pre-fix rule,
# which produced t=10 for a segment-titled animation.

def test_segment_titled_serial_animation_reserves_the_title_margin():
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=True, order='serial',
                       title=['first', 'second', 'third'],
                       duration=4, frame_rate=4, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert fig.layout.margin.t == 40, (
        "a segment-titled plotly animation must reserve the SAME top "
        "margin as a statically-titled figure -- a per-frame title still "
        "renders on every hold frame even though the static `title` is "
        "None")


def test_segment_titled_margin_matches_a_scalar_titled_margin():
    """Not just non-default -- the reserved margin must match a plain
    scalar-titled figure's, since both draw an identical 12pt title."""
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        scalar = hyp.plot(_datasets(), '-', animate=True, order='serial',
                          title='constant', duration=4, frame_rate=4,
                          show=False)
        segment = hyp.plot(_datasets(), '-', animate=True, order='serial',
                           title=['first', 'second', 'third'],
                           duration=4, frame_rate=4, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert segment.layout.margin.t == scalar.layout.margin.t == 40


def test_titleless_serial_animation_keeps_the_smaller_margin():
    """Regression guard on the fix itself: it must key off whether a
    per-frame title will actually be drawn (`segment_titles`), not just
    reserve a title margin for every serial-style animation regardless of
    whether `title=` was ever passed."""
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=True, order='serial',
                       duration=4, frame_rate=4, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert fig.layout.margin.t == 10


def test_segment_titled_hold_frame_title_is_not_clipped_at_canvas_top(
        tmp_path):
    """Real kaleido PNG evidence, not just a layout-dict assertion (task-8
    review: "a layout-dict assertion alone does not prove the pixels").

    `_frame_snapshots` is the same helper the real GIF/video export path
    (`_export_animation_file`) uses to turn one animation frame into a
    static, standalone `go.Figure` -- applying it to frame 0 of a serial
    reveal gives the first HOLD frame, titled 'first' (nothing has been
    revealed yet, so `serial_current_index` picks dataset 0; see
    `test_title_list_tracks_the_revealed_dataset` above). Rendering that
    snapshot for real and inspecting the actual pixels is the only way to
    prove the title text isn't cut off.

    Measured concretely (this file's fix pass): at the pre-fix t=10 margin,
    row 0 of the rendered 640x480 canvas already contained dark ink (min
    RGB channel value 42 -- the title glyphs start being drawn AT the
    canvas edge, i.e. clipped). With the margin reserved (t=40, this test),
    row 0 is clean background (min channel value 255) and the title's ink
    only begins at row 6, safely inside the canvas.
    """
    pytest.importorskip('plotly')
    from PIL import Image

    from hypertools.plot.plotly_backend import _frame_snapshots

    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=True, order='serial',
                       title=['first', 'second', 'third'],
                       duration=4, frame_rate=4, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')

    snapshot = next(iter(_frame_snapshots(fig)))
    assert snapshot.layout.title.text == 'first', (
        'sanity check: this must be the real titled hold frame (frame 0), '
        'not an untitled one -- otherwise the pixel check below would '
        'trivially pass for the wrong reason')

    png_path = str(tmp_path / 'hold_frame.png')
    snapshot.write_image(png_path, width=640, height=480)
    arr = np.asarray(Image.open(png_path).convert('RGB'))

    assert arr[0].min() >= 250, (
        f"canvas row 0 contains ink (min RGB channel value "
        f"{arr[0].min()}) -- the hold-frame title text is being clipped "
        "at the canvas top edge instead of sitting inside the reserved "
        "top margin")
    # and the title text must actually be drawn SOMEWHERE in that reserved
    # band -- otherwise a "fix" that reserved the margin but silently
    # stopped drawing the per-frame title would also pass the check above,
    # for the wrong reason
    top_band = arr[:40]
    assert top_band.min() < 250, (
        "no ink anywhere in the reserved top-margin band (rows 0-39) -- "
        "the hold-frame title does not appear to be rendering at all")


def test_plotly_title_and_on_frame_stay_in_sync():
    """Task-8 review, minor finding: the plotly 'serial' branch used to call
    `serial_current_index(_shown, lengths)` separately for `segment_titles`
    and for `frame_hooks` -- same arguments, byte-identical results, so
    purely duplicate work -- now computed once and shared. This locks in
    that the two consumers still agree post-dedup, on the PLOTLY backend
    specifically: the existing matplotlib-only version of this check
    (`test_title_list_matches_the_published_current_index` above) never
    exercised the plotly code path this finding was about."""
    pytest.importorskip('plotly')
    names = ['first', 'second', 'third']
    seen_ctx = []
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=True, order='serial',
                       title=names, duration=4, frame_rate=4,
                       on_frame=seen_ctx.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    titles = [f.layout.title.text for f in fig.frames]
    assert len(seen_ctx) == len(fig.frames) > 0
    assert [names[c.current_index] for c in seen_ctx] == titles
