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
