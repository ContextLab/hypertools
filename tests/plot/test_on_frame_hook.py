import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.animation_context import FrameContext


def _datasets(n=3, rows=20, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _axes_of(fig):
    """Works for 2-D and 3-D: an animated 2-D figure has exactly one axes
    and it has no `zaxis` (measured)."""
    threed = [a for a in fig.axes if hasattr(a, 'zaxis')]
    return threed[0] if threed else fig.axes[0]


def _drive(ani, n):
    for f in range(n):
        ani._func(f, *ani._args)


# --- basics -----------------------------------------------------------------

def test_on_frame_is_called_once_per_frame():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=2,
                        frame_rate=4, on_frame=seen.append, show=False)
    _drive(ani, 8)
    assert len(seen) == 8
    assert all(isinstance(ctx, FrameContext) for ctx in seen)


def test_frame_context_reports_frame_and_total():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=2,
                        frame_rate=4, on_frame=seen.append, show=False)
    _drive(ani, 8)
    assert [ctx.frame for ctx in seen] == list(range(8))
    assert {ctx.n_frames for ctx in seen} == {8}


def test_frame_context_exposes_figure_axes_artists_and_datasets():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=2, on_frame=seen.append, show=False)
    _drive(ani, 2)
    ctx = seen[-1]
    assert ctx.figure is fig
    assert ctx.axes is _axes_of(fig)
    assert len(ctx.artists) >= 3
    assert len(ctx.datasets) == 3


def test_parallel_mode_reports_no_serial_position():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=2, on_frame=seen.append, show=False)
    _drive(ani, 2)
    assert all(ctx.current_index is None for ctx in seen)
    assert all(ctx.current_fraction is None for ctx in seen)
    assert all(ctx.order == 'parallel' for ctx in seen)
    # `revealed_counts` USED to be None here as well. Since GH #285 it is
    # populated on the parallel/'window'/'spin' paths too (a parallel
    # reveal has a head; what it does not have is a serial POSITION), so
    # only the serial-position fields say "not applicable" now. See
    # tests/test_anim_progress.py for the counts' own contract.
    assert all(ctx.revealed_counts is not None for ctx in seen)
    assert all(len(ctx.revealed_counts) == 3 for ctx in seen)


def test_frame_context_style_docstring_lists_parallel():
    """Minor finding (whole-branch review): animate='parallel' is a real,
    runtime-reachable value of ctx.style (see the previous test's
    animate=True, which resolves to the same backend mode), but the
    FrameContext.style docstring only enumerated True/'serial'/'spin'/
    'window'/'morph' -- 'parallel' was undocumented."""
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate='parallel', duration=1,
                        frame_rate=2, on_frame=seen.append, show=False)
    _drive(ani, 2)
    assert all(ctx.style == 'parallel' for ctx in seen)

    doc = FrameContext.__doc__
    style_doc = doc.split('style : bool or str', 1)[1].split('order :', 1)[0]
    assert "'parallel'" in style_doc, (
        f"FrameContext.style docstring omits 'parallel': {style_doc!r}")


def test_frame_context_is_exported_at_top_level_but_frame_hooks_is_not():
    """`FrameContext` is public: users receive one per callback and will
    annotate and isinstance-check it. `FrameHooks` is the internal registry
    from contract 3 -- users never construct one, so it stays off the
    curated surface that `hypertools/__init__.py:43-52` maintains."""
    assert hyp.FrameContext is FrameContext
    assert 'FrameContext' in hyp.__all__
    assert not hasattr(hyp, 'FrameHooks')
    assert 'FrameHooks' not in hyp.__all__


# --- the serial schedule ----------------------------------------------------

def test_serial_schedule_is_exposed_so_callers_need_not_re_derive_it():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        duration=4, frame_rate=4, on_frame=seen.append,
                        show=False)
    _drive(ani, 16)
    indices = [ctx.current_index for ctx in seen]
    assert indices[0] == 0, 'frame 0 must report the FIRST dataset'
    assert max(indices) == 2
    assert indices == sorted(indices), 'serial reveal advances monotonically'
    for ctx in seen:
        assert ctx.order == 'serial'
        assert 0.0 <= ctx.current_fraction <= 1.0
        assert len(ctx.revealed_counts) == 3
        assert sum(ctx.revealed_counts) <= sum(d.shape[0] for d in ctx.datasets)


def test_revealed_counts_match_the_drawn_artists_with_unequal_lengths():
    """Exercises the UNEQUAL-length branch of the reveal split.

    A LINE format pre-interpolates every animated dataset onto the frame grid
    (measured: input [17, 23, 11] -> [13, 13, 13]), so only a MARKER format
    keeps them unequal. Asserted against the artists themselves, not against
    a second copy of the formula.

    `revealed_counts` is a TUPLE (FrameContext.__post_init__ canonicalizes
    it), so `drawn` is compared as a tuple -- `(17, 4, 0) == [17, 4, 0]` is
    False and this assertion would fail for the wrong reason otherwise.
    """
    seen = []
    ds = [np.random.default_rng(s).normal(size=(n, 4)).cumsum(axis=0)
          for s, n in enumerate((17, 23, 11))]
    fig, ani = hyp.plot(ds, '.', animate=True, order='serial', duration=13,
                        frame_rate=1, on_frame=seen.append, show=False)
    _drive(ani, 13)
    ax = _axes_of(fig)
    assert [d.shape[0] for d in seen[-1].datasets] == [17, 23, 11]
    # DEVIATION from the brief's verbatim `for ctx in seen:` (documented in
    # the Task 7 report, with evidence): `on_frame=seen.append` is still
    # attached, so each `ani._func(...)` call below re-dispatches the hook
    # and appends ANOTHER context to `seen` -- the very list this loop is
    # iterating. That makes the loop self-feeding: it consumes one item and
    # appends one item every iteration, so it never terminates. Measured:
    # the unmodified loop pinned one CPU core at 100% for 3+ minutes with
    # `seen` growing without bound before being killed. Iterating a snapshot
    # (`list(seen)`) fixes the loop bound to the 13 frames already recorded,
    # while every assertion below is unchanged.
    for ctx in list(seen):
        ani._func(ctx.frame, *ani._args)
        drawn = tuple(len(ln.get_data_3d()[0]) for ln in ax.lines[:3])
        assert ctx.revealed_counts == drawn


def test_serial_current_fraction_completes_each_dataset_before_the_next():
    """DEVIATION from the brief's verbatim ``duration=13, frame_rate=1``
    (documented in the Task 7 report, with evidence): `revealed(num) =
    sum(lengths) * num / (total_frames - 1)` steps by 51/12 = 4.25 points
    per frame for lengths (17, 23, 11), and the dataset-0/dataset-1
    boundary sits at revealed == 40 -- which is not an integer multiple of
    4.25 (40 * 12 / 51 = 9.41...), so NO integer frame ever lands exactly
    on "dataset 1 fully revealed, dataset 2 not yet started". Verified with
    exact (`fractions.Fraction`) arithmetic across all 13 frames: the
    dataset-0 boundary (revealed == 17) IS hit exactly at frame 4 (lucky:
    17 is exactly total/3, and total_frames - 1 = 12 is divisible by 3),
    but the dataset-1 boundary is not, and the real `ani._func` run below
    reproducibly reports ``max(by_index[1]) == 0.90909...`` rather than
    ``1.0`` -- a property of these specific numbers, not a bug in
    `serial_current_index` (which is a pure, stateless function of one
    frame's `(counts, lengths)` and cannot special-case a boundary its
    inputs never actually reach).

    Fix: choose `total_frames = sum(lengths) + 1` instead, so
    `total_frames - 1 == sum(lengths)` and `revealed(num) == num` exactly
    for every integer frame -- every dataset boundary (17 and 40) is then
    hit by an exact integer frame. Same dataset shapes as the neighboring
    unequal-lengths test; only the frame count changes.
    """
    seen = []
    lengths = (17, 23, 11)
    ds = [np.random.default_rng(s).normal(size=(n, 4)).cumsum(axis=0)
          for s, n in enumerate(lengths)]
    total_frames = sum(lengths) + 1
    fig, ani = hyp.plot(ds, '.', animate=True, order='serial',
                        duration=total_frames, frame_rate=1,
                        on_frame=seen.append, show=False)
    _drive(ani, total_frames)
    by_index = {}
    for ctx in seen:
        by_index.setdefault(ctx.current_index, []).append(ctx.current_fraction)
    for idx in (0, 1):
        assert max(by_index[idx]) == pytest.approx(1.0)


# --- morph segments ---------------------------------------------------------

def test_morph_reports_segment_index_and_kind():
    """C8: holds and transitions BOTH sweep current_fraction 0->1, so the
    kind must be an explicit field, derived from morph.frame_to_segment."""
    from hypertools.plot.morph import segment_frame_counts, frame_to_segment
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0, 8.0)]
    seen = []
    fig, ani = hyp.plot(clouds, '.', animate='morph', morph_samples=120,
                        duration=6, frame_rate=4, on_frame=seen.append,
                        show=False)
    counts = segment_frame_counts(3, 24)
    assert counts == [5, 5, 5, 5, 4]
    _drive(ani, sum(counts))
    assert len(seen) == 24
    for ctx in seen:
        seg, _step, _n = frame_to_segment(counts, ctx.frame)
        assert ctx.segment_index == seg
        assert ctx.segment_kind == ('hold' if seg % 2 == 0 else 'transition')
        assert ctx.current_index == seg // 2


def test_morph_holds_and_transitions_are_not_separable_by_fraction_alone():
    """Documents WHY segment_kind exists: both kinds span the same range."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0, 8.0)]
    seen = []
    fig, ani = hyp.plot(clouds, '.', animate='morph', morph_samples=120,
                        duration=6, frame_rate=4, on_frame=seen.append,
                        show=False)
    _drive(ani, 24)
    holds = {round(c.current_fraction, 3) for c in seen
             if c.segment_kind == 'hold'}
    moves = {round(c.current_fraction, 3) for c in seen
             if c.segment_kind == 'transition'}
    assert holds & moves, 'fractions overlap, so they cannot discriminate'


def test_partial_tag_morph_current_index_only_names_tagged_datasets():
    """Important finding 1 (whole-branch review): a partial-tag morph list
    (`animate=[None, 'morph', 'morph']`) used to report `ctx.current_index`
    as `segment_index // 2` -- a position WITHIN THE MORPH SEQUENCE, not
    the FINAL dataset index -- so with dataset 0 untagged, current_index
    took values {0, 1} (dataset 0 is never the one shown) instead of the
    correct {1, 2}. `test_morph_reports_segment_index_and_kind` above only
    ever used a scalar animate='morph' (every dataset tagged), where
    sequence position and final index coincide by construction and so
    cannot catch this. Checked on both backends."""
    pytest.importorskip('plotly')
    ds = _datasets(n=3)

    mpl_seen = []
    fig, ani = hyp.plot(ds, '.', animate=[None, 'morph', 'morph'],
                        duration=8, frame_rate=1, on_frame=mpl_seen.append,
                        show=False)
    _drive(ani, mpl_seen[0].n_frames if mpl_seen else 8)
    mpl_indices = {ctx.current_index for ctx in mpl_seen}
    assert mpl_indices == {1, 2}, (
        "partial-tag morph (animate=[None,'morph','morph']) must only "
        f"report the TAGGED dataset indices {{1, 2}}; saw {mpl_indices}")

    ply_seen = []
    hyp.set_interactive_backend('plotly')
    try:
        hyp.plot(ds, '.', animate=[None, 'morph', 'morph'], duration=8,
                 frame_rate=1, on_frame=ply_seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    ply_indices = {ctx.current_index for ctx in ply_seen}
    assert ply_indices == {1, 2}, (
        "partial-tag morph must only report the TAGGED dataset indices "
        f"{{1, 2}} on plotly too; saw {ply_indices}")


# --- 2-D --------------------------------------------------------------------

def test_hook_fires_for_2d_animations():
    """Every v1 helper did `[a for a in fig.axes if hasattr(a, 'zaxis')][0]`,
    which raises IndexError on a 2-D figure (measured: zaxis? [False])."""
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', ndims=2, animate=True,
                        order='serial', duration=2, frame_rate=4,
                        on_frame=seen.append, show=False)
    _drive(ani, 8)
    assert len(seen) == 8
    assert not hasattr(seen[0].axes, 'zaxis')
    assert seen[-1].revealed_counts is not None


# --- hue overlays (review T7) -----------------------------------------------

def test_hook_sees_post_multicolor_artists():
    """_apply_multicolor_animation WRAPS line_ani._func (plot.py:5289), so
    the hook must be installed OUTSIDE that wrapper or it observes empty
    collections."""
    ds = _datasets()
    hue = np.linspace(0.0, 1.0, sum(d.shape[0] for d in ds))
    seen = []
    fig, ani = hyp.plot(ds, '-', hue=hue, animate=True, duration=2,
                        frame_rate=4, on_frame=seen.append, show=False)
    _drive(ani, 8)
    assert len(seen) == 8
    ax = _axes_of(fig)
    overlay = [c for c in ax.collections if c.get_label() == '_nolegend_']
    assert overlay and any(len(c._segments3d) for c in overlay)


# --- registry identity (review C7) ------------------------------------------

def test_hook_can_be_attached_after_construction():
    """The defect v1 could not have caught: a fresh list in __new__ is
    invisible to the closure created inside _draw."""
    seen = []
    result = hyp.plot(_datasets(), '-', animate=True, duration=1,
                      frame_rate=2, show=False)
    result.on_frame(seen.append)
    _drive(result[1], 2)
    assert len(seen) == 2


def test_on_frame_returns_self_for_chaining():
    a, b = [], []
    result = hyp.plot(_datasets(), '-', animate=True, duration=1,
                      frame_rate=2, show=False)
    assert result.on_frame(a.append).on_frame(b.append) is result
    _drive(result[1], 2)
    assert len(a) == len(b) == 2


def test_constructor_and_post_construction_callbacks_both_fire():
    first, second = [], []
    result = hyp.plot(_datasets(), '-', animate=True, duration=1,
                      frame_rate=2, on_frame=first.append, show=False)
    result.on_frame(second.append)
    _drive(result[1], 2)
    assert len(first) == len(second) == 2


# --- errors and limits ------------------------------------------------------

def test_hook_exception_is_not_swallowed():
    def boom(ctx):
        raise RuntimeError('hook failed')

    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=2, on_frame=boom, show=False)
    with pytest.raises(RuntimeError, match='hook failed'):
        ani._func(0, *ani._args)


def test_on_frame_rejects_non_callable():
    with pytest.raises(TypeError, match='on_frame must be callable'):
        hyp.plot(_datasets(), '-', animate=True, duration=1, frame_rate=2,
                 on_frame='not callable', show=False)


def test_on_frame_without_animation_raises():
    with pytest.raises(ValueError, match='on_frame requires an animated plot'):
        hyp.plot(_datasets(), '-', on_frame=lambda ctx: None, show=False)


# --- backend parity ---------------------------------------------------------

def test_on_frame_fires_once_per_frame_on_plotly():
    """plotly DOES have a Python per-frame loop -- at BUILD time, inside
    `_add_animation` (plotly_backend.py:2517; frames appended at :2729 spin,
    :2819 morph, :2865 serial, :2975 parallel). No driving is needed: the
    callbacks have all fired by the time plot() returns.
    """
    pytest.importorskip('plotly')
    seen = []
    hyp.set_interactive_backend('plotly')
    try:
        hyp.plot(_datasets(), '-', animate=True, duration=2, frame_rate=4,
                 on_frame=seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert [ctx.frame for ctx in seen] == list(range(8)), (
        'exactly once per frame index, in order, at build time')
    assert all(isinstance(ctx, FrameContext) for ctx in seen)


@pytest.mark.parametrize('style,order', [
    (True, 'parallel'),
    (True, 'serial'),
    ('spin', 'parallel'),
    ('morph', 'serial'),
])
def test_on_frame_context_metadata_parity_across_backends(style, order):
    """THE parity guarantee: same `on_frame`, same per-frame CONTEXT METADATA.

    Deliberately NOT "output parity". Parity holds over the backend-
    INDEPENDENT fields only. `figure`/`axes`/`artists` are backend-native by
    design (matplotlib artists vs. plotly traces), so a callback that mutates
    them is not source-compatible across backends and rendered output is not
    claimed to match. Those fields are excluded here and documented as such
    -- see the next test, and the per-backend retention pair in Step 1b.
    """
    pytest.importorskip('plotly')

    def _portable(ctx):
        return (ctx.frame, ctx.n_frames, ctx.style, ctx.order,
                ctx.current_index,
                None if ctx.current_fraction is None
                else round(ctx.current_fraction, 9),
                ctx.revealed_counts, ctx.segment_index, ctx.segment_kind,
                [d.shape for d in ctx.datasets])

    kwargs = dict(animate=style, order=order, duration=2, frame_rate=4,
                  show=False)
    if style == 'morph':
        # BELOW `_datasets()`'s row count (20), so the cap actually
        # engages: at morph_samples=50 (> 20) it never triggered at all,
        # so `ctx.datasets` came out (20, dims) either way regardless of
        # whether a backend used the morph-SAMPLED clouds or the raw
        # input -- the exact gap that let plotly's `datasets=tuple(data)`
        # (raw input, whole-branch-review Important finding 2) slip past
        # this "parity" check without ever actually comparing sampled vs.
        # raw shapes.
        kwargs['morph_samples'] = 8

    mpl_seen = []
    fig, ani = hyp.plot(_datasets(), '.', on_frame=mpl_seen.append, **kwargs)
    _drive(ani, mpl_seen[0].n_frames if mpl_seen else 8)

    ply_seen = []
    hyp.set_interactive_backend('plotly')
    try:
        hyp.plot(_datasets(), '.', on_frame=ply_seen.append, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')

    # matplotlib may repeat a frame index across a loop/save; plotly may not.
    # Compare the per-index CONTENT, which is what the contract guarantees.
    mpl_by_index = {ctx.frame: _portable(ctx) for ctx in mpl_seen}
    ply_by_index = {ctx.frame: _portable(ctx) for ctx in ply_seen}
    assert sorted(ply_by_index) == sorted(mpl_by_index)
    assert ply_by_index == mpl_by_index


def test_morph_datasets_are_the_sampled_clouds_not_the_raw_input():
    """Important finding 2 (whole-branch review): ctx.datasets for
    animate='morph' must be the morph-SAMPLED (morph_samples-capped)
    clouds on BOTH backends, matching the FrameContext.datasets contract
    ("the arrays the animation actually DRAWS FROM ... not the raw
    input"). plotly used to record `tuple(data)` -- the RAW, uncapped
    input -- while matplotlib already recorded the sampled clouds; the
    parametrized parity test above only pinned morph_samples=50 on
    20-row data (no cap ever engaged), so it could never catch this."""
    pytest.importorskip('plotly')
    clouds = [np.random.default_rng(i).normal(size=(60, 3)) + i * 4.0
             for i in range(3)]

    mpl_seen = []
    fig, ani = hyp.plot(clouds, '.', animate='morph', morph_samples=10,
                        duration=6, frame_rate=2, on_frame=mpl_seen.append,
                        show=False)
    _drive(ani, mpl_seen[0].n_frames if mpl_seen else 12)
    assert mpl_seen, 'expected at least one recorded frame'
    assert all(d.shape[0] == 10 for ctx in mpl_seen for d in ctx.datasets), (
        'matplotlib ctx.datasets must be the morph_samples-capped clouds')

    ply_seen = []
    hyp.set_interactive_backend('plotly')
    try:
        hyp.plot(clouds, '.', animate='morph', morph_samples=10,
                 duration=6, frame_rate=2, on_frame=ply_seen.append,
                 show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert ply_seen, 'expected at least one recorded frame'
    assert all(d.shape[0] == 10 for ctx in ply_seen for d in ctx.datasets), (
        "plotly ctx.datasets for animate='morph' must be the "
        "morph_samples-capped clouds, not the raw 60-row input")


def test_plotly_frame_context_carries_backend_native_objects():
    """Documented, not faked: on plotly `figure` is the go.Figure and
    `artists` are that frame's traces. A caller that touches these is
    writing backend-specific code and the docstring says so."""
    pytest.importorskip('plotly')
    import plotly.graph_objects as go

    seen = []
    hyp.set_interactive_backend('plotly')
    try:
        hyp.plot(_datasets(), '-', animate=True, duration=1, frame_rate=2,
                 on_frame=seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    ctx = seen[-1]
    assert isinstance(ctx.figure, go.Figure)
    assert len(ctx.artists) >= 3
    assert all(hasattr(a, 'x') for a in ctx.artists), 'traces, not artists'


def test_matplotlib_artists_are_shared_across_frame_deliveries():
    """Matplotlib hands out the SAME artist objects every frame -- the
    FuncAnimation updater mutates them in place. Verified against the real
    backend: line identities are unchanged across frames 0/1/2.

    This is why the contract is "set the complete state for this frame":
    a conditional mutation persists into every later frame. The plan
    previously claimed matplotlib was per-frame throughout; it is not.
    """
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=4, on_frame=seen.append, show=False)
    _drive(ani, 3)
    assert len(seen) == 3
    first = tuple(id(a) for a in seen[0].artists)
    assert all(tuple(id(a) for a in ctx.artists) == first for ctx in seen), (
        'matplotlib re-delivers the same artist objects, not copies')


def test_plotly_spin_artists_are_the_static_data_traces():
    """Regression: the spin branch builds no `frame_traces` (its frame payload
    is camera-layout only, plotly_backend.py:2695-2699), so a literal
    `artists=frame_traces` raises NameError there. Spin publishes the traces it
    actually renders -- the figure's static ones -- never an empty tuple."""
    pytest.importorskip('plotly')
    seen = []
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate='spin', duration=1,
                       frame_rate=4, on_frame=seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert len(seen) == 4
    assert all(len(ctx.artists) > 0 for ctx in seen), 'never empty'
    assert all(hasattr(a, 'x') for a in seen[0].artists), 'traces, not artists'
    # shared, not per-frame: every frame publishes the SAME trace objects
    assert all(ctx.artists[0] is seen[0].artists[0] for ctx in seen)
    assert seen[0].artists[0] in tuple(fig.data)


def test_plotly_surface_spin_artists_include_the_per_frame_mesh_updates():
    """Surfaced spin DOES send per-frame data (`surf_data`, the re-shaded
    Mesh3d updates at plotly_backend.py:2711-2735). Those trail the static
    traces, so a caller can reach both."""
    pytest.importorskip('plotly')
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(40, 3))
    seen = []
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot([cloud], animate='spin', surface=True, duration=1,
                       frame_rate=4, on_frame=seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    import plotly.graph_objects as go

    # NOT `hasattr(t, 'x')` -- go.Mesh3d has an .x too (verified), so that
    # predicate matches every trace and the assertion can never fail.
    # Discriminate on TYPE.
    assert any(isinstance(t, go.Mesh3d) for t in seen[0].artists), (
        'the frame\'s re-shaded mesh updates are appended')
    assert isinstance(seen[0].artists[-1], go.Mesh3d)
    # the trailing mesh entries are PER-FRAME: different objects each frame
    assert seen[0].artists[-1] is not seen[1].artists[-1]
    # ...while the LEADING entries are the shared figure traces themselves,
    # which is what makes this the documented mixed case
    assert seen[0].artists[0] is seen[1].artists[0]
    assert seen[0].artists[0] in tuple(fig.data)


def test_plotly_spin_mutation_is_retained_and_is_figure_wide():
    """Spin's documented consequence: because the traces are shared, a
    mutation is figure-wide rather than per-frame. Asserted, not hidden."""
    pytest.importorskip('plotly')

    def rename(ctx):
        if ctx.frame == 1:
            ctx.artists[0].name = 'touched-on-frame-1'

    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate='spin', duration=1,
                       frame_rate=4, on_frame=rename, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert fig.data[0].name == 'touched-on-frame-1', (
        'the mutation lands on the shared figure trace')


@pytest.mark.parametrize('style', [True, 'serial', 'window', 'morph'])
def test_plotly_non_spin_frames_are_isolated_per_frame(style):
    """The MIRROR IMAGE of the spin test above, and the reason the guide
    documents two opposite failure modes rather than one.

    plotly's parallel/serial/window/morph branches each build their own
    `frame_traces`, so a callback that mutates on frame 1 only affects
    frame 1 -- the exact opposite of spin and of matplotlib, where the same
    callback would affect the whole animation. Measured against the real
    backend 2026-07-30 (before this plan changes anything):
    `fig.frames[0].data[0] is not fig.frames[1].data[0]` for all four.

    A caller who writes a conditional mutation therefore gets DIFFERENT
    wrong behaviour per backend, which is why the portable contract is
    "set the complete state every frame" and not "mutations persist".
    """
    pytest.importorskip('plotly')

    def rename(ctx):
        if ctx.frame == 1:
            ctx.artists[0].name = 'touched-on-frame-1'

    kwargs = dict(morph_samples=40) if style == 'morph' else {}
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=style, duration=1,
                       frame_rate=4, on_frame=rename, show=False, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')

    assert fig.frames[1].data[0].name == 'touched-on-frame-1'
    assert fig.frames[0].data[0].name != 'touched-on-frame-1', (
        'frame 0 carries its own payload; the mutation must NOT leak back')
    assert fig.frames[2].data[0].name != 'touched-on-frame-1', (
        'nor forward -- this is what makes these styles per-frame')
    assert fig.frames[0].data[0] is not fig.frames[1].data[0]


@pytest.mark.parametrize('backend,style', [
    ('matplotlib', True),        # revealed_counts from the head windows
    ('matplotlib', 'serial'),    # revealed_counts from serial_reveal_counts
    ('matplotlib', 'morph'),     # artists=[morph_state["artist"]]
    ('plotly', True),            # artists=frame_traces
    ('plotly', 'serial'),        # artists=frame_traces, revealed_counts=_shown
    ('plotly', 'spin'),          # artists=tuple(fig.data[i] ...) -- the odd one
])
def test_frame_context_containers_are_canonical_tuples(backend, style):
    """`artists`, `datasets` and `revealed_counts` are TUPLES on every
    backend and every style -- a public field may not change type
    according to which branch built it.

    This is the regression guard for `FrameContext.__post_init__`. Eleven
    call sites record frame state and each has a different sequence in
    hand; before the normalizer, matplotlib passed lists and plotly's spin
    branch passed a tuple, so `type(ctx.artists)` varied by style.
    """
    if backend == 'plotly':
        pytest.importorskip('plotly')
    seen = []
    kwargs = dict(morph_samples=40) if style == 'morph' else {}
    hyp.set_interactive_backend(backend)
    try:
        result = hyp.plot(_datasets(), '.', animate=style, duration=1,
                          frame_rate=4, on_frame=seen.append, show=False,
                          **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')
    if backend == 'matplotlib':
        _drive(result[1], 3)

    assert seen, 'the hook must have fired'
    for ctx in seen:
        assert type(ctx.artists) is tuple, type(ctx.artists)
        assert type(ctx.datasets) is tuple, type(ctx.datasets)
        assert (ctx.revealed_counts is None
                or type(ctx.revealed_counts) is tuple), ctx.revealed_counts
        # frozen means MEMBERSHIP is fixed; the artists inside stay mutable
        with pytest.raises(AttributeError):
            ctx.artists.append(None)


# --- mutation retention: the per-backend guarantee (Step 1b) ----------------
# Cross-backend OUTPUT parity is deliberately NOT asserted anywhere: artists
# and traces are backend-native, so a mutating callback is not source-
# compatible across backends. What each backend owes the caller is that a
# mutation it was handed is RETAINED in the frame that backend renders.

def test_matplotlib_callback_mutation_is_retained_in_the_rendered_frame():
    """The hook exists to mutate. Setting a title from the callback must
    survive into the frame matplotlib actually renders -- and, because a
    frame index may be re-delivered, re-running the same index must land on
    the same title rather than compounding."""
    captured = {}

    def retitle(ctx):
        ctx.axes.set_title(f'f{ctx.frame}')
        captured['ax'] = ctx.axes

    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=4, on_frame=retitle, show=False)
    _drive(ani, 3)
    assert captured['ax'].get_title() == 'f2', (
        'the mutation made during the last driven frame is still on the axes')

    # idempotence: re-delivering an earlier index reproduces that index's
    # state exactly, which is what makes matplotlib's repeat harmless.
    # (`_func` here is the TEST HARNESS standing in for matplotlib's own
    # renderer, exactly as `_drive` does -- it is not the user-facing reach
    # into private internals this plan removes.)
    ani._func(1, *ani._args)
    assert captured['ax'].get_title() == 'f1'


def test_plotly_callback_mutation_is_retained_in_the_stored_frame():
    """Same guarantee on plotly, and it pins the DISPATCH ORDER: Step 6a puts
    the hook immediately BEFORE `frames.append(go.Frame(**frame_kwargs))`, so
    a trace the callback mutates is captured by the stored frame. Dispatching
    after the append would silently drop every mutation and this test is what
    catches that."""
    pytest.importorskip('plotly')

    def rename(ctx):
        ctx.artists[0].name = f'frame-{ctx.frame}'

    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=True, duration=1,
                       frame_rate=4, on_frame=rename, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert fig.frames[2].data[0].name == 'frame-2'
    assert fig.frames[0].data[0].name == 'frame-0'


def test_return_model_bundle_hands_back_a_raw_funcanimation():
    """Documented limitation: the bundle never constructs a HyperAnimation
    (plot.py:4584-4586, :4612-4614), so .on_frame() is not available there --
    but on_frame= passed to plot() still fires."""
    seen = []
    bundle = hyp.plot(_datasets(), '-', animate=True, duration=1,
                      frame_rate=2, on_frame=seen.append,
                      return_model=True, show=False)
    ani = bundle['animation']
    with pytest.raises(AttributeError, match='on_frame'):
        ani.on_frame(seen.append)
    _drive(ani, 2)
    assert len(seen) == 2
