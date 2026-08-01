#!/usr/bin/env python
"""The per-frame context handed to a `hyp.plot(..., on_frame=...)` callback,
and the single shared registry those callbacks live in.

Before this existed, callers reached into matplotlib's private
`FuncAnimation._func`/`._args` to run code per frame, and re-derived
hypertools' own serial-reveal schedule by hand (4 of the 5 animated gallery
examples did exactly that). `FrameContext` publishes the state those callers
were reconstructing.

`FrameHooks` exists because of an ordering problem: the per-frame updater
closure is created inside `matplotlib_backend._draw`, long before `plot()`
wraps the result in a `HyperAnimation`. A callback list created in
`HyperAnimation.__new__` would therefore be a fresh, unreferenced object that
the closure never sees. `plot()` creates ONE `FrameHooks`, threads it into
`_draw`, and `HyperAnimation` ADOPTS it -- so `anim.on_frame(cb)` after
construction reaches the same list the dispatcher reads.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class FrameContext:
    """State of one drawn animation frame.

    **Backend note.**
    The same `on_frame` runs on both backends and sees the same values in
    every field below EXCEPT `figure`, `axes` and `artists`, which are
    backend-native: matplotlib `Figure`/`Axes`/artists, or the `go.Figure`
    and that frame's traces. The two backends also call back on different
    SCHEDULES -- matplotlib at render time (so a frame index may recur across
    a loop or a save), plotly exactly once per index at build time -- so
    callbacks must be deterministic and idempotent for a given frame
    context. They must not depend on call count, call order, wall-clock
    time, or accumulated external state.

    Mutating artists is expected and supported -- that is what the hook is
    for. What is unsupported is ACCUMULATION: ``label.set_text(title(ctx))``
    is fine, ``count += 1`` or ``alpha *= 0.9`` is not, because matplotlib
    may deliver the same frame index more than once. If you need a running
    quantity, precompute it once and index it by ``ctx.frame``.

    Attributes
    ----------
    frame : int
        Index of the frame just drawn, counting from 0.
    n_frames : int
        Total frames in the animation. For most styles this is
        ``round(duration * frame_rate)``; for ``animate='morph'`` it is
        ``sum(segment_frame_counts(...))``, which may differ by rounding.
    figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure being animated. BACKEND-NATIVE.
    axes : matplotlib.axes.Axes or None
        The axes the trajectories are drawn on -- an ``Axes3D`` for 3-D
        plots, a plain ``Axes`` for 2-D ones (which have no ``zaxis``).
        BACKEND-NATIVE: ``None`` on plotly, which has no axes object (its
        equivalent state lives on the figure's ``layout``).
    artists : tuple
        The drawn artists, in dataset order: the head artists first, then
        any trail artists, matching the backend's own bookkeeping.
        The CONTAINER is always a tuple -- see "Notes" below --
        while the artists inside it are the backend's own live objects and
        are meant to be mutated.
        BACKEND-NATIVE: on plotly these are that frame's ``go.Scatter``/
        ``go.Scatter3d`` traces, in the same order.

        ARTIST LIFETIME -- read this before writing a callback. Whether
        ``artists`` holds fresh objects per frame or the same objects
        re-delivered depends on the backend and style:

        ==================================  ===========================
        backend / style                     lifetime
        ==================================  ===========================
        matplotlib, ALL styles              shared live artists,
                                            mutated in place each render
        plotly spin (no surfaces)           shared figure traces
        plotly spin (surfaced)              shared traces, then that
                                            frame's Mesh3d updates
        plotly parallel/serial/window/      per-frame trace payloads
        morph
        ==================================  ===========================

        Matplotlib never hands you a fresh artist: ``FuncAnimation``'s
        updater mutates the same ``Line2D``/collection objects every
        frame, so ``ctx.artists[0]`` on frame 1 and on frame 2 are the
        SAME object in different states.

        THE PORTABLE RULE, on both backends: ASSIGN the complete desired
        value on EVERY invocation, including the default. What breaks is
        not a per-frame DECISION, it is a per-frame ASSIGNMENT -- writing
        the attribute on some frames and leaving it untouched on others.
        The rule is the same on both backends but the reason is NOT, and
        the failure modes are opposite:

        * Where artists are SHARED (matplotlib all styles, plotly spin),
          anything you set persists until something overwrites it, so
          ``if ctx.frame == 0: artist.set_color('red')`` colours the
          ENTIRE animation rather than frame 0.
        * Where they are PER-FRAME (plotly parallel/serial/window/morph),
          the same callback colours ONLY frame 0 -- each frame carries an
          independent trace payload that the callback mutates before it
          is stored. Measured 2026-07-30: ``fig.frames[0].data[0] is not
          fig.frames[1].data[0]`` for every one of those four styles.

        So a skipped assignment does not merely misbehave -- it misbehaves
        DIFFERENTLY per backend. Highlighting exactly one frame is
        perfectly legitimate; just put the condition in the VALUE, not
        around the call::

            # correct everywhere -- assigns on every frame
            artist.set_color('red' if ctx.frame == target else DEFAULT)

            # correct everywhere -- also assigns on every frame
            artist.set_color(COLOURS[ctx.frame])

            # BROKEN -- assigns on one frame, leaves the rest to chance
            if ctx.frame == target:
                artist.set_color('red')

        Note that "a mutation is retained in the rendered frame" does NOT
        mean artists are isolated per frame. It means the backend renders
        what you set; where artists are shared, it renders it for every
        subsequent frame too.
    datasets : tuple of numpy.ndarray
        The arrays the animation actually DRAWS FROM, in dataset order --
        not the raw input. For a line format `plot()` pre-interpolates every
        animated dataset onto the frame grid, so these may be denser or
        coarser than what you passed in; `revealed_counts[i]` indexes into
        ``datasets[i]``.
    style : bool or str
        The resolved backend animate mode (``True``/``'serial'``/``'spin'``/
        ``'window'``/``'morph'``) -- i.e. after ``order=`` has been folded in.
    order : {'parallel', 'serial'}
        The resolved ordering.
    current_index : int or None
        For serial-style animations, the index of the dataset currently
        being revealed. For ``animate='morph'`` it is the dataset the
        current segment belongs to (``segment_index // 2``: the shape being
        held, or the SOURCE of the transition). ``None`` for parallel
        animations, where every dataset advances together.
    current_fraction : float or None
        Progress through the current dataset (serial) or the current
        SEGMENT (morph), in [0, 1]. ``None`` when `current_index` is
        ``None``. **This does not distinguish a morph hold from a morph
        transition** -- both sweep 0 -> 1 over their own segment. Use
        `segment_kind` for that.
    revealed_counts : tuple of int or None
        Number of rows of each dataset currently drawn. ``None`` for
        parallel and morph animations -- ``None`` is preserved as
        ``None``, never normalized to an empty tuple.
    segment_index : int or None
        For ``animate='morph'``, the index into the hold/morph schedule
        (``hypertools.plot.morph.frame_to_segment``). ``None`` otherwise.
    segment_kind : {'hold', 'transition'} or None
        ``'hold'`` for even `segment_index` (a fully-formed cloud is being
        held) and ``'transition'`` for odd (one cloud is easing into the
        next) -- the parity rule `morph.morph_positions` implements.
        ``None`` for non-morph animations.

    Notes
    -----
    ``artists``, ``datasets`` and ``revealed_counts`` are always TUPLES
    (``revealed_counts`` is ``None`` or a tuple). This is a public
    guarantee, not an accident of whichever branch built the frame.

    Eleven separate call sites record frame state -- seven matplotlib
    updaters and four plotly frame-build branches -- and each has a
    different sequence in hand: ``list(lines) + [...]``, a list
    comprehension, ``tuple(fig.data[i] for i in trace_indices)``. Left
    alone, ``type(ctx.artists)`` would vary by backend and style, which is
    not something a public field may do.

    All eleven funnel through a SINGLE internal construction site, so
    normalizing in `__post_init__` covers every one of them, and covers any
    branch added later without that branch having to know. This is why the
    coercion lives here and not at the recorders.

    Tuples rather than lists because the dataclass is ``frozen=True``: a
    list would make that promise half-true, letting a caller
    ``ctx.artists.append(...)`` or ``ctx.revealed_counts.sort()`` and
    corrupt the context. The CONTAINED artists stay mutable on purpose --
    mutating them is what the hook is for. What is fixed is MEMBERSHIP.
    """

    frame: int
    n_frames: int
    figure: Any
    axes: Any
    artists: Tuple[Any, ...] = ()
    datasets: Tuple[Any, ...] = ()
    style: Any = None
    order: str = 'parallel'
    current_index: Optional[int] = None
    current_fraction: Optional[float] = None
    revealed_counts: Optional[Tuple[int, ...]] = None
    segment_index: Optional[int] = None
    segment_kind: Optional[str] = None

    def __post_init__(self):
        """Canonicalize the container types. `object.__setattr__` is the
        documented way to assign inside a frozen dataclass."""
        object.__setattr__(self, 'artists', tuple(self.artists))
        object.__setattr__(self, 'datasets', tuple(self.datasets))
        if self.revealed_counts is not None:
            object.__setattr__(self, 'revealed_counts',
                               tuple(self.revealed_counts))


class FrameHooks:
    """The ONE mutable callback registry for an animated plot.

    Created in `plot()` (so it exists before the backend builds its updater
    closures), threaded into `matplotlib_backend._draw`, and adopted -- never
    re-created -- by `HyperAnimation.__new__`.

    Backend updaters call `record(...)` with whatever they know about the
    frame; they never invoke callbacks. On matplotlib, `plot()` installs
    `dispatch` as the OUTERMOST wrapper of `line_ani._func`, after any other
    wrapping (notably `_apply_multicolor_animation`'s), so callbacks always
    see final artists. On plotly, `_add_animation` records and dispatches
    inside its own frame-building loop (one call per
    `frames.append(go.Frame(...))` site) -- the same registry, the same
    `FrameContext` fields, a different schedule; see `FrameContext`'s
    backend note.
    """

    __slots__ = ('callbacks', 'state')

    def __init__(self, callbacks=None):
        self.callbacks = list(callbacks or [])
        self.state = {}

    def add(self, callback):
        """Register `callback`, raising `TypeError` if it is not callable
        (the same message `plot()`'s own `on_frame=` validation raises, so
        the error looks identical whether it's caught at construction time
        or via `HyperAnimation.on_frame`). Returns `self`, so callers that
        reach this directly can chain `.add(...).add(...)` too."""
        if not callable(callback):
            raise TypeError(
                f"on_frame must be callable; got {type(callback).__name__}.")
        self.callbacks.append(callback)
        return self

    def record(self, **state):
        """Store this frame's state. Cheap and unconditional: a no-callback
        animation pays one dict assignment per frame."""
        self.state = state

    def dispatch(self, figure, axes):
        """Build a FrameContext from the recorded state and run every
        callback. Exceptions propagate -- a broken hook must be visible, not
        swallowed into a silently-wrong animation."""
        if not self.callbacks or not self.state:
            return
        ctx = FrameContext(figure=figure, axes=axes, **self.state)
        for callback in self.callbacks:
            callback(ctx)
