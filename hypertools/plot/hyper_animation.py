"""HyperAnimation: the object an animated ``hyp.plot(...)`` returns.

A thin wrapper around the matplotlib ``Figure`` and ``FuncAnimation`` produced
by an animated plot. It exists so that, in a notebook, the animation plays
inline automatically (via ``_repr_html_``), and so the caller can export it
directly::

    anim = hyp.plot(data, animate='spin')   # auto-plays inline in Jupyter
    anim.to_html5_video()                    # HTML5 <video> string
    anim.save('trajectory.gif')              # write a file
    fig = anim.figure                        # the matplotlib Figure

Before QC 2026-07 an animated ``plot`` returned a bare ``(figure, animation)``
tuple, so the natural ``anim = hyp.plot(...); anim.to_html5_video()`` failed
(``'tuple'`` has no ``to_html5_video``). ``HyperAnimation`` subclasses ``tuple``
and IS ``(figure, animation)``, so every existing pattern keeps working --
``fig, anim = hyp.plot(...)``, ``out[0]``, ``isinstance(out, tuple)``,
``len(out) == 2`` -- while the export/display methods are now available on the
returned object itself.
"""


def mark_draw_started(animation):
    """Silence matplotlib's 'Animation was deleted without rendering anything'
    UserWarning for ``animation`` (release-1.0 audit, X4-warnings-012).

    The warning fires from ``Animation.__del__`` when the private
    ``_draw_was_started`` flag is still False; matplotlib only reads the flag
    there (and itself force-sets it True in ``Animation.save``), so marking it
    True has no effect on rendering -- saving/display still work exactly as
    before. Used by ``HyperAnimation.__del__`` when a wrapper is discarded,
    and by ``plot.py`` when handing the RAW ``FuncAnimation`` back in a
    ``return_model=True`` bundle (that path never constructs a
    ``HyperAnimation``, so without this the animation -- kept in a reference
    cycle by its own canvas callbacks -- warned at the next cyclic-gc pass,
    misattributed to whatever code ran later)."""
    try:
        if getattr(animation, '_draw_was_started', None) is False:
            animation._draw_was_started = True
    except Exception:
        # never let cleanup raise during interpreter shutdown/GC
        pass


class HyperAnimation(tuple):
    """A ``(figure, animation)`` tuple with animation export/display helpers.

    Indexing/unpacking behave exactly like the legacy 2-tuple
    (``fig, anim = result``; ``result[0]`` is the figure, ``result[1]`` the
    animation), so this is a drop-in replacement for the old return value.

    ``.on_frame(callback)`` registers a per-frame callback (matplotlib
    only -- see ``hyp.plot``'s ``on_frame=`` docstring for the
    backend-portable form and the ``FrameContext`` it receives).
    """

    def __new__(cls, figure, animation, frame_hooks=None):
        self = super().__new__(cls, (figure, animation))
        # ADOPT the registry plot() already threaded into the backend -- do
        # NOT create one here. The per-frame updater closure was built inside
        # `_draw` long before this wrapper existed, so a list created here
        # would be a fresh, unreferenced object and `on_frame()` could never
        # fire (plan 1.1 Task 7, review C7).
        self._frame_hooks = frame_hooks
        # Mark the inner animation as draw-started NOW, not only in
        # __del__. When a wrapper dies inside a reference cycle (a test's
        # captured traceback, a figure whose callbacks point back at the
        # animation) the cyclic collector runs the two finalizers in
        # arbitrary order, and if ``Animation.__del__`` goes first it warns
        # "deleted without rendering" before the wrapper can silence it --
        # seen on 4 of 12 CI jobs on 2026-09-04, never locally. matplotlib
        # only reads the flag in that ``__del__`` (see ``mark_draw_started``),
        # so setting it early changes nothing about rendering or saving.
        mark_draw_started(animation)
        return self

    @property
    def figure(self):
        """The matplotlib ``Figure`` the animation draws on."""
        return self[0]

    @property
    def animation(self):
        """The underlying ``matplotlib.animation.Animation`` (usually a
        ``FuncAnimation``). Keeping a reference to it keeps the animation
        alive."""
        return self[1]

    @property
    def n_frames(self):
        """How many frames this animation draws.

        `hyp.plot` always hands `FuncAnimation` an int frame count --
        `max(1, round(frame_rate * duration))` for parallel/serial/spin, and
        `sum(segment_frame_counts(...))` for a morph -- so this is exact
        rather than an estimate. Reading it is the supported alternative to
        matplotlib's private `_save_count`.
        """
        return int(self[1]._save_count)

    @property
    def n_segments(self):
        """Hold/transition segments for ``animate='morph'``; ``None``
        otherwise.

        `n` clouds give ``2n - 1`` segments: `n` holds interleaved with
        `n - 1` transitions, beginning and ending on a hold. There is no
        closing transition back to the first cloud unless you ask for one:
        ``hyp.plot(..., animate='morph', loop=True)`` adds it (GH #285),
        giving ``2(n + 1) - 1`` segments and reusing the first cloud's own
        sampled points so the loop point does not jump. Measured against
        ``morph.segment_frame_counts``: 2 clouds -> 3, 3 -> 5, 5 -> 9; with
        ``loop=True``, 5 clouds -> 11.
        """
        return getattr(self[1], '_hyp_morph_segments', None)

    def draw_frame(self, frame):
        """Render frame `frame`, and return `self` so calls chain.

        The supported way to drive an animation from a test or a script
        without reaching into `FuncAnimation._func`/`._args`. Frames are
        idempotent and order-independent by contract (see `FrameContext`),
        so any index may be drawn at any time.
        """
        if not 0 <= frame < self.n_frames:
            raise IndexError(
                f'frame {frame} is out of range; this animation has '
                f'{self.n_frames} frames, so valid indices are 0 and '
                f'{self.n_frames - 1}')
        self[1]._func(frame, *self[1]._args)
        return self

    def drawn_extent(self, frames=None, threshold=5):
        """The union bounding box of everything this animation DRAWS,
        measured from rendered pixels, in FIGURE fractions (GH #285).

        ``frames=None`` (the default) samples 12 frames evenly across the
        animation -- the drawn extent of a 3-D plot changes with the camera
        angle, so one frame is not representative. Pass an int for a
        different sample size, or an iterable of frame indices to measure
        exactly those. Returns a ``matplotlib.transforms.Bbox`` with y
        increasing UPWARD (``bbox.p0`` is the bottom-left corner).

        Costs one full canvas render per sampled frame, and leaves the
        figure showing the last frame it measured. See
        :func:`hypertools.plot.animate.drawn_extent` for the full contract.
        """
        from .animate import drawn_extent as _drawn_extent
        return _drawn_extent(self, frames=frames, threshold=threshold)

    def on_frame(self, callback):
        """Register `callback` to run after every drawn frame.

        The callback receives a
        :class:`~hypertools.plot.animation_context.FrameContext`. Returns
        `self`, so calls chain. Exceptions from a callback propagate.

        Not available on the ``return_model=True`` bundle, which hands back
        the raw ``FuncAnimation``; pass ``on_frame=`` to ``plot()`` instead
        on that path.
        """
        if self._frame_hooks is None:
            raise RuntimeError(
                "this HyperAnimation carries no frame-hook registry (it was "
                "constructed directly rather than by hyp.plot); pass "
                "on_frame= to hyp.plot instead.")
        self._frame_hooks.add(callback)
        return self

    # --- export / display --------------------------------------------------

    def to_html5_video(self, *args, **kwargs):
        """Return an HTML5 ``<video>`` string for the animation (needs FFmpeg;
        delegates to ``matplotlib.animation.Animation.to_html5_video``)."""
        return self.animation.to_html5_video(*args, **kwargs)

    def to_jshtml(self, *args, **kwargs):
        """Return an interactive JavaScript HTML animation (no FFmpeg needed;
        delegates to ``matplotlib.animation.Animation.to_jshtml``)."""
        return self.animation.to_jshtml(*args, **kwargs)

    def save(self, filename, *args, **kwargs):
        """Save the animation to a file. The writer is chosen by file extension
        -- .gif and .png/.apng (animated PNG) via Pillow, .svg as a
        frame-capped vector animation, .mp4/.mov/.avi/.m4v/.mkv via ffmpeg
        (only the video formats need ffmpeg) -- matching what
        ``hyp.plot(..., save_path=...)`` supports; any other extension raises
        ``ValueError`` naming the supported formats. ``filename`` may be a
        str or any path-like (e.g. ``pathlib.Path``). ``fps`` overrides the
        animation's own frame rate and ``dpi`` the figure's resolution (as
        in ``matplotlib.animation.Animation.save``); any other keyword
        raises ``TypeError`` rather than being ignored. Passing an explicit
        ``writer`` (or positional args) delegates straight to
        ``matplotlib.animation.Animation.save`` instead, with every keyword.

        QC 2026-07: ``.save('x.svg')`` / ``.save('x.png')`` used to crash (raw
        ``Animation.save`` tried to pipe h264 into an svg/png), even though the
        same extensions work via ``save_path=``.
        """
        if args or 'writer' in kwargs:
            return self.animation.save(filename, *args, **kwargs)
        from .animate import _save_animation
        fps = kwargs.pop('fps', None) or self._fps()
        dpi = kwargs.pop('dpi', None)
        if kwargs:
            # Silently dropping a keyword is how `save(path, dpi=75)` wrote
            # a 10 MB GIF at the figure's dpi and nobody noticed (2026-09-03).
            raise TypeError(
                f"HyperAnimation.save() got unexpected keyword argument(s) "
                f"{sorted(kwargs)}; it takes fps= and dpi=, or pass writer= "
                f"to delegate to matplotlib's Animation.save with any keyword")
        return _save_animation(self.animation, str(filename), fps, dpi=dpi)

    def _fps(self):
        """Frames per second from the animation's frame interval (default 30)."""
        interval = getattr(self.animation, '_interval', None)
        if interval:
            return max(1, round(1000.0 / interval))
        return 30

    def _repr_html_(self):
        """Rich display in Jupyter/Colab: play the animation inline. Prefer an
        HTML5 video (needs FFmpeg); fall back to the JS-HTML animation, which
        needs no external tools."""
        try:
            return self.animation.to_html5_video()
        except Exception:
            try:
                return self.animation.to_jshtml()
            except Exception:
                return None

    def __repr__(self):
        return (f"HyperAnimation(figure={self.figure!r}, "
                f"animation={type(self.animation).__name__})")

    def __del__(self):
        """Silence matplotlib's 'Animation was deleted without rendering
        anything' UserWarning when a HyperAnimation is discarded/rebound
        without ever being saved or displayed (release-1.0 audit,
        X4-warnings-012: a common exploratory pattern -- ``hyp.plot(...,
        animate=True)`` in a loop, or an unbound call -- scolded users at
        garbage collection for hypertools' own object lifecycle). The
        warning fires from ``Animation.__del__`` when its private
        ``_draw_was_started`` flag is still False; marking the flag here,
        as this wrapper is collected, keeps matplotlib quiet without
        affecting rendering (nothing is drawn or closed -- saving/display
        still work exactly as before if the inner animation object
        outlives the wrapper). Delegates to ``mark_draw_started`` (also
        used by the ``return_model=True`` bundle path in ``plot.py``)."""
        try:
            mark_draw_started(self[1])
        except Exception:
            # never let cleanup raise during interpreter shutdown/GC
            pass
