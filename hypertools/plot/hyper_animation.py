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


class HyperAnimation(tuple):
    """A ``(figure, animation)`` tuple with animation export/display helpers.

    Indexing/unpacking behave exactly like the legacy 2-tuple
    (``fig, anim = result``; ``result[0]`` is the figure, ``result[1]`` the
    animation), so this is a drop-in replacement for the old return value.
    """

    def __new__(cls, figure, animation):
        self = super().__new__(cls, (figure, animation))
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

    # --- export / display --------------------------------------------------

    def to_html5_video(self, *args, **kwargs):
        """Return an HTML5 ``<video>`` string for the animation (needs FFmpeg;
        delegates to ``matplotlib.animation.Animation.to_html5_video``)."""
        return self.animation.to_html5_video(*args, **kwargs)

    def to_jshtml(self, *args, **kwargs):
        """Return an interactive JavaScript HTML animation (no FFmpeg needed;
        delegates to ``matplotlib.animation.Animation.to_jshtml``)."""
        return self.animation.to_jshtml(*args, **kwargs)

    def save(self, *args, **kwargs):
        """Save the animation to a file (gif/mp4/...); delegates to
        ``matplotlib.animation.Animation.save``."""
        return self.animation.save(*args, **kwargs)

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
