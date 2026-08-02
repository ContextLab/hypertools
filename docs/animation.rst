.. _animation:

Animating plots
===============

Every animation in HyperTools comes out of one call: ``hypertools.plot``
with ``animate=``. This guide covers what you can vary -- the animation
style, the order data is revealed in, trails, titles, per-dataset styling,
and per-frame callbacks -- and what differs between the matplotlib and
plotly backends.

Style and order are independent
-------------------------------

``animate=`` names a *style*; ``order=`` names the *ordering*. They are
orthogonal, which is new in 1.0.1:

.. code-block:: python

    import hypertools as hyp

    data = hyp.load('weights')

    hyp.plot(data, '-', animate=True)                      # parallel reveal
    hyp.plot(data, '-', animate=True, order='serial')      # one at a time
    hyp.plot(data, '-', animate='spin')                    # rotate, no reveal

``animate='serial'`` remains a permanent alias for
``animate=True, order='serial'``, and ``animate='morph'`` is inherently
serial. ``order='serial'`` has no meaning for ``'spin'`` or ``'window'``;
passing it there warns and is ignored, rather than raising.

Trails
------

``chemtrails=``, ``precog=`` and ``bullettime=`` leave a visible history
behind (or ahead of) the moving head. As of 1.0.1 they compose with a serial
reveal on **both** backends:

.. code-block:: python

    hyp.plot(data, '-', animate=True, order='serial', chemtrails=True)

Titles that change with the animation
-------------------------------------

Pass a **list** of strings as ``title=`` to name each segment of a
serial-style animation. For a morph, the holds are named and the
transitions are left blank automatically:

.. code-block:: python

    hyp.plot([a, b, c], '-', animate='morph',
             title=['first', 'second', 'third'])

Anywhere else a non-string ``title=`` raises ``TypeError``. Use ``names=``
for per-dataset legend entries and ``labels=`` for per-observation
annotations.

Per-dataset styling
-------------------

``color=``, ``linewidth=`` and -- new in 1.0.1 -- ``alpha=`` accept one value
per dataset:

.. code-block:: python

    hyp.plot([a, b, c], '-', animate=True,
             color=['red', 'blue', 'green'], alpha=[1.0, 0.6, 0.3])

Some inputs assign alpha internally (row-MultiIndex frames, nested lists).
Those keep their own values and say so with a warning rather than silently
discarding yours.

Large morphs and ``simplify=``
------------------------------

Morphing clouds larger than about 2000 points is intractable to render.
``simplify=True`` (the default) silently downsamples them so the render
finishes. Pass ``simplify=False`` to get a ``ValueError`` instead, which
restores the guarantee that no real data point is ever dropped:

.. code-block:: python

    hyp.plot(big_clouds, animate='morph')                   # downsampled
    hyp.plot(big_clouds, animate='morph', simplify=False)   # raises
    hyp.plot(big_clouds, animate='morph', morph_samples=500) # you decide

An explicit ``morph_samples=`` always wins, and below the threshold
``simplify`` does nothing at all.

Per-frame callbacks
-------------------

``on_frame=`` runs your function once per frame with a
:class:`~hypertools.FrameContext`. **Passing it to** ``plot()`` **works on
both backends** and is the portable form:

.. code-block:: python

    def label_frame(ctx):
        # ctx.frame and ctx.n_frames are backend-independent
        print(f'frame {ctx.frame} of {ctx.n_frames}')

    hyp.plot(data, '-', animate=True, on_frame=label_frame)

The context carries the frame index and total, the resolved ``style`` and
``order``, the arrays being drawn, the serial-reveal counts, and -- for
morphs -- ``segment_index`` and ``segment_kind``. All of those are the same
on either backend.

What you *do* with the context is usually backend-specific, because
``ctx.figure``, ``ctx.axes`` and ``ctx.artists`` are backend-native. The
matplotlib form:

.. code-block:: python

    # MATPLOTLIB ONLY -- ctx.axes is None on plotly
    def annotate(ctx):
        ctx.axes.set_title(f'frame {ctx.frame} of {ctx.n_frames}')

    fig, ani = hyp.plot(data, '-', animate=True, on_frame=annotate)

and the plotly equivalent, which reaches the frame's traces instead. Its
artists are ``go.Scatter``/``go.Scatter3d`` traces, whose color lives at
``.line.color``; a surfaced ``animate='spin'`` frame's ``go.Mesh3d`` update
has no ``.line`` at all, so this spelling is scoped to the line traces, not
every plotly artist type:

.. code-block:: python

    # PLOTLY ONLY -- ctx.artists are that frame's traces
    def rename(ctx):
        ctx.artists[0].name = f'frame {ctx.frame}'

    hyp.set_interactive_backend('plotly')
    fig = hyp.plot(data, '-', animate=True, on_frame=rename)

.. _animation-artist-lifetime:

Artist lifetime: what ``ctx.artists`` actually hands you
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether ``ctx.artists`` holds fresh objects each frame or the *same*
objects re-delivered depends on the backend and the style:

.. list-table::
   :header-rows: 1

   * - backend / style
     - lifetime
   * - matplotlib, **all** styles
     - shared live artists, mutated in place on every render
   * - plotly ``animate='spin'`` (no surfaces)
     - shared figure traces
   * - plotly ``animate='spin'`` (surfaced)
     - shared traces, then that frame's ``Mesh3d`` updates
   * - plotly parallel / serial / window / morph
     - per-frame trace payloads

Matplotlib never hands you a fresh artist. ``FuncAnimation``'s updater
mutates the same ``Line2D`` and collection objects every frame, so
``ctx.artists[0]`` on frame 1 and on frame 2 are the *same* object in two
different states. Plotly's spin is the same story for a different reason:
it moves only the camera and re-sends no point data, so its frames share
the figure's traces.

**The rule that follows applies to both backends: assign the complete
value you want on every invocation, including the default** -- never
write the attribute on some frames and leave it untouched on others. The
rule is portable; the *reason* is not, and the two failure modes are
opposite.

Where artists are **shared**, anything you set persists until something
overwrites it::

    # MATPLOTLIB ONLY (set_color is a matplotlib Artist method).
    # Shared artists, so this colours the WHOLE animation, not frame 0.
    def broken(ctx):
        if ctx.frame == 0:
            ctx.artists[0].set_color('red')

    def correct(ctx):
        ctx.artists[0].set_color(COLOURS[ctx.frame])   # set it every frame

Where they are **per-frame**, the very same conditional does the opposite
-- it touches an independent payload that only that frame keeps::

    # PLOTLY ONLY -- ctx.artists are that frame's traces, and
    # parallel/serial/window/morph frames are independent, so this
    # colours ONLY frame 0.
    def also_broken(ctx):
        if ctx.frame == 0:
            ctx.artists[0].line.color = 'red'

    def also_correct(ctx):
        ctx.artists[0].line.color = COLOURS[ctx.frame]

Writing a callback as though each frame had its own artists is the common
mistake, and writing one as though they were shared is the mirror image of
it. Under matplotlib and under plotly spin there is only ever one object,
so a conditional mutation looks like it "sticks" -- because it does. Under
plotly's other styles it silently does not.

This is also why *"a mutation is retained in the rendered frame"* does not
mean artists are isolated per frame. It means the backend renders what you
set; where artists are shared it renders it for every later frame too. A
surfaced spin is the mixed case: its ``Mesh3d`` updates trail the shared
traces in ``ctx.artists`` and those trailing entries *are* per-frame.

Highlighting exactly one frame is a perfectly good thing to want, and none
of this forbids it. Put the condition in the **value**, not around the
call, so the attribute is still assigned on every frame::

    HIGHLIGHT, DEFAULT = 'red', 'steelblue'

    def highlight_one_frame(ctx):                   # correct on both backends
        colour = HIGHLIGHT if ctx.frame == TARGET else DEFAULT
        ctx.artists[0].set_color(colour)            # matplotlib spelling

Assign on every invocation and none of this can bite you.

.. _animation-post-construction:

Registering after construction is matplotlib-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On matplotlib you can attach a callback to an animation you already have::

    anim = hyp.plot(data, '-', animate=True)   # a HyperAnimation
    anim.on_frame(annotate)                     # fires on subsequent draws

**This is not available on plotly, and cannot be.** An animated matplotlib
plot returns a :class:`~hypertools.HyperAnimation`, whose frames are drawn
lazily at render time -- so there is still a window in which to register.
An animated plotly plot returns a plain ``plotly.graph_objects.Figure``:
its frames are *already built* by the time ``plot()`` returns, so there is
no later frame to call back into, and the returned object has no
``.on_frame()`` method.

If you are writing backend-portable code, **pass the callback to**
``plot()``. That is the form that works everywhere:

.. code-block:: python

    hyp.plot(data, '-', animate=True, on_frame=my_callback)   # both backends

    anim = hyp.plot(data, '-', animate=True)                  # matplotlib only
    anim.on_frame(my_callback)

.. _animation-callback-contract:

The callback contract
~~~~~~~~~~~~~~~~~~~~~

**Callbacks must be deterministic and idempotent for a given frame
context. They must not depend on call count, call order, wall-clock time,
or accumulated external state.**

Mutating what the context hands you is the *point* of the hook and is fully
supported -- the example above sets a title on every frame. What is
unsupported is **accumulation**::

    def ok(ctx):                     # idempotent: same frame, same result
        label.set_text(TITLES[ctx.frame])

    def broken(ctx):                 # accumulates: a repeated frame drifts
        ctx.artists[0].set_alpha(ctx.artists[0].get_alpha() * 0.9)

If you need a running quantity, precompute it once and index it by
``ctx.frame``::

    ACC = compute_running_accuracy(...)     # once, before plotting

    def show_accuracy(ctx):
        label.set_text(f'{ACC[ctx.frame]:.0f}%')

Backend scheduling
~~~~~~~~~~~~~~~~~~

The two backends call back on different schedules, and that is why the
contract exists:

.. list-table::
   :header-rows: 1

   * - backend
     - when it calls
     - how often per frame index
   * - matplotlib
     - at render time
     - **one or more times** -- a looping animation or a save replays frames
   * - plotly
     - at build time, before ``plot()`` returns
     - exactly once

Both backends deliver the same *context metadata* for a given frame index.
They do **not** produce interchangeable rendered output from a mutating
callback: ``ctx.figure``, ``ctx.axes`` and ``ctx.artists`` are
backend-native (on plotly ``ctx.axes`` is ``None`` and ``ctx.artists`` are
that frame's traces), so a callback that touches them is backend-specific
code. Each backend does guarantee that a mutation you make is retained in
the frame it renders.

Forecasting during an animation
--------------------------------

``predict=`` works with the time-progressing animation styles
(``animate=True``, ``'parallel'``, ``'serial'``, ``'window'``) on **both**
backends. The forecast is recomputed from the history revealed so far and
re-anchored on the last revealed observation, so the forecast trace grows with
the animation instead of standing still:

.. code-block:: python

    fig, ani = hyp.plot(data, '-', predict='Kalman', t=10,
                        animate=True, duration=8, frame_rate=20)

``t`` is measured in **raw observations of the analyzed data** -- not in
animation frames, and not in drawn vertices. ``t=1`` forecasts the next
observation. Because an animation is paced on a resampled frame grid (see
``duration``/``frame_rate``), an animated forecast joins the drawn trajectory
to within one raw observation rather than exactly.

Everything is computed before the first frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Animating static data means every observation is known before drawing starts,
so every forecast the animation will *ever* draw is knowable up front. Two
things follow, and both are contracts rather than implementation details:

- The whole fan is folded into the plot's centre/scale statistics, so it lands
  inside the cube **by construction**. Nothing is clipped or clamped.
- Each frame is a table lookup, so ``ani.save()`` and ``to_jshtml()`` replay
  identically no matter what order matplotlib asks for frames in.

Fits are memoized per (dataset, revealed-count), so a 900-frame animation of a
60-row dataset costs at most 59 fits rather than 900.

Keeping earlier forecasts on screen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``forecast_trail=`` is the forecast analogue of ``chemtrails=``: earlier
forecasts stay visible and fade, so a viewer can see how the prediction
*changed* as history accumulated.

.. code-block:: python

    fig, ani = hyp.plot(data, '-', predict='Kalman', t=10,
                        animate=True, forecast_trail=True,
                        duration=8, frame_rate=20)

``True`` retains 16 past forecasts; an int sets the cap. Without ``predict=``
it raises ``ValueError`` rather than silently doing nothing. Like the live
forecast, the fan is recomputed from the frame index rather than accumulated
in a buffer, so it depends only on which frame is being drawn -- a saved
animation and an interactively-played one are identical, and asking for
frames out of order (which ``save()`` does) gives the same picture.

How a forecast is styled
~~~~~~~~~~~~~~~~~~~~~~~~

A forecast is the *same series projected forward*, so it inherits the identity
of the observed trace it continues -- the same **colour**, **linestyle** and
**linewidth** -- and differs only in transparency:

.. code-block:: text

    forecast_alpha = observed_alpha * 0.5

An observed line with no ``alpha=`` set is matplotlib's *opaque*, i.e. 1.0, so
the default forecast alpha is 0.5. Per-dataset styling carries through
dataset by dataset: ``alpha=[1.0, 0.4]`` gives forecasts at ``[0.5, 0.2]``, and
a dotted dataset gets a dotted forecast. Both backends apply the identical
rule (on plotly, colour/width/dash with the alpha baked into the ``rgba(...)``
line colour and echoed in ``meta['hyp_forecast_alpha']``).

.. versionchanged:: 1.0.1
   Before 1.0.1 every forecast was drawn ``linestyle='--'`` at a hard-coded
   ``alpha=0.6``, whatever its data looked like -- so a forecast of a dotted,
   hairline or already-translucent dataset read as a *different* series rather
   than as its continuation.

``forecast_trail=`` fades from **that** dataset's live forecast alpha, down to
a floor proportional to it -- so a retained forecast is never more opaque than
the live forecast it decays from, however faint the dataset.

``hue=`` and ``cluster=``: static only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hue=`` and ``cluster=`` regroup the drawn traces by category rather than by
dataset, so there is no longer one trace per dataset. A **static** plot copes:
it draws one forecast per dataset, anchored at that dataset's last
observation and styled like the trace holding it.

An **animated** plot does not, and warns rather than failing quietly:

.. code-block:: python

    # draws forecasts
    hyp.plot(data, '-', predict='Kalman', t=10, hue=categories)

    # warns; no per-frame forecast is drawn
    hyp.plot(data, '-', predict='Kalman', t=10, hue=categories, animate=True)

The reason is structural rather than an oversight. The per-frame schedule
maps frame-grid rows onto each **dataset's** raw observations, so it needs a
per-dataset reveal to schedule against; regrouping leaves only per-run traces
revealing themselves, and a run is not a dataset. Plot statically to see
forecasts alongside ``hue=``/``cluster=``.

Identifying forecast artists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast artists carry an explicit tag, so a per-frame callback can find them
without guessing from linestyle (which, since a forecast now inherits its
trace's linestyle, no longer distinguishes them at all) --
``artist._hyp_forecast_role`` on matplotlib
(``'static'``, ``'live'`` or ``'trail'``) and ``trace.meta['hyp_forecast_role']``
on plotly. Trail artists additionally carry ``_hyp_forecast_age``.

Note that forecast artists are deliberately **not** in
``FrameContext.artists``: that sequence has cardinality assumptions (one entry
per drawn dataset) that a variable number of forecast overlays would violate.

What ``animate='morph'`` does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``animate='morph'`` (including the per-dataset morph list form) raises
``NotImplementedError`` with ``predict=``. A morph interpolates between point
*clouds*, so there is no time axis to forecast along -- this is a statement
about what a morph means, not a gap to be filled later.

Migrating from ``_func``/``_args``
----------------------------------

Before 1.0.1 the only way to run code per frame was to monkeypatch
matplotlib's private ``FuncAnimation._func`` and read ``_args``. That
reached into matplotlib internals, worked on one backend only, and broke
whenever the private signature changed. Replace it:

.. code-block:: python

    # before -- private, matplotlib-only
    _orig = ani._func

    def _wrapped(num, *args):
        out = _orig(num, *args)
        label.set_text(TITLES[num])
        return out

    ani._func = _wrapped

    # after -- public, both backends
    fig, ani = hyp.plot(data, '-', animate=True,
                        on_frame=lambda ctx: label.set_text(TITLES[ctx.frame]))

If you were re-deriving the serial reveal counts by hand from ``_args``,
use ``ctx.revealed_counts``; if you were computing which morph segment a
frame belonged to, use ``ctx.segment_index`` and ``ctx.segment_kind``
rather than thresholding ``ctx.current_fraction`` -- a hold and a
transition are not separable by fraction alone.
