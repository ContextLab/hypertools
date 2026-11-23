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
orthogonal, which is new in 1.1.0:

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
behind (or ahead of) the moving head. As of 1.1.0 they compose with a serial
reveal on **both** backends:

.. code-block:: python

    hyp.plot(data, '-', animate=True, order='serial', chemtrails=True)

A trail is drawn in its dataset's own format and color: a line's trail is the
faded line, a line-and-marker format's (``'-o'``, ``'o-'``, ``'.-'``,
``'--s'``) the faded line with faded markers on the observations, and a
markers-only format's its faded markers.

.. versionadded:: 1.2.1

Four arguments style the reveal:

.. code-block:: python

    hyp.plot(data, '-', animate=True, chemtrails=True, linewidth=1,
             trail_alpha=0.5,       # trails at 0.5 x alpha (default 0.3)
             tail_linewidth=3,      # the recent window, drawn thicker
             head_marker='o',       # a marker on the last revealed point
             hold=2)                # hold the last frame for 2 seconds

``trail_alpha=`` is the opacity of the trails as a fraction of each dataset's
own ``alpha=`` (both backends). ``tail_linewidth=`` is the line width of the
tail, the recent window that ``tail_duration=`` sizes, while the trail keeps
``linewidth=``. ``head_marker=`` draws a marker at the reveal head only (a
matplotlib marker, ``True`` for ``'o'``, or a dict such as ``{'marker': 'D',
'size': 10, 'color': 'k'}``); its color and size follow the dataset unless
the dict sets them. ``hold=`` appends ``round(hold * frame_rate)`` frames
that draw the last frame again, so the pause is in a saved ``.mp4`` or
``.gif``; per-frame titles, captions and ``on_frame=`` callbacks see the last
frame's context during the hold. An ``.mp4`` stores every held frame. A
``.gif`` plays for the same time, but its writer (Pillow) stores a run of
identical frames as one frame with a longer duration, so it has fewer stored
frames than ``n_frames``. ``tail_linewidth=``, ``head_marker=`` and ``hold=``
are matplotlib only: an animated plot under ``backend='plotly'`` raises
``NotImplementedError``. On a static plot (either backend)
``tail_linewidth=``, ``head_marker=`` and ``trail_alpha=`` are ignored with a
warning, and ``hold=`` raises ``ValueError``. ``tail_linewidth=`` also warns
when no dataset is drawn with a line (a markers-only format). All four may
also be given inside the ``animate={...}`` dict.

With a continuous ``hue=`` the tail and the trail are colored observation by
observation, whatever the format: a line (``'-'``), markers (``'o'``), or
both (``'-o'``, the colored line with a colored marker on each observation).

Fading earlier datasets
------------------------

``dataset_fade=`` fades already-revealed datasets by how long ago they were
revealed, on a serial reveal, so recent history reads brightest:

.. code-block:: python

    hyp.plot([a, b, c], '-', animate=True, order='serial',
             dataset_fade={'floor': 0.15, 'decay': 0.7})

The alpha for dataset ``i`` is ``floor + (1 - floor) * decay ** (current -
i)``, where ``current`` is the dataset being revealed right now; ``floor``
is what an infinitely old dataset fades to and ``decay`` is the per-dataset
falloff. It applies before any ``on_frame=`` callback, so a callback can
still override one dataset's alpha. It requires a serial reveal
(``animate='serial'`` or ``order='serial'``); a parallel animation has no
"how long ago" to decay over and raises ``ValueError``. Supported on both
backends -- matplotlib alpha, plotly per-frame trace opacity.

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

``title=`` also accepts two DYNAMIC forms, on both backends. A **callable**
gets that frame's :class:`~hypertools.FrameContext`, the same object
``on_frame=`` receives, and returns the string to draw:

.. code-block:: python

    hyp.plot(data, '-', animate=True,
             title=lambda ctx: f'{ctx.progress:.0%} through')

A **format pattern** containing ``{index`` reads the row index of a
pandas DataFrame or Series input at the reveal head -- a ``DatetimeIndex``
is the intended case:

.. code-block:: python

    df = pd.DataFrame(data, index=pd.date_range('2020-01-01', periods=12,
                                                 freq='MS'))
    hyp.plot(df, '-', animate=True, title='{index:%B %Y}')

This needs an input carrying a non-default index; without one it raises
``ValueError`` before the pipeline runs. Style either form with
``title_kwargs=``, ``title_color=`` and ``font=``, exactly like a fixed
title:

.. code-block:: python

    hyp.plot(data, '-', animate=True, title='{index:%B %Y}',
             title_kwargs={'size': 14, 'weight': 'bold'},
             title_color='darkred')

``title_wrap=`` hard-wraps a long title at a character count, on either
form:

.. code-block:: python

    hyp.plot(data, '-', animate=True,
             title='A rather long title that should wrap nicely',
             title_wrap=20)

A multi-line title (explicit newlines, or from ``title_wrap=``) now
reserves room for every line above an animated 3-D plot, not just one.

On the matplotlib backend the room above an animated plot is made for the
tallest title the animation shows: a fixed title's own lines, the tallest
entry of a per-segment list, and, for a callable or ``'{index...}'`` title,
the tallest of every frame's titles, so a title that grows lines on any
frame has its room from the start (the callable is called once per frame
for this before the animation is returned). The figure is
made taller by what the title needs, in 2-D and in 3-D, and nothing already
drawn moves or shrinks.

Per-dataset styling
-------------------

``color=``, ``linewidth=`` and -- new in 1.1.0 -- ``alpha=`` accept one value
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

Looping a morph
---------------

``loop=True`` closes an ``animate='morph'`` sequence: after the last cloud,
it transitions back to the first one and holds it again, so a looping
player never hard-cuts from the last shape to the first:

.. code-block:: python

    hyp.plot([a, b, c], '-', animate='morph', loop=True)

The closing hold reuses the FIRST cloud's own already-sampled points, so
the loop point does not jump the way appending ``clouds[0]`` as one more
dataset would. ``n`` clouds give ``2(n + 1) - 1`` hold/transition segments
instead of ``2n - 1``. ``title=`` still takes one entry per dataset, not
one per segment. Raises ``ValueError`` for every other ``animate=`` mode,
since those already end where a repeat would start. Supported on both
backends.

Per-frame callbacks
-------------------

``on_frame=`` runs your function for every frame that is drawn, with that
frame's :class:`~hypertools.FrameContext`. On plotly that is exactly once
per frame, while the frames are built; on matplotlib it is every time a
frame is rendered, so a frame can come around more than once -- see
:ref:`the callback contract <animation-callback-contract>` below.
**Passing it to** ``plot()`` **works on both backends** and is the portable
form:

.. code-block:: python

    def label_frame(ctx):
        # ctx.frame and ctx.n_frames are backend-independent
        print(f'frame {ctx.frame} of {ctx.n_frames}')

    hyp.plot(data, '-', animate=True, on_frame=label_frame)

The context carries the frame index and total, the resolved ``style`` and
``order``, the arrays being drawn, how far the reveal has gotten
(``revealed_counts``, ``window_bounds``), overall progress through the clip
(``progress``), and -- for morphs -- ``segment_index`` and
``segment_kind``. All of those are the same on either backend.

What you *do* with the context is usually backend-specific, because
``ctx.figure``, ``ctx.axes`` and ``ctx.artists`` are backend-native. On
matplotlib ``ctx.artists`` holds one artist per dataset's reveal head, then
the trails. Two things are drawn by a second artist that is not in the list
and follows the listed one: the markers of a line-and-marker format
(``'-o'``) under a continuous ``hue=``, which are hidden when the listed
line is hidden and take its ``alpha``; and a ``head_marker=``, which is
hidden when its dataset's head artist is hidden. So
``ctx.artists[0].set_visible(False)`` hides the whole trace. The
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
    # Shared artists, so this colors the WHOLE animation, not frame 0.
    def broken(ctx):
        if ctx.frame == 0:
            ctx.artists[0].set_color('red')

    def correct(ctx):
        ctx.artists[0].set_color(COLORS[ctx.frame])   # set it every frame

Where they are **per-frame**, the very same conditional does the opposite
-- it touches an independent payload that only that frame keeps::

    # PLOTLY ONLY -- ctx.artists are that frame's traces, and
    # parallel/serial/window/morph frames are independent, so this
    # colors ONLY frame 0.
    def also_broken(ctx):
        if ctx.frame == 0:
            ctx.artists[0].line.color = 'red'

    def also_correct(ctx):
        ctx.artists[0].line.color = COLORS[ctx.frame]

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
        color = HIGHLIGHT if ctx.frame == TARGET else DEFAULT
        ctx.artists[0].set_color(color)            # matplotlib spelling

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
An animated plotly plot returns a ``plotly.graph_objects.Figure``:
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

One repeat on matplotlib is certain: a save (``save_path=``, or ``.save()``
on the returned animation) draws frame 0 twice, so a callback is called
``n_frames + 1`` times for an animation of ``n_frames`` frames, single plot
or ``panels=`` grid alike. That is matplotlib's own doing:
``Animation.save`` first calls ``FuncAnimation._init_draw``, which draws
the first frame to initialize the figure, and then draws every frame, the
first included. (``Animation._start``, which starts an animation shown in
a window, makes the same initial draw.) A callback that is idempotent per
frame cannot tell.

Both backends deliver the same *context metadata* for a given frame index.
They do **not** produce interchangeable rendered output from a mutating
callback: ``ctx.figure``, ``ctx.axes`` and ``ctx.artists`` are
backend-native (on plotly ``ctx.axes`` is ``None`` and ``ctx.artists`` are
that frame's traces), so a callback that touches them is backend-specific
code. Each backend does guarantee that a mutation you make is retained in
the frame it renders.

Reveal progress in callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ctx.progress`` is how far through the whole clip the current frame is, in
``[0, 1]`` -- ``0.0`` on the first frame, ``1.0`` on the last. It is
populated on every style and both backends, including ``'spin'`` and
``'morph'``, where no dataset reveals row by row:

.. code-block:: python

    def label(ctx):
        print(f'{ctx.progress:.0%} through, revealed {ctx.revealed_counts}')

    hyp.plot(data, '-', animate=True, on_frame=label)

``ctx.revealed_counts`` is the number of rows of each drawn dataset
currently on screen, one entry per dataset. As of 1.1.0 it is populated for
``'parallel'``, ``'window'`` and ``'spin'`` too, not just serial reveals --
before, every style but serial reported ``None``. ``ctx.window_bounds`` is
the ``(start, end)`` row range actually visible for each dataset: ``start``
is 0 for an ordinary cumulative reveal and greater than 0 once a sliding
window (``animate='window'``, or a trail flag) has moved past a dataset's
beginning, and ``end`` always equals ``revealed_counts[i]``. Both are
``None`` for ``animate='morph'``, which interpolates whole clouds rather
than revealing rows.

Forecasting during an animation
--------------------------------

``predict=`` works with the time-progressing animation styles
(``animate=True``, ``'parallel'``, ``'serial'``, ``'window'``) on **both**
backends. The forecast is recomputed from the history revealed so far and
drawn from the endpoint the current frame draws, so the forecast trace grows
with the animation instead of standing still:

.. code-block:: python

    fig, ani = hyp.plot(data, '-', predict='Kalman', t=10,
                        animate=True, duration=8, frame_rate=20)

``t`` is measured in **raw observations of the analyzed data** -- not in
animation frames, and not in drawn vertices. ``t=1`` forecasts the next
observation. Because an animation is paced on a resampled frame grid (see
``duration``/``frame_rate``), a frame's drawn head usually falls *between* two
observations. The forecast starts exactly there, so it meets the trajectory it
continues on every frame; the points it predicts stay where the model put
them, ``t`` observations on from the last one revealed.

.. versionchanged:: 1.1.0
   The forecast used to hang off the last raw observation at or before the
   head, so it stood still for as many frames as the head took to reach the
   next observation and trailed the line's tip by up to a whole step -- most
   visibly on data with fewer rows than frames.

Everything is computed before the first frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Animating static data means every observation is known before drawing starts,
so every forecast the animation will *ever* draw is knowable up front. Two
things follow, and both are contracts rather than implementation details:

- The whole fan is folded into the plot's center/scale statistics, so it lands
  inside the cube **by construction**. Nothing is clipped or clamped.
- Each frame is a table lookup, so ``ani.save()`` and ``to_jshtml()`` replay
  identically no matter what order matplotlib asks for frames in.

Fits are memoized per (dataset, revealed-count), so a 900-frame animation of a
60-row dataset costs at most 59 fits rather than 900.

Keeping earlier forecasts on screen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``forecast_trail=`` is the forecast analog of ``chemtrails=``: earlier
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
of the observed trace it continues -- the same **color**, **linestyle** and
**linewidth** -- and differs only in transparency:

.. code-block:: text

    forecast_alpha = observed_alpha * 0.5

An observed line with no ``alpha=`` set is matplotlib's *opaque*, i.e. 1.0, so
the default forecast alpha is 0.5. Per-dataset styling carries through
dataset by dataset: ``alpha=[1.0, 0.4]`` gives forecasts at ``[0.5, 0.2]``, and
a dotted dataset gets a dotted forecast. Both backends apply the identical
rule. On plotly the forecast trace copies the color, width and dash, and
carries the alpha in the ``rgba(...)`` line color of a 2-D trace or in the
trace ``opacity`` of a 3-D one; either way the value is echoed in
``meta['hyp_forecast_alpha']``.

.. versionchanged:: 1.1.0
   Before 1.1.0 every forecast was drawn ``linestyle='--'`` at a hard-coded
   ``alpha=0.6``, whatever its data looked like -- so a forecast of a dotted,
   hairline or already-translucent dataset read as a *different* series rather
   than as its continuation.

``forecast_trail=`` fades from **that** dataset's live forecast alpha, down to
a floor proportional to it -- so a retained forecast is never more opaque than
the live forecast it decays from, however faint the dataset.

A forecast given its **own color** -- by ``forecast_palette=``,
``forecast_hue=``, ``forecast_cluster=``, or a color letter in
``forecast_fmt=`` -- keeps its trace's alpha instead of halving it: the color
is then what tells it apart, and fading it as well hid it among translucent
traces.

A **collection of models** (``predict=['Kalman', 'ARIMA', ...]``) draws one
overlay per model on every trace. Each keeps its dataset's color (which series
it continues) and takes a linestyle per model -- solid, dashed, dotted,
dash-dot, in model order -- so the two questions are answered by two
encodings; ``forecast_palette=`` colors by model instead, and
``forecast_fmt=`` (one entry per model) replaces the cycle.

Every ``predict=`` form lists its forecast **once in the legend**, static or
animated, under the model's name (the same name ``hyp.predict(x, model=[...])``
gives it), after the data entries and before ``truth``. The key wears the
forecasts' linestyle and their color when they share one; when one model's
forecasts of several datasets wear several colors the key is a neutral gray,
and it is never drawn below 0.8 alpha, so it stays legible however faint the
forecasts are.

Animated forecasts under ``hue=``/``cluster=``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``hue=`` or ``cluster=`` splits a trajectory into per-category runs, each
run is drawn as its own trace, but the reveal still follows the **dataset**:
one run finishes as the next begins, so the trajectory sweeps once and changes
color as it crosses a category boundary. (Before 1.1.0 every run advanced
at once, so one trajectory animated in several places simultaneously.) A forecast
is fit per dataset from exactly the observations on screen, so it means the
same thing it does without ``hue=``:

.. code-block:: python

    # both draw forecasts, and agree at every frame
    hyp.plot(data, '-', predict='Kalman', t=10, hue=categories)
    hyp.plot(data, '-', predict='Kalman', t=10, hue=categories, animate=True)

A live forecast inherits the color of the run drawing the head, and therefore
changes color with it; a retained ``forecast_trail=`` member keeps the color
it was drawn with, so the fan records the history rather than being repainted.
Pass ``forecast_hue=``, ``forecast_cluster=`` or ``forecast_palette=`` to give
the forecasts a grouping of their own; those are resolved once from the
full-history forecasts and stay fixed for every frame.

Because each run is resampled onto its own frame grid, a regrouped reveal can
lag the un-regrouped one by up to one grid step -- under a single frame of
trajectory, and never early, so a forecast is never fit on an observation that
has not been drawn.

Both backends draw this identically, frame for frame; that is worth stating
because they nearly did not. Each pace their reveal with the same shared
window arithmetic rather than a transcription of it, and both take their
per-frame forecast rows from the same schedule.

**Marker-only** categorical regrouping is the one case still refused. With a
marker format, ``hue=`` groups every observation of a category together
regardless of which dataset it came from, so the drawn traces are categories
rather than datasets and there is nothing to anchor a per-dataset forecast to.
``plot()`` says so instead of dropping it silently, and the forecasts are
still *computed* -- ``return_model=True`` reports them:

.. code-block:: python

    bundle = hyp.plot(data, 'o', predict='Kalman', t=10, hue=categories,
                      animate=True, return_model=True)

    bundle['predict']['forecasts']    # the fit -- one array per drawn trace
    bundle['predict']['drawn']        # False
    bundle['predict']['draw_reason']  # why the figure has none

``return_model=`` hands back model output, so a fit that succeeded is reported
whether or not the figure could render it; ``drawn`` is what keeps "no
forecast was computed" and "a forecast was computed but not drawn"
distinguishable.

Styling the forecasts separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inheritance is the default, and four kwargs each replace one aspect of it.
Everything they do not name stays inherited.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - What a forecast's color means
   * - *(nothing)*
     - the identity of the observed trace it continues
   * - ``forecast_palette=``
     - one color per forecast from a palette of its own (one per *model*
       for a collection), drawn at the trace's own alpha
   * - ``forecast_hue=``
     - a grouping you supply, one value per forecast (see below)
   * - ``forecast_cluster=``
     - which forecasts are heading to the same place

``forecast_hue=`` and ``forecast_fmt=`` take one value per *drawn trace*,
which is one per input dataset until a hierarchical (MultiIndex) ``x=``
regroups the data: ``plot()`` forecasts every final trace, so a hierarchy
wants one value per leaf group **plus** one per derived mean.

``forecast_fmt=`` sets the line/marker style, in the same format-string
grammar as ``fmt``, and changes nothing else -- unless the string carries a
color letter (``'r:'``), which recolors the forecast too, on both backends:

.. code-block:: python

    # dotted forecasts, still in each trace's own color
    hyp.plot(data, '-', predict='Kalman', t=10, forecast_fmt=':')

    # observed data grouped by condition; forecasts colored by destination
    hyp.plot(data, '-', predict='Kalman', t=10, hue=conditions,
             forecast_cluster='KMeans', forecast_n_clusters=3,
             forecast_palette='viridis')

``forecast_cluster=`` clusters the forecast **endpoints** -- where each series
is predicted to end up -- so its colors answer a question the observed data
cannot: *which of these are heading to the same place?* That is why it does
not simply recluster the observed data; inheriting the observed assignment is
what the default already gives, so defining it that way would make the kwarg
a no-op. Nor does it cluster every predicted point (one forecast would then
change color along its own short path) or whole flattened trajectories
(sensitive to ``t``, to sampling and to dimensionality, where an endpoint has
one stable meaning). Endpoints are taken in the space the figure draws, after
``reduce=``/``align=``, so the grouping matches the geometry on screen.

In an **animation**, forecast endpoint groups are resolved once from the
full-history forecasts -- the same forecasts ``return_model=True`` hands back
in the model bundle -- and stay fixed for every frame. They are not
reclustered per frame, so the colors hold still while the forecast geometry
evolves.

That is a deliberate choice rather than an implementation detail. Cluster
labels are arbitrary names for groups, so reclustering each frame would let
a forecast change color whenever the fit shifted a point across a boundary,
and would repaint a whole retained ``forecast_trail=`` fan whose earlier
members were drawn under the old grouping. A fixed assignment also keeps a
saved animation identical to a played one, which is the same reason the
schedule itself is precomputed.

``forecast_hue=`` and ``forecast_cluster=`` are mutually exclusive -- both
decide how the forecasts are grouped -- exactly as ``hue=`` and ``cluster=``
are for the observed data. ``forecast_n_clusters=`` is separate from
``n_clusters=`` on purpose: the observations and the forecast endpoints are
different point sets, and a good number of groups for one need not be a good
number for the other.

All of them require ``predict=``; without it they raise ``ValueError`` rather
than being quietly dropped, since a silently ignored style kwarg leaves you
looking at an unchanged plot with no clue which argument did nothing.

Identifying forecast artists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast artists carry an explicit tag, so a per-frame callback can find them
without guessing from linestyle (which, since a forecast now inherits its
trace's linestyle, no longer distinguishes them at all) --
``artist._hyp_forecast_role`` on matplotlib
(``'static'``, ``'live'`` or ``'trail'``) and ``trace.meta['hyp_forecast_role']``
on plotly. Trail artists additionally carry ``_hyp_forecast_age``.

Every forecast artist also names the series it belongs to:
``artist._hyp_forecast_dataset`` on matplotlib, ``trace.meta['hyp_dataset']``
on plotly -- an index into the original input datasets. The role tag says what
an artist *is*; this says *whose* it is, so a callback (or hypertools itself)
can pair a forecast with its data by identity rather than by drawing order.

Note that forecast artists are deliberately **not** in
``FrameContext.artists``: that sequence has cardinality assumptions (one entry
per drawn dataset) that a variable number of forecast overlays would violate.

The *observed* trajectories are tagged the same way, and for the same reason:
neither ``ax.collections`` nor ``fig.data`` is a list of data artists. On
matplotlib a multicolored (continuous ``hue=``) trace is a
``Line3DCollection`` carrying ``collection._hyp_trace_index``, alongside the
six wireframe collections that draw the bounding cube; on plotly every data
trace carries ``trace.meta['hyp_trace_index']``, alongside the cube trace, any
density/surface layer, the forecast overlays and the ``colorbar=True``
carrier -- none of which is named. Both tags are the trace's index in the
drawn data, so a callback can pair a trace with its source without counting
positions.

A multicolored trace also carries its ``alpha=`` **inside** those per-point
colors, on both backends: matplotlib folds it into the segment colors of the
collection that replaces the line artist (an alpha left on the discarded
artist would simply be lost), and plotly serializes the line's color array as
``rgba(...)`` for the same reason. Per-point **marker** colors carry no alpha
on either backend -- a hue-colored scatter is drawn at full opacity.

What ``animate='morph'`` does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``animate='morph'`` (including the per-dataset morph list form) raises
``NotImplementedError`` with ``predict=``. A morph interpolates between point
*clouds*, so there is no time axis to forecast along -- this is a statement
about what a morph means, not a gap to be filled later.

Animated panel grids
--------------------

.. versionadded:: 1.2.0

``panels=`` (alias ``subplots=``) draws one panel per dataset, and with
``animate=`` the whole grid is ONE animation (matplotlib only): every panel
runs on the same clock, and a single ``HyperAnimation`` is returned for the
figure, so one ``save_path=`` writes one video of all the panels:

.. code-block:: python

    anim = hyp.plot([a, b, c, d], '-', panels=(1, 4), animate=True,
                    duration=10, tail_duration=2, rotations=1,
                    title=['A', 'B', 'C', 'D'], save_path='grid.mp4')

Each panel is animated exactly as the single-axes call for its data is:
every ``animate=`` style and ``order=`` value, the trail flags, ``predict=``
and per-frame titles apply per panel. The ordering is per
panel too, not a sequence across panels: ``order='serial'`` reveals the
datasets *within* each panel one after another, and the panels never take
turns. ``panel_fit=`` and a list-valued ``reduce=`` (one panel per reducer)
work as they do for a static grid.

A few things are specific to a grid:

* Without ``size=`` the figure is sized by the grid -- 3.2 in per panel each
  way, and at least 6.4 in wide -- so a saved clip of a row of panels is not
  mostly blank canvas. It is then widened by the strip of a shared legend or
  colorbar, made taller by a ``caption=``, and given room for axis labels where
  they need it (below). A static grid is laid out by the same code (since
  1.2.0; it was a fixed 6.4 x 4.8 in before): a still and a clip of the same
  grid animated with ``rotations=0`` have the same figure size, panel
  positions and cubes, and a rotating grid differs only by the room its
  cubes' sweep needs. ``size=`` sets the whole figure's size as usual.
* Shared axis labels are drawn on the outer panels only, as in a static
  grid: a label that is the same in every panel of a column is drawn under
  the column's lowest panel, one that is the same along a row on one panel
  of the row (see ``panels=`` in :func:`hypertools.plot`).
* ``animate=`` may be given in its dictionary form
  (``animate={'style': 'spin', 'rotations': 2}``) exactly as for a single
  plot; per-dataset values inside it go to their panels, and
  ``{'style': False}`` is a static grid.
* Every panel must draw the same number of frames, or the call raises
  ``ValueError`` (``animate='morph'`` needs the same number of datasets in
  each panel).
* Each 3-D panel's cube starts at the size a static panel's is drawn at
  (scaled by ``zoom=``) and is zoomed out as far as it takes to stay inside its
  own cell through the whole rotation, so ``zoom=`` can shrink a panel's
  cube but not enlarge it past its cell. The axis labels a panel draws
  (``xlabel=``/``ylabel=``/``zlabel=``, or a DataFrame's column names) swing
  around its cube, so the cubes are drawn smaller still -- all of them
  alike -- until the labels stay inside the panel's own cell at every view
  the animation takes. With ``rotations=0`` there is one view: the outer
  panels' labels hang toward the grid's outer edge, the figure is given
  room for them there, and the cubes keep nearly their unlabeled size. The
  figure is enlarged only where the cubes would be under 70% of their
  unlabeled size, and by no more than restores
  that. An explicit ``size=`` is honored down to cubes half their unlabeled
  size and raises below that, naming a size that works; pass ``xlabel='', ylabel='', zlabel=''`` to drop the labels instead.
* A per-frame ``title=`` (a callable or an ``'{index...}'`` pattern, one for
  every panel or a list with one per panel) gets its room above each panel
  like a fixed title. The room is made for the tallest of the first, middle
  and last frames' titles, so a title that grows lines as the animation
  runs stays on the canvas and clear of the row above (a frame in between
  that needs more lines than all three runs above its room).
* Colors, the legend and the colorbar belong to the whole grid, as in a
  static grid: each dataset keeps the color the single-axes call of all the
  datasets gives it, ``hue=`` is read (categorical or continuous) and
  mapped through one scale, a ``legend=`` list names the grid's datasets or
  its categories, and one legend
  and/or one colorbar is drawn in a strip right of the panels. They are
  static across frames, and every rotating cube stays clear of the strip.
  ``legend_kwargs={'loc': ...}`` moves the legend's strip to another side of
  the grid (``'upper center'`` above it, ``'lower center'`` below it,
  ``'center left'`` left of it, ``'upper right'`` at the top of the right-hand
  strip); ``bbox_to_anchor`` places it yourself in figure fractions, with no
  room reserved.
  (Clusters fit per panel -- ``panel_fit='independent'`` or a list-valued
  ``reduce=`` with ``cluster=``/``n_clusters=`` -- keep a legend per panel,
  because their colors mean something different in each one.)
* ``on_frame=`` is called once for the whole figure each time a frame is
  drawn, not once per panel (a save draws frame 0 twice, so a callback
  sees it twice; see
  :ref:`the callback contract <animation-callback-contract>`). Its
  ``FrameContext`` has ``axes`` as a tuple of the panel axes, ``artists``
  and ``datasets`` as every panel's in panel order, and ``panels`` as one
  ordinary per-panel ``FrameContext`` each (that panel's own axes, artists,
  ``revealed_counts`` and so on).
* ``backend='plotly'`` raises ``NotImplementedError`` (a static plotly grid
  works), ``companion=`` raises ``ValueError``, and so do panels whose
  analyzed data differ in width.

Captions
--------

.. versionadded:: 1.2.0

``caption=`` (matplotlib only) draws one line of text centered under the
figure -- under a single plot, or under a whole ``panels=`` grid -- and
updates it on every frame. It takes the forms a per-frame ``title=`` takes:
a plain string, a ``'{index...}'`` pattern formatted with the input's own row
index under the reveal head, or a callable given the frame's
``FrameContext``:

.. code-block:: python

    hyp.plot([a, b, c, d], '-', panels=(1, 4), animate=True,
             caption=lambda ctx: f'timepoint {ctx.frame + 1} of {ctx.n_frames}')

    hyp.plot(df, '-', animate=True, caption='{index:%B %Y}')

The caption sits a small gap below the lowest thing drawn; where the figure
has no blank room there (an animated 3-D plot fills its canvas) the figure
is made taller by the caption's height, so the caption never overlaps a
panel, a legend or a colorbar and never leaves the canvas. A caption wider
than the figure is wrapped to the figure's width, on every frame. Room is
made for the tallest of every frame's wrapped captions, so a caption that
grows on any frame has its room from the start (a callable caption is
called once per frame for this before the animation is returned). An
empty caption draws nothing and adds no room. On a static plot the caption
is drawn once, a pattern or callable being resolved for the finished figure.

A callable receives the frame's real ``FrameContext`` (for a grid, the
figure-level one, with one context per panel in ``ctx.panels``). Under a grid
an ``'{index...}'`` pattern shows what a panel's own ``title='{index}'``
shows: the first dataset that carries an index is read, at the reveal head of
the panel that draws it (the first panel under a list-valued ``reduce=``).
``caption=`` raises ``NotImplementedError`` under ``backend='plotly'`` (a
per-frame ``title=`` works on both backends) and ``ValueError`` with a
caller-supplied ``ax=``.

.. versionadded:: 1.2.1

``caption_kwargs=`` styles and places the caption: ``size``, ``color``,
``weight``, ``style``, ``family``, ``alpha``, ``linespacing``, ``position``
(``'bottom'``, the default, or ``'top'``) and ``loc`` (``'center'``,
``'left'`` or ``'right'``). The style is applied before the caption is
wrapped and its room measured, and a per-frame caption is measured on every
frame, so a caption of any size is never clipped. matplotlib decides what
each key accepts (a size may be a number, a numeric string or a size name,
and a key set to ``None`` keeps its default); a value it refuses, or a size
that is not positive and finite, raises ``ValueError`` naming the argument
and the key.

``suptitle=`` is one fixed title for the whole figure, above everything it
holds. It is the figure-level title of a ``panels=`` grid, static or
animated, where ``title=`` names the panels; ``suptitle_kwargs=`` styles it
with the same keys (except ``position``). Its room is measured like the
caption's, so it clears the panel titles, the shared legend and colorbar,
and a caption placed at the top:

.. code-block:: python

    hyp.plot([a, b, c, d], '-', panels=(2, 2), animate=True,
             title=['a', 'b', 'c', 'd'], suptitle='Four walks, one clock',
             caption=lambda ctx: f'frame {ctx.frame + 1} of {ctx.n_frames}',
             caption_kwargs={'size': 13, 'color': 'gray', 'loc': 'right'})

Companion panels
-----------------

``companion=`` (matplotlib only) draws extra 2-D panels beside an animated
plot, revealed in lockstep with it -- a linked time-series panel that used
to require building a second axes by hand and driving it from an
``on_frame=`` hook:

.. code-block:: python

    hyp.plot(data, '-', animate=True,
             companion={'kind': 'series', 'data': data[:, 0],
                        'ylabel': 'x[0]'})

Each panel is a dict naming the series to draw (``data``, read against the
same rows a ``title='{index...}'`` pattern reads), whether it reveals up to
the head or is drawn whole (``reveal``), an optional rolling-mean trend
line (``smooth``), and layout (``position``, ``size``, ``pad``). What it
does NOT do: no 3-D companion panels, no panel with its own animation
schedule, no legend or colorbar of its own, and no interaction with ``ax=``
or ``panels=``. There is no plotly support -- an animated plotly figure is
a list of frame payloads over one figure's traces, so a companion panel
would need its own subplot layout and per-frame payloads; passing
``companion=`` under ``backend='plotly'`` raises ``NotImplementedError``.

Measuring what was drawn
--------------------------

``anim.drawn_extent()`` returns the union bounding box of everything the
animation actually draws, measured from rendered pixels rather than data
limits, as a ``matplotlib.transforms.Bbox`` in figure fractions:

.. code-block:: python

    anim = hyp.plot(data, '-', animate=True)
    bbox = anim.drawn_extent()

By default it samples 12 frames evenly across the clip, since a 3-D
animation's drawn extent changes with the camera angle and one frame is
not representative. Pass an int for a different sample size, or an
iterable of frame indices to measure exactly those. It costs one full
canvas render per sampled frame and leaves the figure showing the last
frame it measured.

Migrating from ``_func``/``_args``
----------------------------------

Before 1.1.0 the only way to run code per frame was to monkeypatch
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
