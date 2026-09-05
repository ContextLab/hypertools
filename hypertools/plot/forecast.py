#!/usr/bin/env python
"""Forecast scheduling for animated `hyp.plot(..., predict=...)` calls.

A forecast used to be a fixed overlay drawn once, so `predict=` refused every
animate mode that reveals data over time. Animating STATIC data (every
observation known up front, merely revealed frame by frame) means every
forecast the animation will ever draw is knowable up front too -- so this
module computes the whole schedule BEFORE drawing, folds it into the display
bounding box, and lets each frame be a table lookup.

Three spaces, never conflated (see the plan's Contract 2):

- ANALYZE space  -- `xform` post normalize/reduce/align, pre-resample,
                    pre-centre/scale. Forecasts are computed HERE, and `t` is
                    measured in these RAW samples.
- FRAME GRID     -- `plot._interp_anim_line` resamples every animated line
                    dataset to exactly `round(frame_rate * duration)` rows.
- DISPLAY box    -- the centred/rescaled [-1, 1] cube (`plot.py:4568-4585`).
"""

import time
import warnings

import numpy as np

from ..predict.predict import predict as _predict

# matplotlib's own fmt-string grammar. `forecast_fmt=` promises to mean
# exactly what the same string means in `fmt=`, which is only true if the
# same parser decides what is legal -- so it is checked here, once, rather
# than by whichever backend happens to be drawing. Guarded (this is a
# private matplotlib API): a future relocation degrades to "unvalidated"
# rather than crashing at import.
#
# This is the SINGLE guarded import: `plot.py` takes the symbol from here,
# so the two cannot end up parsing `forecast_fmt=` with different parsers.
try:
    from matplotlib.axes._base import _process_plot_format
except ImportError:  # pragma: no cover
    _process_plot_format = None

#: Project a schedule's build time once the first fit has been timed, and say
#: so if it exceeds this many seconds. Striding the schedule to make it
#: cheaper was ruled out deliberately -- sampling the reveal would change
#: WHAT IS PLOTTED, and the outcome matters more than the speed. So a large
#: dataset is simply slow, and the library's obligation is notice rather than
#: a quietly different animation.
#:
#: The projection is extrapolated from the first REAL fit, which carries
#: one-time warm-up, so it tends to over-state: measured 427 s projected
#: against 330 s actual for 3 x 500 rows x 900 frames. Over-stating is the
#: safe direction for a notice, but it is why the message calls itself a
#: rough figure rather than a countdown.
DEFAULT_SLOW_WARNING_SECONDS = 10.0

#: Fewest observations we will fit a forecaster to.
DEFAULT_MIN_HISTORY = 2

#: A forecast is drawn at this fraction of its OBSERVED trace's alpha.
#:
#: Maintainer decision (1.1.0): a forecast should read as the same series
#: projected forward, so it inherits its observed trace's identity --
#: colour, linestyle AND linewidth -- and differs ONLY in transparency.
#: This deliberately replaces the pre-1.1.0 rule (always ``linestyle='--'``
#: at a hard-coded ``alpha=0.6`` regardless of how the data was drawn),
#: under which a forecast of a dotted, hairline or already-translucent
#: dataset looked like a different series rather than its continuation.
#:
#: This constant is the single source of truth for BOTH backends:
#: `hypertools.plot.plot._forecast_style_from` (matplotlib) and
#: `hypertools.plot.plotly_backend._forecast_style_from` (plotly) express
#: the same policy in their own terms, so the two cannot drift.
FORECAST_ALPHA_SCALE = 0.5

#: `trail_alpha`'s floor, as a FRACTION of the LIVE forecast's alpha.
#:
#: Relative, not absolute: an absolute floor on a faint dataset would make
#: an OLD trail more opaque than the live forecast it decays from (with
#: `alpha=0.05` the live forecast is 0.025, well under the old absolute
#: 0.08 floor). 0.16 of the default live alpha (0.5) is 0.08 -- the same
#: floor the absolute constant gave for a fully-opaque dataset.
TRAIL_FLOOR_FRACTION = 0.16


def forecast_alpha(observed_alpha, alpha_scale=FORECAST_ALPHA_SCALE):
    """The alpha to draw a forecast at, given its observed trace's alpha.

    Parameters
    ----------
    observed_alpha : float or None
        The observed trace's alpha. ``None`` is matplotlib's "no alpha set",
        i.e. fully opaque, and counts as 1.0 (so the default forecast alpha
        is `alpha_scale` itself).
    alpha_scale : float, default `FORECAST_ALPHA_SCALE`
        Fraction of the observed alpha to draw the forecast at.

    Returns
    -------
    float
    """
    base = 1.0 if observed_alpha is None else float(observed_alpha)
    return base * float(alpha_scale)


def forecast_from_history(history, model, t, min_history=DEFAULT_MIN_HISTORY):
    """Forecast `t` steps on from `history`, as a displacement path.

    Parameters
    ----------
    history : array-like, shape (n_observed, n_dims)
        The trajectory revealed so far, in ANALYZE space (already reduced,
        not yet resampled onto the frame grid and not yet centred/scaled).
    model : str or dict
        Anything `hypertools.predict` accepts ('Kalman', 'ARIMA', 'Laplace',
        'GaussianProcess', 'Chronos', ...).
    t : int
        Forecast horizon, in RAW analyze-space steps. ``t=1`` is the next
        observation.
    min_history : int, default 2
        Refuse to forecast from fewer rows than this.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(t + 1, n_dims)``, dtype float64. Row 0 is all zeros (the
        anchor itself), so ``history[-1] + result`` is the forecast path in
        analyze space. ``None`` when `history` is shorter than `min_history`
        -- callers must hide the artist rather than draw an empty trace.

    Notes
    -----
    `hypertools.predict` returns a ``pandas.DataFrame``; its index is
    deliberately discarded here (a forecast's index is a continuation the
    plotting code has no use for).
    """
    history = np.asarray(history, dtype=float)
    if history.ndim != 2:
        raise ValueError(
            f"history must be 2-D (n_observed, n_dims); got shape "
            f"{history.shape}.")
    if len(history) < max(2, min_history):
        return None

    forecast = np.asarray(_predict(history, model=model, t=t), dtype=float)
    # `predict` returns exactly `t` NEW rows -- every one of them a future
    # step -- so the last OBSERVED row is the anchor. Using forecast[0] as the
    # anchor would throw away a whole step and force the first displacement
    # to zero (the bug the market gallery example shipped with).
    displacement = forecast - history[-1]
    return np.vstack([np.zeros((1, history.shape[1])), displacement])


def revealed_raw_counts(n_raw, n_grid, num, total_frames):
    """RAW analyze-space rows revealed at frame `num` (parallel/window).

    `update_lines_parallel` reveals `data[start:end]` of the FRAME-GRID array,
    where `end` comes from `trails.anim_window_bounds` -- the one
    implementation of the reveal, called from `matplotlib_backend.py:1185`. It
    is reused here rather than re-derived. (Since GH #285
    `FrameContext.revealed_counts` IS populated for parallel animations, but
    it counts FRAME-GRID rows -- what the artist drew -- while this function
    answers the different question of how many RAW analyze-space rows that
    corresponds to, which is what the forecast anchor needs.) `end` does not
    depend on the trail window, so 0 is passed for it.

    `plot._interp_anim_line` puts frame-grid row ``j`` at RAW parameter
    position ``j * (n_raw - 1) / (n_grid - 1)`` with exact endpoints, so the
    last raw sample at or before the drawn head (grid row ``end - 1``) is
    index ``floor(pos)`` and ``floor(pos) + 1`` rows are revealed.
    """
    from .trails import anim_window_bounds
    n_raw = int(n_raw)
    n_grid = int(n_grid)
    if n_grid < 2 or n_raw < 2:
        return n_raw
    _, end, _ = anim_window_bounds(num, total_frames, n_grid, 0)
    pos = (end - 1) * (n_raw - 1) / (n_grid - 1)
    return min(n_raw, int(np.floor(pos)) + 1)


class DatasetRevealSchedule:
    """Which ORIGINAL rows of each source dataset are on screen at each frame.

    `hue=`/`cluster=` draw one trace per contiguous same-category run, so the
    backends' own reveal is expressed in RUNS. A forecast is fit per DATASET,
    from the observations revealed so far, and needs the inverse: original row
    indices, in temporal order, for a dataset that may be spread over several
    traces.

    The rows are read back off the `RunWindow`s
    `trails.dataset_window_bounds` produces -- the SAME objects the backends
    slice their artists with -- rather than recomputed from the frame index.
    Two derivations of "what is on screen" can drift while each passes its own
    tests; one cannot. The bridge vertex `patch_lines` appends is covered by
    the same rule, because a bridged run's drawn span reaches it: the frame on
    which a run completes is the frame on which its successor's first
    observation becomes visible, through either trace.

    `visible_rows` returns the row TUPLE rather than a count. Under the fixed
    reveal it is always a prefix, so the two carry the same information today;
    the tuple is what `ForecastSchedule` memoizes on, so a future reveal that
    exposed a non-prefix would produce a different cache key rather than a
    silent collision between two equal-sized but different histories.
    """

    def __init__(self, ownership, grid_lengths, n_frames, window_frames,
                 serial=False):
        from .trails import dataset_window_bounds, run_head_param
        self.ownership = ownership
        self.grid_lengths = [int(g) for g in grid_lengths]
        self.n_frames = int(n_frames)
        self.window_frames = int(window_frames)
        self.serial = bool(serial)
        self._rows = []
        for frame in range(self.n_frames):
            if self.serial:
                counts = self._serial_counts(frame)
            else:
                windows = dataset_window_bounds(
                    frame, self.n_frames, ownership, self.grid_lengths,
                    self.window_frames)
                counts = []
                for d in range(ownership.n_datasets):
                    head = None
                    for r in ownership.runs_of(d):
                        p = run_head_param(windows[r], ownership, r)
                        if p is not None:
                            head = p if head is None else max(head, p)
                    counts.append(0 if head is None
                                  else min(ownership.row_count(d),
                                           int(head) + 1))
            self._rows.append([tuple(range(k)) for k in counts])

    def _serial_counts(self, frame):
        """`order='serial'` already sweeps runs in order (`serial_reveal_counts`
        walks the trace list), so a dataset's count is the sum of its runs'."""
        from .matplotlib_backend import serial_reveal_counts
        own = self.ownership
        grid_counts = serial_reveal_counts(
            list(self.grid_lengths), frame, self.n_frames)
        out = []
        for d in range(own.n_datasets):
            total = 0
            for r in own.runs_of(d):
                g = self.grid_lengths[r]
                _, n_rows = own.run_span(r)
                span = own.draw_span(r)
                shown = min(grid_counts[r], g)
                if g < 2 or span <= 0 or shown <= 0:
                    total += min(n_rows, max(0, shown))
                else:
                    pos = (shown - 1) * span / (g - 1)
                    total += min(n_rows, int(np.floor(pos)) + 1)
            out.append(min(own.row_count(d), total))
        return out

    def visible_rows(self, dataset, frame):
        """Original row indices of `dataset` on screen at `frame`, in order."""
        return self._rows[min(max(int(frame), 0), self.n_frames - 1)][dataset]

    def head_run(self, dataset, frame):
        """The run DRAWING this dataset's last visible row, or `None` when
        nothing of it is on screen yet."""
        rows = self.visible_rows(dataset, frame)
        if not rows:
            return None
        return self.ownership.run_holding(dataset, rows[-1])


class DisplayTransform:
    """The centre/scale affine `plot()` applies at `plot.py:4569-4582`.

    ``2 * (((a - mean) - offset) / scale) - 1``. Recorded at setup so a
    forecast computed in ANALYZE space can be mapped into the SAME display
    box the data was mapped into -- rather than being recomputed from
    function-locals that no longer exist by frame time.
    """

    __slots__ = ('mean', 'offset', 'scale')

    def __init__(self, mean, offset, scale):
        self.mean = np.asarray(mean, dtype=float)
        self.offset = float(offset)
        self.scale = float(scale) or 1.0

    def __call__(self, a):
        centred = np.asarray(a, dtype=float) - self.mean
        return 2.0 * ((centred - self.offset) / self.scale) - 1.0


def project_schedule_cost(timings, remaining_rows):
    """Project the seconds still to come from fits already TIMED.

    `timings` maps a revealed-history length to the seconds its fit took
    (one entry per length, already pooled); `remaining_rows` is the history
    length of every fit still to do. Returns
    ``(projected_seconds, per_row, setup, (short, long))``.

    A fit costs roughly ``setup + per_row * rows`` -- measured ~30 ms at
    60x3 and ~220 ms at 500x3 -- so both terms have to come out of the data:

    * a constant per fit is an order of magnitude wrong on exactly the
      schedules this exists to warn about, because `todo` is ordered by
      GROWING revealed history and the earliest fits are the cheapest there
      will ever be (measured on a gallery figure: 12.9 s projected against
      176 s spent, 13.6x low);
    * a pure per-row rate over-projects a small schedule, because the first
      fit also carries one-off setup.

    Two points separate the constant from the slope, and they must be at
    DIFFERENT lengths -- which is the whole reason this takes a mapping
    keyed by length rather than a list of samples. See
    `ForecastSchedule.__init__` for the ordering that makes that a live
    hazard rather than a hypothetical one.
    """
    if len(timings) < 2:
        raise ValueError(
            f"projecting a schedule needs timed fits at two DIFFERENT "
            f"history lengths; got {sorted(timings)}")
    short, long = min(timings), max(timings)
    per_row = (timings[long] - timings[short]) / (long - short)
    per_row = max(per_row, 0.0)          # noise can invert two samples
    setup = max(timings[long] - per_row * long, 0.0)
    projected = sum(setup + per_row * rows for rows in remaining_rows)
    return projected, per_row, setup, (short, long)


class ForecastSchedule:
    """Every forecast an animation will ever draw, computed before drawing.

    Built from STATIC data (all observations known up front, revealed frame
    by frame), which is what makes precomputation possible and what lets the
    display bounding box be built to contain every forecast -- so nothing is
    ever clamped. Streaming data uses a different rule entirely; see
    `hypertools/io/streaming.py:382-401`.

    `counts[f][i]` is the number of RAW analyze-space rows dataset `i` has
    revealed at frame `f`. Fits are memoized on `(i, count)`, so a 900-frame
    animation of a 60-row dataset costs at most 59 fits.
    """

    def __init__(self, histories, counts=None, model=None, t=None, rows=None,
                 min_history=DEFAULT_MIN_HISTORY, transform=None,
                 slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS):
        if (counts is None) == (rows is None):
            raise ValueError(
                "pass exactly one of counts= (a revealed ROW COUNT per "
                "dataset per frame) or rows= (the revealed ROW INDICES); got "
                f"{'both' if counts is not None else 'neither'}.")
        self.histories = [np.asarray(h, dtype=float) for h in histories]
        # ONE internal representation. A count `k` means "the first k rows",
        # which is what every reveal has produced since the runs of a dataset
        # were given a shared clock -- but the KEY is the row tuple, so two
        # frames exposing equal-sized DIFFERENT histories cannot collide in
        # the cache. That is the whole reason this is not `(dataset, count)`.
        if rows is None:
            rows = [[tuple(range(int(k))) for k in frame] for frame in counts]
        self.rows = [[tuple(int(i) for i in r) for r in frame]
                     for frame in rows]
        self.counts = [[len(r) for r in frame] for frame in self.rows]
        self.model = model
        self.t = int(t)
        self.min_history = int(min_history)
        self.transform = transform
        self.n_frames = len(self.counts)
        self.n_datasets = len(self.histories)
        self.n_fits = 0
        self._paths = {}

        # Every DISTINCT (dataset, revealed-ROWS) pair needs one fit; the
        # rest are cache hits. Known before any work is done, so the size of
        # the job is knowable up front even though its cost is not.
        todo = []
        for frame_rows in self.rows:
            for i, r in enumerate(frame_rows):
                if (i, r) not in self._paths:
                    self._paths[(i, r)] = None
                    todo.append((i, r))
        self._paths.clear()

        warned = slow_warning_seconds is None
        # Seconds keyed by revealed-history LENGTH, for fits that really
        # ran. A mapping rather than a list of samples, because the slope
        # needs two DIFFERENT lengths -- see the sampling comment below.
        timings = {}
        spent = 0.0                  # wall clock actually spent so far
        self.projection = None       # filled in when a projection is made
        for n_done, (i, r) in enumerate(todo):
            start = time.perf_counter()
            path = forecast_from_history(self.histories[i][list(r)],
                                         self.model, self.t,
                                         min_history=self.min_history)
            elapsed = time.perf_counter() - start
            spent += elapsed
            if path is not None:
                self.n_fits += 1
            self._paths[(i, r)] = path
            # Time only REAL fits. The earliest (dataset, count) pairs are
            # histories shorter than min_history, where forecast_from_history
            # returns None without fitting anything -- timing one of those
            # projects 0.0 s for a job that may take minutes, which is worse
            # than not warning at all.
            if path is not None:
                timings.setdefault(len(r), []).append(elapsed)
            # Wait for two DISTINCT history lengths. `todo` is ordered by
            # FRAME and then by DATASET, so every dataset is fitted at one
            # revealed length before any of them moves on: measured on a
            # 3-dataset parallel schedule, the first three entries all
            # reveal 7 rows. Sampling "the first two timed fits" therefore
            # drew both points at the SAME length, the slope divided by a
            # zero row difference and was clamped to 0, and the estimator
            # silently collapsed back into the constant-per-fit projection
            # it exists to replace -- with nothing failing, because a
            # factor-of-ten tolerance covers the difference on small data.
            if not warned and len(timings) >= 2:
                pooled = {rows: float(np.median(times))
                          for rows, times in timings.items()}
                remaining = [len(rows) for _, rows in todo[n_done + 1:]]
                projected, per_row, setup, lengths = project_schedule_cost(
                    pooled, remaining)
                total = spent + projected
                timed = sum(len(times) for times in timings.values())
                # Recorded so a test can show the SLOPE path ran, rather
                # than only that some number came out the other end.
                self.projection = {
                    'lengths': lengths, 'per_row': per_row, 'setup': setup,
                    'spent': spent, 'remaining': projected, 'total': total,
                    'timed_fits': timed,
                }
                if total > slow_warning_seconds:
                    warnings.warn(
                        f"predict= over this animation needs {len(todo)} "
                        f"forecast fits (one per distinct revealed history "
                        f"length), projected at roughly {total:.1f} s in "
                        f"total before the first frame can be drawn: "
                        f"{spent:.1f} s already spent on {timed} timed fits "
                        f"and about {projected:.1f} s still to come. That "
                        f"projection is extrapolated from fits timed at two "
                        f"history lengths ({lengths[0]} and {lengths[1]} "
                        f"rows), so treat it as an order of magnitude rather "
                        f"than a countdown. Every fit is kept: sampling the "
                        f"reveal instead would change what is plotted. To "
                        f"speed it up, shorten the series or lower "
                        f"frame_rate/duration; to silence this notice, pass "
                        f"slow_warning_seconds=None.",
                        stacklevel=3)
                warned = True

    # -- construction ------------------------------------------------------
    @classmethod
    def for_parallel(cls, histories, grid_lengths, model, t, n_frames,
                     min_history=DEFAULT_MIN_HISTORY,
                     slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS):
        """Schedule for a parallel/`'window'` animation.

        Every dataset advances together, so each one's revealed row count
        comes straight from `revealed_raw_counts` -- i.e. from the library's
        own `trails.anim_window_bounds`, not a second copy of the reveal
        arithmetic.
        """
        counts = [[revealed_raw_counts(len(h), g, f, n_frames)
                   for h, g in zip(histories, grid_lengths)]
                  for f in range(n_frames)]
        return cls(histories, counts=counts, model=model, t=t,
                   min_history=min_history,
                   slow_warning_seconds=slow_warning_seconds)

    @classmethod
    def for_serial(cls, histories, grid_lengths, model, t, n_frames,
                   min_history=DEFAULT_MIN_HISTORY,
                   slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS):
        """Serial reveals one dataset at a time, so its schedule comes from
        the backend's own `serial_reveal_counts` (animation-core Task 7),
        mapped from frame-grid rows onto raw rows dataset by dataset."""
        from .matplotlib_backend import serial_reveal_counts
        counts = []
        for f in range(n_frames):
            grid_counts = serial_reveal_counts(list(grid_lengths), f, n_frames)
            row = []
            for h, g, shown in zip(histories, grid_lengths, grid_counts):
                n_raw = len(h)
                if g < 2 or n_raw < 2 or shown <= 0:
                    row.append(min(n_raw, max(0, shown)))
                else:
                    pos = (min(shown, g) - 1) * (n_raw - 1) / (g - 1)
                    row.append(min(n_raw, int(np.floor(pos)) + 1))
            counts.append(row)
        return cls(histories, counts=counts, model=model, t=t,
                   min_history=min_history,
                   slow_warning_seconds=slow_warning_seconds)

    @classmethod
    def for_regrouped(cls, histories, reveal, model, t, n_frames,
                      min_history=DEFAULT_MIN_HISTORY,
                      slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS):
        """Schedule for an animation whose data `hue=`/`cluster=` regrouped.

        The revealed rows come from a `DatasetRevealSchedule` rather than from
        the drawn traces, because a dataset may now be spread over several of
        them. They are passed through as ROW TUPLES, not summarised to counts:
        the reveal is what defines a frame's history, and the cache key must
        say so.
        """
        rows = [[reveal.visible_rows(i, f) for i in range(len(histories))]
                for f in range(n_frames)]
        return cls(histories, rows=rows, model=model, t=t,
                   min_history=min_history,
                   slow_warning_seconds=slow_warning_seconds)

    # -- lookups -----------------------------------------------------------
    def revealed_rows(self, dataset, frame):
        """The ORIGINAL row indices `dataset` has revealed at `frame`.

        Frames past the end clamp to the last scheduled frame, so a backend
        that renders one frame beyond `n_frames` (matplotlib can, on a loop
        or a save) gets the final forecast rather than an IndexError.
        """
        return self.rows[min(frame, self.n_frames - 1)][dataset]

    def revealed(self, dataset, frame):
        """How many raw analyze-space rows `dataset` has revealed at `frame`."""
        return len(self.revealed_rows(dataset, frame))

    def anchor(self, dataset, frame):
        """The last revealed observation, in this schedule's coordinates."""
        rows = self.revealed_rows(dataset, frame)
        if not rows:
            return None
        return self.histories[dataset][rows[-1]]

    def path(self, dataset, frame):
        """Displacement path (t + 1, d) for `dataset` at `frame`, or None."""
        return self._paths[(dataset, self.revealed_rows(dataset, frame))]

    def polyline(self, dataset, frame):
        """The DRAWN forecast: anchor + displacement, or None."""
        path = self.path(dataset, frame)
        if path is None:
            return None
        return self.anchor(dataset, frame) + path

    def stacked_paths(self):
        """Every forecast vertex this schedule will ever draw, stacked.

        This is what Task 4 folds into the joint centre/scale statistics so
        the display box contains all of it by construction.
        """
        rows = []
        for (i, r), path in self._paths.items():
            if path is None or not r:
                continue
            rows.append(self.histories[i][r[-1]] + path)
        if not rows:
            return np.zeros((0, self.histories[0].shape[1]))
        return np.vstack(rows)

    def pin_ramp(self, column, steps):
        """Replace `column` of every path with an exact arithmetic ramp.

        `ndims=1` series mode (GH #285) draws each column against its row
        INDEX, which it materialises as an extra x column before forecasting
        -- so the model also "forecasts" a perfectly regular ramp. Its x
        values are an extrapolation the PLOT is entitled to decide (the
        index continues by one observation per step, exactly), not something
        to inherit from a fit, so they are pinned here: displacement row `k`
        of dataset `i` gets ``k * steps[i]``, leaving the forecast VALUE
        columns untouched.

        `steps` is one x-step per dataset, in the schedule's own dataset
        order. Mutates in place and returns `self` (schedules are built and
        immediately handed on, never shared).
        """
        for (i, _rows), path in self._paths.items():
            if path is None:
                continue
            path[:, column] = np.arange(len(path), dtype=float) * steps[i]
        return self

    def to_display(self, transform):
        """A copy of this schedule with every history mapped through
        `transform`, so `polyline()` returns display-box coordinates."""
        out = object.__new__(ForecastSchedule)
        out.histories = [transform(h) for h in self.histories]
        out.rows = self.rows
        out.counts = self.counts
        out.model, out.t = self.model, self.t
        out.min_history = self.min_history
        out.transform = transform
        out.n_frames, out.n_datasets = self.n_frames, self.n_datasets
        out.n_fits = 0            # no refitting: displacements are affine-mapped
        # a displacement is a DIFFERENCE of positions, so the mean cancels and
        # only the scale survives: d_display = 2 * d_analyze / scale
        out._paths = {key: (None if p is None else 2.0 * p / transform.scale)
                      for key, p in self._paths.items()}
        return out


class MultiModelSchedule:
    """Several `ForecastSchedule`s -- one per model -- behind ONE flat index.

    `predict=['Kalman', 'ARIMA', 'GP']` (GH #285) draws one overlay per
    model per dataset. Everything downstream of the schedule (the live/trail
    artists, `_update_forecasts`, the plotly frame builder) is written
    against a single flat "forecast index", so rather than teach each of
    them about models, the per-model schedules are addressed here through
    the SAME index the flat `raw_forecasts` list uses: MODEL-MAJOR, i.e.
    ``index = model_number * n_datasets + dataset_number``, matching
    ``hyp.predict``'s ``{name: [one forecast per dataset]}`` return order.

    Only the methods the drawing code actually calls are forwarded; each is
    the corresponding `ForecastSchedule` method on the sub-schedule that
    owns the index.
    """

    def __init__(self, schedules, names=None):
        if not schedules:
            raise ValueError(
                'MultiModelSchedule needs at least one ForecastSchedule.')
        self.schedules = list(schedules)
        self.names = list(names) if names is not None else None
        self.per_model = self.schedules[0].n_datasets
        self.n_datasets = self.per_model * len(self.schedules)
        self.n_frames = self.schedules[0].n_frames
        self.t = self.schedules[0].t
        self.n_fits = sum(s.n_fits for s in self.schedules)

    def _locate(self, index):
        """(sub-schedule, dataset index within it) for a flat index."""
        return (self.schedules[index // self.per_model],
                index % self.per_model)

    def revealed_rows(self, dataset, frame):
        """The ORIGINAL row indices this model/dataset slot has revealed
        at `frame` (see `ForecastSchedule.revealed_rows`)."""
        sched, i = self._locate(dataset)
        return sched.revealed_rows(i, frame)

    def revealed(self, dataset, frame):
        """How many raw rows this model/dataset slot has revealed at
        `frame` (see `ForecastSchedule.revealed`)."""
        sched, i = self._locate(dataset)
        return sched.revealed(i, frame)

    def anchor(self, dataset, frame):
        """The last revealed observation this slot's forecast starts from
        (see `ForecastSchedule.anchor`)."""
        sched, i = self._locate(dataset)
        return sched.anchor(i, frame)

    def path(self, dataset, frame):
        """This slot's displacement path at `frame`, or None (see
        `ForecastSchedule.path`)."""
        sched, i = self._locate(dataset)
        return sched.path(i, frame)

    def polyline(self, dataset, frame):
        """The DRAWN forecast for this slot at `frame`, or None (see
        `ForecastSchedule.polyline`)."""
        sched, i = self._locate(dataset)
        return sched.polyline(i, frame)

    def stacked_paths(self):
        """Every vertex every model will ever draw (see
        `ForecastSchedule.stacked_paths`)."""
        rows = [s.stacked_paths() for s in self.schedules]
        rows = [r for r in rows if len(r)]
        if not rows:
            return self.schedules[0].stacked_paths()
        return np.vstack(rows)

    def pin_ramp(self, column, steps):
        """Pin `column` of every model's paths to an exact ramp (see
        `ForecastSchedule.pin_ramp`); every model shares one x axis."""
        for sched in self.schedules:
            sched.pin_ramp(column, steps)
        return self

    def to_display(self, transform):
        """A copy with every model's schedule mapped through `transform`
        (see `ForecastSchedule.to_display`)."""
        return MultiModelSchedule(
            [s.to_display(transform) for s in self.schedules], self.names)


#: Past forecasts retained by `forecast_trail=True`.
DEFAULT_FORECAST_TRAIL = 16


def trail_frames(frame, n_retained, stride=1):
    """Frames whose forecasts are retained at `frame`, NEWEST FIRST.

    Takes no `n_frames`: the fan is bounded below by 0 and above by `frame`
    itself, so the animation's length never enters.

    Pure -- the fan at frame N depends only on N. There is deliberately no
    accumulating buffer: `FuncAnimation` replays from frame 0 for
    ``save()``/``to_jshtml()`` and may deliver frames out of order, and a
    stateful fan would make a saved GIF differ from an interactively-played
    animation.

    Parameters
    ----------
    frame : int
        The frame being drawn.
    n_retained : int
        How many past forecasts to keep.
    stride : int, default 1
        Frames between retained forecasts. ``1`` keeps every frame.

    Returns
    -------
    list of int
        Retained frame indices, newest first; empty at frame 0.
    """
    out = []
    for age in range(1, int(n_retained) + 1):
        past = frame - age * int(stride)
        if past < 0:
            break
        out.append(past)
    return out


def trail_alpha(age, n_retained, live_alpha=None, floor=None):
    """Alpha for a forecast `age` frames old. Age 0 is the live forecast.

    Parameters
    ----------
    age : int
        Frames since the forecast was live. ``0`` IS the live forecast.
    n_retained : int
        Depth of the fan (`forecast_trail=`).
    live_alpha : float, optional
        Alpha of the LIVE forecast this fan decays from -- i.e.
        ``forecast_alpha(observed_alpha)``, NOT the observed alpha itself.
        Defaults to `FORECAST_ALPHA_SCALE` (the live alpha of a fully-opaque
        dataset), so a paused animation of default-styled data looks exactly
        like the static plot.
    floor : float, optional
        Absolute alpha the fan decays down to. Defaults to
        ``TRAIL_FLOOR_FRACTION * live_alpha`` -- RELATIVE to the live alpha,
        so a trail can never come out more opaque than the live forecast it
        decays from (which an absolute floor would do on a faint dataset).

    Returns
    -------
    float

    Notes
    -----
    The result never reaches 0 -- an unwritten artist is hidden with EMPTY
    data instead, because alpha cannot express "nothing here" and a floor of
    0 would make a stale artist and an empty one indistinguishable.
    """
    if live_alpha is None:
        live_alpha = FORECAST_ALPHA_SCALE
    live_alpha = float(live_alpha)
    if floor is None:
        floor = TRAIL_FLOOR_FRACTION * live_alpha
    floor = min(float(floor), live_alpha)
    if age <= 0:
        return live_alpha
    decay = 1.0 - (age / max(1, int(n_retained) + 1))
    return max(floor, floor + (live_alpha - floor) * decay)


# ---------------------------------------------------------------------------
# forecast_*= overrides
#
# The DEFAULT is inheritance: a forecast reads as its observed trace projected
# forward (`FORECAST_ALPHA_SCALE` above), and that is unchanged. These kwargs
# let the forecasts carry their OWN grouping, palette and line style, so a
# figure can say two different things at once -- what the observed series ARE,
# and what the predictions DO.
#
# Resolved HERE, backend-neutrally, into one plain dict per dataset, because
# resolving it twice is how the two backends drift apart. Each backend's
# `_forecast_style_from` then translates the same dict into its own colour /
# dash / opacity vocabulary.
# ---------------------------------------------------------------------------

#: Appended to every "wrong number of values" message below. `plot()` fits one
#: forecast per DRAWN TRACE, which is one per input dataset until a
#: hierarchical `x=` regroups the data -- then the leaves AND the derived
#: means each get one (Task 8 lifted the refusal that used to make that
#: combination impossible). The messages said "dataset(s)" throughout, so a
#: user counting their three sectors read "got 3 value(s) for 4 dataset(s)"
#: and had no way to learn where the fourth came from; `fmt=`'s own mismatch
#: message (plot.py) already spells this out and these now match it.
_FORECAST_COUNT_NOTE = (
    "There is one forecast per DRAWN TRACE, which is one per input dataset "
    "unless a hierarchical (MultiIndex) x= regrouped the data -- then it is "
    "one per leaf group PLUS one per derived mean.")


def _forecast_label_colors(labels, palette):
    """One RGB tuple per dataset, from `labels` (one per DATASET).

    Categorical throughout: two datasets share a colour exactly when they
    share a label. `mat2colors`' 1-D numeric path is deliberately NOT used --
    it BINS values into a gradient, so integer cluster labels 0/1/2 would come
    out as three shades of one hue rather than three distinguishable colours.

    ``None`` marks an UNLABELED dataset (see `_is_missing_label`): it forms
    one group drawn in the neutral `colors.NAN_COLOR` gray and consumes no
    palette slot, so the named categories keep the first slots -- the same
    rule `plot()` applies to a partially-labeled `hue=` (release-1.0 audit,
    F02-013), so a missing forecast label means what a missing observed one
    does.
    """
    import seaborn as sns
    from .colors import _get_palette, NAN_COLOR

    ordered = []
    for lab in labels:
        if lab is not None and lab not in ordered:
            ordered.append(lab)
    colors = _get_palette(palette, max(1, len(ordered)), sns)
    index = {lab: i for i, lab in enumerate(ordered)}
    return [NAN_COLOR if lab is None else tuple(colors[index[lab]])[:3]
            for lab in labels]


def _validate_fmt(fmt, kwarg='forecast_fmt'):
    """Reject a format string matplotlib's own `fmt=` grammar cannot parse.

    Checked HERE so both backends refuse the same strings at the same
    moment: left to drawing time, matplotlib raises from inside
    `_apply_forecast_override` and plotly's `_resolve_fmt` quietly ignores
    the characters it does not recognise.
    """
    if _process_plot_format is None:  # pragma: no cover
        return
    try:
        _process_plot_format(fmt)
    except ValueError as exc:
        raise ValueError(
            f"{kwarg}={fmt!r} is not a format string matplotlib can parse "
            f"(it takes the same grammar as fmt=): {exc}") from exc


def _resolve_fmt_list(fmt, n_datasets):
    """`forecast_fmt=` -> one validated format string per dataset."""
    # `list(b'--')` is `[45, 45]` -- two ints, silently taken as one format
    # per dataset. Decoding is the only reading of a bytes fmt that is not
    # nonsense.
    if isinstance(fmt, bytes):
        fmt = fmt.decode('utf-8')
    if isinstance(fmt, str):
        fmts = [fmt] * n_datasets
    else:
        if not isinstance(fmt, (list, tuple, np.ndarray)):
            raise TypeError(
                f"forecast_fmt= takes one format string, or a list/tuple of "
                f"them (one per dataset); got "
                f"{type(fmt).__name__} ({fmt!r}).")
        fmts = [f.decode('utf-8') if isinstance(f, bytes) else f for f in fmt]
        bad = [f for f in fmts if not isinstance(f, str)]
        if bad:
            raise TypeError(
                f"every forecast_fmt= entry must be a format string; got "
                f"{bad[0]!r} ({type(bad[0]).__name__}).")
        if len(fmts) != n_datasets:
            raise ValueError(
                f"forecast_fmt= must be one format string, or one per "
                f"FORECAST; got {len(fmts)} for {n_datasets} forecast(s). "
                f"{_FORECAST_COUNT_NOTE}")
    for f in fmts:
        _validate_fmt(f)
    return fmts


def _forecast_endpoints(forecasts):
    """The last point of every forecast, stacked -- what `forecast_cluster=`
    actually groups.

    Every failure here is reported against the FORECASTS, because the
    library's own messages would not be: an empty forecast raises IndexError
    from numpy, ragged ones raise `vstack`'s "all the input array dimensions
    except for the concatenation axis must match exactly", and a non-finite
    endpoint raises sklearn's "Input contains NaN" -- none of which mentions
    a forecast.

    Every forecast must be exactly 2-D, `(steps, dimensions)` -- the shape
    `plot()` always produces, one-feature input included. `np.atleast_2d`
    would ACCEPT a raw `(t,)` array by reading it as `(1, t)`, making the
    whole trajectory one t-dimensional "endpoint" and clustering a geometry
    the overlay never draws. Silently reinterpreting a shape is the one
    failure mode this function exists to prevent.
    """
    if forecasts is None:
        raise ValueError(
            "forecast_cluster= groups the forecasts by WHERE THEY END UP, so "
            "it needs the forecasts themselves; none were given.")
    endpoints = []
    for i, fc in enumerate(forecasts):
        try:
            arr = np.asarray(fc, dtype=float)
        except ValueError as exc:
            # a ragged nested list raises numpy's "setting an array element
            # with a sequence ... inhomogeneous shape", which names neither
            # the kwarg nor which forecast was malformed
            raise ValueError(
                f"forecast_cluster= could not read forecast {i} as an array "
                f"of numbers: {exc}") from exc
        if arr.ndim != 2:
            raise ValueError(
                f"forecast_cluster= needs each forecast as (steps, "
                f"dimensions); forecast {i} has shape {arr.shape}."
                + (f" For a one-dimensional forecast, pass it as "
                   f"({arr.shape[0]}, 1)." if arr.ndim == 1 else ""))
        if arr.shape[0] == 0:
            raise ValueError(
                f"forecast_cluster= groups each forecast by its last point, "
                f"but forecast {i} has no rows.")
        endpoints.append(arr[-1])
    widths = sorted({len(e) for e in endpoints})
    if len(widths) > 1:
        raise ValueError(
            f"forecast_cluster= clusters every forecast's endpoint together, "
            f"so they must all have the same number of dimensions; got "
            f"{widths}.")
    stacked = np.vstack(endpoints)
    bad = [i for i in range(len(stacked)) if not np.isfinite(stacked[i]).all()]
    if bad:
        raise ValueError(
            f"forecast_cluster= cannot group non-finite (NaN/inf) endpoints; "
            f"forecast(s) {bad} end at one.")
    return stacked


def resolve_forecast_overrides(n_datasets, forecasts=None, *, hue=None,
                               cluster=None, n_clusters=None, palette=None,
                               fmt=None, stacklevel=2):
    """Resolve `forecast_hue=`/`forecast_cluster=`/`forecast_palette=`/
    `forecast_fmt=` into one override dict per dataset.

    Parameters
    ----------
    n_datasets : int
        How many forecasts there are -- one per DRAWN TRACE. That is one per
        INPUT dataset for flat input, whatever the observed data was
        regrouped into by `hue=`/`cluster=`; for a hierarchical (MultiIndex)
        `x=` it is one per leaf group PLUS one per derived mean, since
        `plot()` forecasts every final trace.
    forecasts : list of numpy.ndarray or None
        The forecast paths, in the space they are DRAWN in. Only
        `forecast_cluster=` needs them: it clusters their ENDPOINTS, and
        clustering the raw model space would group by a geometry the user
        cannot see when `reduce=`/`align=` changed it.

        With `cluster=`, exactly `n_datasets` of them, each 2-D
        ``(steps, dimensions)`` -- what `plot()` always passes. Both are
        checked rather than assumed: this resolver is importable on its own,
        and both mismatches otherwise fail silently or misleadingly (a raw
        ``(t,)`` array would be read as one t-dimensional endpoint).
    hue : sequence or None
        One value per dataset. Datasets sharing a value share a colour. A
        missing value (`None`, NaN, `pd.NA`) marks that dataset UNLABELED:
        every one of them forms a single group drawn in neutral gray and
        consuming no palette slot, exactly as `plot()` treats a
        partially-labeled `hue=`. A bare string is rejected rather than read
        as one label per character, and each value must be hashable -- it
        becomes a key in the label -> colour map.
    cluster : cluster spec or None
        Clusters the forecast ENDPOINTS -- where each series is predicted to
        end up -- so a forecast's colour answers "which of these are heading
        to the same place?".

        It deliberately does NOT recluster the observed data: inheriting the
        observed assignment is what the default already gives, so that
        reading would make the kwarg a no-op. Nor does it cluster every
        predicted POINT (one forecast would change colour along its own short
        path, contradicting "coloured by where it is heading"), nor whole
        flattened trajectories (sensitive to `t`, to sampling and to
        dimensionality, where an endpoint has one stable meaning).

        `plot()` calls this ONCE per figure, animated or not, with the
        FULL-HISTORY forecasts -- so an animation's live and trailing
        forecasts all carry the grouping of where each series ends up, held
        fixed for every frame rather than recomputed as the reveal
        progresses. See the `forecast_cluster=` docs for why.
    n_clusters : int or None
        Passed to the clusterer. Separate from `plot`'s `n_clusters=` on
        purpose: the observed data and the forecasts are different point
        sets, and there is no reason a good number of groups for one is a
        good number for the other.
    palette : str, list of colors, matplotlib Colormap, or None
        Colours for the grouping. With no grouping given, spent one colour
        per dataset. `None` means "no colour override" -- callers pass the
        figure's own `palette=` when they want the observed one inherited.
    fmt : str, sequence of str, or None
        Line/marker style for the forecasts, independent of the observed
        `fmt`. Validated here with matplotlib's own `fmt=` parser, so both
        backends refuse the same strings at the same moment.

    Returns
    -------
    list of dict
        One per dataset, each with optional ``'color'`` and ``'fmt'`` keys.
        A MISSING key means inherit that aspect from the observed trace --
        which is why these are sparse dicts rather than complete styles:
        only the aspects actually named are overridden.
    """
    overrides = [{} for _ in range(n_datasets)]

    if hue is not None and cluster is not None:
        raise ValueError(
            "forecast_hue= and forecast_cluster= both decide how the "
            "forecasts are grouped and coloured, so passing both would mean "
            "silently picking a winner. Pass one (this mirrors hue= and "
            "cluster= for the observed data).")
    if n_clusters is not None and cluster is None:
        warnings.warn(
            "forecast_n_clusters= has no effect without forecast_cluster=; "
            "it sets the number of groups the FORECAST endpoints are "
            "clustered into. (The observed data's cluster count is "
            "n_clusters=.)", stacklevel=stacklevel)

    labels = None
    if hue is not None:
        # a bare string is a sequence of CHARACTERS, so `list('ab')` is a
        # perfectly-shaped pair of labels for a two-dataset plot -- silently
        # right by accident and, at any other dataset count, wrong with a
        # length message that reads as nonsense.
        if isinstance(hue, (str, bytes)):
            raise TypeError(
                f"forecast_hue= takes one value per dataset, and a bare "
                f"string would be read as one label per CHARACTER; got "
                f"{hue!r}. Pass a sequence, e.g. [{hue!r}] for a single "
                f"dataset.")
        labels = list(hue)
        # a per-observation hue (the shape `hue=` takes) reaches here as a
        # list of ARRAYS, and grouping by `==` on those raises numpy's
        # "truth value of an array is ambiguous" from deep inside the colour
        # code. Say what was actually wrong instead.
        if any(isinstance(v, (list, tuple, np.ndarray)) for v in labels):
            raise ValueError(
                "forecast_hue= takes one SCALAR value per dataset (a "
                "forecast is a single trace, so there is no per-observation "
                "hue to take); got a sequence as one of its values. Pass "
                "hue= to colour the observed data per observation.")
        if len(labels) != n_datasets:
            raise ValueError(
                f"forecast_hue= must have exactly one value per FORECAST (a "
                f"forecast is one trace, not one value per observation); got "
                f"{len(labels)} value(s) for {n_datasets} forecast(s). "
                f"{_FORECAST_COUNT_NOTE}")
        # every "no label here" spelling becomes ONE sentinel, so missing
        # labels form a single unlabeled group rather than one group per
        # distinct NaN object (`colors.is_missing_label` -- the same
        # normalization `plot()` applies to a categorical `hue=`)
        from .colors import is_missing_label
        labels = [None if is_missing_label(v) else v for v in labels]
        # labels become the KEYS of a label -> colour map, so an unhashable
        # one (a dict, a set) passes the sequence guard above and then fails
        # as a bare "unhashable type" from inside the colour code
        # `hash(v)` rather than `isinstance(v, Hashable)`: the ABC only asks
        # whether `__hash__` EXISTS, and a tuple holding a list has one that
        # raises when called.
        for v in labels:
            if v is None:
                continue
            try:
                hash(v)
            except TypeError as exc:
                raise TypeError(
                    f"every forecast_hue= value must be usable as a group "
                    f"label, which means hashable (datasets sharing a value "
                    f"share a colour); got {v!r} "
                    f"({type(v).__name__}).") from exc
    elif cluster is not None:
        # `plot()` passes `len(raw_forecasts)` as `n_datasets`, so it cannot
        # disagree with itself here -- but a direct caller can, and the
        # consequences are silent rather than loud: too many forecasts write
        # past the end of `overrides` with a bare IndexError, and too few
        # style every dataset from labels that clustered a DIFFERENT set of
        # endpoints, reported (below) with a point count taken from
        # `n_datasets` rather than from the forecasts actually stacked.
        if forecasts is not None and len(forecasts) != n_datasets:
            raise ValueError(
                f"forecast_cluster= needs exactly one forecast per dataset "
                f"(it groups each dataset by where ITS forecast ends up); got "
                f"{len(forecasts)} forecast(s) for {n_datasets} dataset(s).")
        if n_datasets < 2:
            warnings.warn(
                f"forecast_cluster= needs at least two forecasts to group "
                f"({n_datasets} given): every partition of a single point is "
                f"the same partition. Falling back to the observed trace's "
                f"style.", stacklevel=stacklevel)
        else:
            from ..cluster.cluster import cluster as _clusterer
            endpoints = _forecast_endpoints(forecasts)
            try:
                labels = [int(v) for v in np.asarray(
                    _clusterer(endpoints, cluster=cluster,
                               n_clusters=n_clusters)).ravel()]
            except ValueError as exc:
                # The clusterer's own validation decides what is legal --
                # inventing a forecast-specific rule here would let the two
                # disagree. Only the MESSAGE is amended: sklearn says
                # "n_samples=3 should be >= n_clusters=5", and nothing in
                # that tells a user that `n_samples` is their dataset count.
                raise ValueError(
                    f"forecast_cluster= clusters one endpoint per dataset, so "
                    f"it has {n_datasets} point(s) to work with"
                    f"{f' and forecast_n_clusters={n_clusters}' if n_clusters is not None else ''}"
                    f". The clusterer rejected that: {exc}") from exc

    if labels is not None:
        for i, color in enumerate(_forecast_label_colors(
                labels, 'hls' if palette is None else palette)):
            overrides[i]['color'] = color
    elif palette is not None:
        # nothing to group BY, so the palette is spent one colour per
        # dataset -- the only grouping a forecast set has on its own
        for i, color in enumerate(_forecast_label_colors(
                list(range(n_datasets)), palette)):
            overrides[i]['color'] = color

    if fmt is not None:
        for i, f in enumerate(_resolve_fmt_list(fmt, n_datasets)):
            overrides[i]['fmt'] = f

    return overrides
