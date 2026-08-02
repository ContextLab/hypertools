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

import numpy as np

from ..predict.predict import predict as _predict

#: Fewest observations we will fit a forecaster to.
DEFAULT_MIN_HISTORY = 2


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
    is reused here rather than re-derived (`FrameContext.revealed_counts` is
    documented ``None`` for parallel animations, so it cannot serve). `end`
    does not depend on the trail window, so 0 is passed for it.

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

    def __init__(self, histories, counts, model, t,
                 min_history=DEFAULT_MIN_HISTORY, transform=None):
        self.histories = [np.asarray(h, dtype=float) for h in histories]
        self.counts = [list(row) for row in counts]
        self.model = model
        self.t = int(t)
        self.min_history = int(min_history)
        self.transform = transform
        self.n_frames = len(self.counts)
        self.n_datasets = len(self.histories)
        self.n_fits = 0
        self._paths = {}
        for frame_counts in self.counts:
            for i, k in enumerate(frame_counts):
                if (i, k) in self._paths:
                    continue
                hist = self.histories[i][:k]
                path = forecast_from_history(hist, self.model, self.t,
                                             min_history=self.min_history)
                if path is not None:
                    self.n_fits += 1
                self._paths[(i, k)] = path

    # -- construction ------------------------------------------------------
    @classmethod
    def for_parallel(cls, histories, grid_lengths, model, t, n_frames,
                     min_history=DEFAULT_MIN_HISTORY):
        """Schedule for a parallel/`'window'` animation.

        Every dataset advances together, so each one's revealed row count
        comes straight from `revealed_raw_counts` -- i.e. from the library's
        own `trails.anim_window_bounds`, not a second copy of the reveal
        arithmetic.
        """
        counts = [[revealed_raw_counts(len(h), g, f, n_frames)
                   for h, g in zip(histories, grid_lengths)]
                  for f in range(n_frames)]
        return cls(histories, counts, model, t, min_history=min_history)

    @classmethod
    def for_serial(cls, histories, grid_lengths, model, t, n_frames,
                   min_history=DEFAULT_MIN_HISTORY):
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
        return cls(histories, counts, model, t, min_history=min_history)

    # -- lookups -----------------------------------------------------------
    def revealed(self, dataset, frame):
        """RAW analyze-space rows `dataset` has revealed at `frame`.

        Frames past the end clamp to the last scheduled frame, so a backend
        that renders one frame beyond `n_frames` (matplotlib can, on a loop
        or a save) gets the final forecast rather than an IndexError.
        """
        return self.counts[min(frame, self.n_frames - 1)][dataset]

    def anchor(self, dataset, frame):
        """The last revealed observation, in this schedule's coordinates."""
        k = self.revealed(dataset, frame)
        if k < 1:
            return None
        return self.histories[dataset][k - 1]

    def path(self, dataset, frame):
        """Displacement path (t + 1, d) for `dataset` at `frame`, or None."""
        return self._paths[(dataset, self.revealed(dataset, frame))]

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
        for (i, k), path in self._paths.items():
            if path is None or k < 1:
                continue
            rows.append(self.histories[i][k - 1] + path)
        if not rows:
            return np.zeros((0, self.histories[0].shape[1]))
        return np.vstack(rows)

    def to_display(self, transform):
        """A copy of this schedule with every history mapped through
        `transform`, so `polyline()` returns display-box coordinates."""
        out = object.__new__(ForecastSchedule)
        out.histories = [transform(h) for h in self.histories]
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


def trail_alpha(age, n_retained, live_alpha=0.6, floor=0.08):
    """Alpha for a forecast `age` frames old. Age 0 is the live forecast.

    `live_alpha` matches the static overlay's 0.6, so a paused animation
    looks like a static plot. The result never reaches 0 -- an unwritten
    artist is hidden with EMPTY data instead, because alpha cannot express
    "nothing here" and a floor of 0 would make a stale artist and an empty
    one indistinguishable.
    """
    if age <= 0:
        return live_alpha
    decay = 1.0 - (age / max(1, int(n_retained) + 1))
    return max(floor, floor + (live_alpha - floor) * decay)
