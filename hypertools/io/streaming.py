#!/usr/bin/env python
"""Streaming-data support (issue #101).

Streams are treated as just another supported data type -- no flag needed
(they are detected from the structure of the input). The normalization and
dimensionality-reduction models are fitted on the first ``stream_init``
samples; every subsequent sample is projected through those *fitted* models
and added to the plot dynamically.

Supported stream types:
    - any Python iterator/generator yielding samples
    - Hugging Face ``datasets.IterableDataset`` (``load_dataset(...,
      streaming=True)``, https://huggingface.co/docs/datasets/en/stream)

Each sample may be a numeric vector (list/tuple/array/Series) or a dict of
features (as yielded by Hugging Face streams); numeric fields are extracted
in insertion order and concatenated, and non-numeric fields are ignored.
"""

import collections.abc
import itertools
import os
import warnings

import numpy as np
import pandas as pd


def _validate_stream_save_path(save_path):
    """Validate a streaming ``save_path`` BEFORE any samples are consumed,
    returning ``(path, ext)``.

    Streams are rendered frame-by-frame as they arrive, so only
    frame-grabbing writers work: Pillow (.gif, .png/.apng) and -- with
    FFmpeg installed -- the video containers ffmpeg can mux into. Unknown
    extensions used to fall through to the Pillow writer and die at
    finalize time with PIL's raw 'unknown file extension' error, AFTER the
    whole stream had been consumed (release-1.0 audit,
    D09-tutorials-applied-009). For the same reason, a video extension is
    checked for FFmpeg availability here too: matplotlib only looks for
    the ffmpeg binary when the writer is constructed, which happens after
    the stream head has been consumed (release-1.0 audit: re-review of
    D09-tutorials-applied-009).
    """
    from ..plot.animate import _FFMPEG_EXTENSIONS
    path = os.fspath(save_path)
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext not in ('gif', 'png', 'apng') + tuple(_FFMPEG_EXTENSIONS):
        what = f"extension {'.' + ext!r}" if ext else 'missing extension'
        raise ValueError(
            f'unsupported streaming save format ({what}) for {path!r}; '
            'streamed animations support .gif, .png/.apng (animated PNG), '
            'and -- with FFmpeg installed -- '
            + '/'.join('.' + e for e in _FFMPEG_EXTENSIONS)
            + '. (.svg export is only available for non-streaming '
            'animate= plots.)')
    if ext in _FFMPEG_EXTENSIONS:
        from matplotlib import animation
        if not animation.writers.is_available('ffmpeg'):
            raise RuntimeError(
                f'save_path={path!r} selects the .{ext} video format, '
                'which requires FFmpeg, but matplotlib could not find an '
                'ffmpeg binary (checked before consuming the stream, so '
                'no samples were pulled). Install ffmpeg (e.g. `conda '
                'install ffmpeg`, `brew install ffmpeg`, or `apt install '
                'ffmpeg`) and make sure it is on the PATH (or point '
                "matplotlib.rcParams['animation.ffmpeg_path'] at it), or "
                'save to .gif/.png/.apng, which use Pillow and need no '
                'FFmpeg.')
    return path, ext



def is_stream(x):
    """True when x is streaming data: a Python iterator/generator, or a
    Hugging Face ``datasets.IterableDataset``. Materialized containers
    (lists, tuples, arrays, DataFrames) and strings are not streams."""
    if isinstance(x, (list, tuple, str, np.ndarray, pd.DataFrame, pd.Series,
                      dict)):
        return False
    # generators and other iterators
    if isinstance(x, collections.abc.Iterator):
        return True
    # Hugging Face IterableDataset (duck-typed so `datasets` is not a
    # required dependency)
    for klass in type(x).__mro__:
        if klass.__name__ == 'IterableDataset' and \
                klass.__module__.startswith('datasets'):
            return True
    return False


def row_to_vector(row):
    """Convert one stream sample to a 1d float vector.

    - numeric vectors (list/tuple/ndarray/Series) pass through
    - dicts (e.g. Hugging Face rows): numeric scalars and flat numeric
      lists/arrays are concatenated in insertion order; strings and other
      non-numeric fields are ignored
    """
    if isinstance(row, dict):
        parts = [p for p in (_numeric_part(v) for v in row.values())
                 if p is not None]
        if not parts:
            raise ValueError(
                'could not extract numeric features from stream sample with '
                f'keys {list(row.keys())}; streaming samples must contain '
                'numeric fields')
        return np.concatenate(parts)
    if isinstance(row, pd.Series):
        row = row.values
    vec = np.asarray(row, dtype=np.float64).ravel()
    if vec.size == 0:
        raise ValueError('empty stream sample')
    return vec


def _numeric_part(val):
    """1d float array from a numeric scalar / flat numeric sequence, else
    None."""
    if isinstance(val, bool) or val is None or isinstance(val, str):
        return None
    if np.isscalar(val):
        try:
            return np.asarray([val], dtype=np.float64)
        except (TypeError, ValueError):
            return None
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(val, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None
        return arr if arr.size else None
    return None


def _safe_std(m, **kwargs):
    sd = m.std(**kwargs)
    return np.where(sd == 0, 1.0, sd)


def _fit_stream_models(head, reduce, ndims, normalize):
    """Fit normalization stats and the reduction model on the head samples.

    Returns (head_reduced, project, model): project(chunk_matrix) applies
    the fitted normalization + reduction to new samples; model is the
    fitted reduction estimator (None when no reduction was needed).
    """
    from ..reduce.reduce import _resolve_model

    # normalization stats are computed ONCE, on the head, and reused for
    # every future sample (the fitted-transform semantics of issue #101).
    # A single stream is one dataset, so 'across'/'within' coincide;
    # 'row' is stateless.
    if normalize in (None, False):
        def norm(m):
            """No-op normalizer: return `m` unchanged."""
            return m
    elif normalize == 'row':
        def norm(m):
            """Z-score each row of `m` independently (stateless, per-chunk)."""
            return (m - m.mean(axis=1, keepdims=True)) / \
                _safe_std(m, axis=1, keepdims=True)
    elif normalize in ('across', 'within', True):
        mu = head.mean(axis=0, keepdims=True)
        sd = _safe_std(head, axis=0, keepdims=True)

        def norm(m):
            """Z-score `m` using the mean/std fitted on the head samples."""
            return (m - mu) / sd
    else:
        raise ValueError(f'unsupported normalize option for streaming data: '
                         f'{normalize!r}')
    head_n = norm(head)

    # reduction spec: name / dict / class / instance, mirroring tools.reduce
    if isinstance(reduce, dict):
        model_spec = reduce.get('model')
        # accept the canonical 'kwargs' key (falling back to the legacy 'params')
        # so a streaming reduce spec honors constructor kwargs like every other
        # dispatcher (QC 2026-07: only 'params' was read, so
        # reduce={'model':'PCA','kwargs':{'whiten':True}} silently used defaults).
        params = dict(reduce.get('kwargs', reduce.get('params', {})))
    else:
        model_spec = reduce
        params = {}
    params.setdefault('n_components', ndims)

    if model_spec is None or head_n.shape[1] <= params['n_components']:
        return head_n, norm, None

    if isinstance(model_spec, str):
        model = _resolve_model(model_spec)(**params)
    elif isinstance(model_spec, type):
        model = model_spec(**params)
    else:
        model = model_spec  # already-instantiated estimator

    if not hasattr(model, 'transform'):
        name = model_spec if isinstance(model_spec, str) \
            else type(model).__name__
        raise ValueError(
            f'streaming data requires a reduction model with a transform() '
            f'method so that new samples can be projected with the model '
            f'fitted on the initial samples; {name} has none. Use e.g. '
            f'IncrementalPCA, PCA, or UMAP.')

    model.fit(head_n)

    def project(m):
        """Normalize (via `norm`) and reduce `m` using the fitted stream model."""
        return model.transform(norm(m))

    return model.transform(head_n), project, model


def plot_stream(stream, fmt='-', stream_init=10000, stream_chunk=100,
                stream_max=None, stream_window=None, ndims=3,
                reduce='IncrementalPCA', normalize=None, align=None,
                cluster=None, n_clusters=None, save_path=None, show=True,
                frame_rate=30, **plot_kwargs):
    """Plot streaming data: fit models on the first ``stream_init`` samples,
    then project and draw each subsequent chunk of ``stream_chunk`` samples
    dynamically. Called by hyp.plot() when it detects a streaming input;
    not part of the public API.

    Streaming continues until the stream is exhausted, ``stream_max``
    samples have been consumed (when ``stream_max < stream_init``, the
    head itself is capped at ``stream_max``; never MORE than
    ``stream_max`` samples are pulled from the stream), the user
    interrupts (Ctrl-C), or the stream raises -- infinite streams render
    continually, and any animation being saved is finalized whenever
    streaming stops, including on interrupt and on error. ``save_path``
    supports the frame-grabbing formats: .gif and .png/.apng via Pillow,
    plus (with FFmpeg installed) the video containers .mp4/.mov/.avi/
    .m4v/.mkv; other extensions raise ``ValueError``, and a video
    extension with no FFmpeg available raises ``RuntimeError`` -- both
    before any samples are consumed. A stream error at ANY point after
    the first sample -- mid-stream OR while consuming the initial
    ``stream_init`` head samples (source disconnect, bad sample, ...) --
    does NOT discard the consumed data: a ``RuntimeWarning`` is emitted
    and the figure is returned with everything consumed so far (models
    fitted on the salvaged head when the error struck during the head
    phase), the exception stored under ``stream_info['error']`` (QC
    2026-07, F22-io-streaming-lsl-003; head-phase salvage added in the
    release-1.0 audit re-review).
    ``stream_window`` optionally limits the *display* to the most recent
    samples (comet style); all consumed data is still retained on the
    returned figure's ``stream_info``.

    The display box (axis limits and the data->box affine) is FROZEN from
    the head samples; later samples that land outside it are drawn clamped
    to the box surface, and a ``RuntimeWarning`` is emitted when a large
    fraction of streamed samples is clamped (their true projected values
    stay in ``stream_info['xform_data']``). Streamed trajectories are
    drawn as raw polylines (one vertex per sample) from the first frame
    on, without the interpolation/smoothing applied to static plots.

    Streamed samples need >= 2 features to span a trajectory: a
    single-channel stream raises ``ValueError`` (unless it ends within
    the head, in which case it renders like a static 1-D plot).

    Returns a matplotlib Figure (streaming plots are always drawn with the
    matplotlib backend -- a ``backend=`` request is ignored with a
    ``UserWarning`` by ``hyp.plot``); ``fig.stream_info`` is a dict holding
    ``'data'`` (the raw consumed samples), ``'xform_data'`` (the projected
    trajectory), ``'n_samples'``, ``'reduce_model'`` (the fitted reduction
    model, or None), ``'truncated'`` (whether streaming was stopped -- by
    ``stream_max``, an interrupt, or an error -- before the stream was
    observed to end; True even when the stream held exactly ``stream_max``
    samples, since no extra sample is ever pulled to check), and
    ``'error'`` (the exception that ended streaming early, or None).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from ..plot.plot import plot as hyp_plot

    if align not in (None, False):
        raise ValueError('align is not supported for streaming data (a '
                         'stream is a single dataset)')
    if cluster is not None or n_clusters is not None:
        raise ValueError('cluster is not yet supported for streaming data')

    # parameter validation, BEFORE any samples are consumed (QC 2026-07,
    # F22-io-streaming-lsl-006/-008: invalid values used to surface as
    # cryptic islice/unpacking errors, or silently change behavior)
    stream_init = int(stream_init)
    if stream_init < 1:
        raise ValueError(
            f'stream_init must be a positive integer; got {stream_init}')
    stream_chunk = int(stream_chunk)
    if stream_chunk < 1:
        raise ValueError(
            f'stream_chunk must be a positive integer; got {stream_chunk}')
    if stream_max is not None:
        stream_max = int(stream_max)
        if stream_max < 1:
            raise ValueError(
                f'stream_max must be a positive integer or None; got '
                f'{stream_max}')
    if stream_window is not None:
        stream_window = int(stream_window)
        if stream_window < 1:
            raise ValueError(
                f'stream_window must be a positive integer or None; got '
                f'{stream_window}')
    if ndims is not None and ndims not in (1, 2, 3):
        raise ValueError(
            f'ndims must be 1, 2, or 3 for streaming plots; got {ndims} '
            '(streamed samples are drawn in at most 3 dimensions)')
    save_ext = None
    if save_path is not None:
        # fail fast on unwritable formats, BEFORE any samples are consumed
        save_path, save_ext = _validate_stream_save_path(save_path)

    it = iter(stream)
    # the head is capped by stream_max too, so the documented sample cap
    # is never overshot (QC 2026-07, F22-io-streaming-lsl-007)
    head_take = stream_init if stream_max is None \
        else min(stream_init, stream_max)
    # consume the head one sample at a time so a stream error DURING the
    # head phase salvages the samples already received, exactly like the
    # mid-stream salvage below -- an acquisition that dies at sample 9000
    # of a 10000-sample head used to lose all data, the figure, and the
    # save file (release-1.0 audit: re-review of
    # F22-io-streaming-lsl-003)
    head_error = None
    head_rows = []
    try:
        for r in itertools.islice(it, head_take):
            head_rows.append(r)
    except BaseException as e:
        if not head_rows:
            raise  # nothing consumed yet -- nothing to salvage
        head_error = e
    if not head_rows:
        raise ValueError('stream produced no samples')
    # a malformed sample inside the head likewise salvages the good
    # prefix (matching the mid-stream bad-sample behavior)
    head_vecs = []
    try:
        for r in head_rows:
            head_vecs.append(row_to_vector(r))
    except Exception as e:
        if not head_vecs:
            raise
        if head_error is None:
            head_error = e
        head_rows = head_rows[:len(head_vecs)]
    head = np.vstack(head_vecs)

    if head_error is None and head.shape[1] < 2 \
            and list(itertools.islice(it, 1)):
        # fail fast (before the figure/writer exist) instead of crashing
        # inside the first redraw (QC 2026-07, F22-io-streaming-lsl-001);
        # a 1-channel stream that ENDED within the head falls through and
        # renders like a static 1-D plot
        raise ValueError(
            'streamed samples have a single feature/channel; streaming '
            'requires >= 2 features to draw a trajectory. Include a '
            'time/index value in each sample (e.g. yield [t, value]), or '
            'materialize the stream (e.g. np.array(list(stream))) and use '
            'a static hyp.plot, which draws 1-D data against the sample '
            'index.')

    head_red, project, model = _fit_stream_models(
        head, reduce, ndims, normalize)

    if head_red.shape[1] > 3:
        raise ValueError(
            f'streamed samples span {head_red.shape[1]} dimensions after '
            'the reduce step, but plots are at most 3-dimensional. Set '
            "reduce/ndims so samples are projected to <= 3 dimensions "
            "(e.g. reduce='IncrementalPCA', ndims=3).")

    # initial plot on the head (already normalized/reduced -> disable both)
    fig = hyp_plot(head_red, fmt, reduce=None, normalize=None, ndims=ndims,
                   show=False, **plot_kwargs)
    artist = next(ln for ln in fig.axes[0].lines if len(ln.get_data()[0]))

    # the axis limits and the data->box transform are FROZEN from the head:
    # hyp.plot centered the head (column means) and scaled it into the
    # [-1, 1] cube (global min/max); the exact same affine transform is
    # applied to every future sample so the view NEVER changes as data
    # streams in. Samples that land outside the box are clamped to the
    # closest point on its surface.
    head_mu = head_red.mean(axis=0, keepdims=True)
    head_centered = head_red - head_mu
    box_m1 = head_centered.min()
    box_m2 = (head_centered - box_m1).max() or 1.0

    def to_box(pts):
        """Map `pts` into the frozen [-1, 1] display box fit from the head samples.

        Applies the head-derived centering/scaling affine transform and
        clamps any out-of-range values to the box surface, so the plot's
        axis limits never change as new samples stream in.
        """
        t = 2.0 * ((pts - head_mu) - box_m1) / box_m2 - 1.0
        return np.clip(t, -1.0, 1.0)

    is3d = head_red.shape[1] > 2
    raw = [head]
    accum = [head_red]
    n_seen = len(head_rows)

    interactive = show and matplotlib.get_backend().lower() not in (
        'agg', 'pdf', 'svg', 'ps')
    # choose the frame-grabbing writer by extension, mirroring the
    # non-streaming animate= dispatch in plot/animate._save_animation:
    # Pillow for .gif/.png/.apng (with real-time cumulative frame timing),
    # ffmpeg for video containers (.mp4 and friends -- these used to crash
    # at finalize time with PIL's raw 'unknown file extension: .mp4' even
    # though animate= supports mp4; release-1.0 audit,
    # D09-tutorials-applied-009).
    writer = None
    writer_tmp = None  # temp .png target for .apng (renamed at finish)
    if save_path is not None:
        from ..plot.animate import _RealTimePillowWriter
        if save_ext in ('gif', 'png', 'apng'):
            writer = _RealTimePillowWriter(
                fps=frame_rate, grid_ms=10 if save_ext == 'gif' else 1)
            target = save_path
            if save_ext == 'apng':
                # Pillow only emits animated PNG for the .png extension:
                # write to a unique temp .png in the target directory and
                # rename it onto the requested name after finish() (same
                # workaround as animate._save_animation).
                import tempfile
                fd, writer_tmp = tempfile.mkstemp(
                    suffix='.png',
                    dir=os.path.dirname(os.path.abspath(save_path)))
                os.close(fd)
                target = writer_tmp
            writer.setup(fig, target, dpi=fig.dpi)
        else:
            from ..plot.animate import _ffmpeg_quality_kwargs
            writer = animation.writers['ffmpeg'](fps=frame_rate,
                                                 **_ffmpeg_quality_kwargs())
            writer.setup(fig, save_path, dpi=fig.dpi)

    def _redraw():
        # fixed head-fitted transform + clamp: the space inside the cube is
        # stable for the whole stream (no per-chunk re-scaling "twitch")
        shown = np.vstack(accum)
        if stream_window is not None:
            shown = shown[-stream_window:]
        full = to_box(shown)
        if is3d:
            artist.set_data_3d(full[:, 0], full[:, 1], full[:, 2])
        else:
            artist.set_data(full[:, 0], full[:, 1])
        if writer is not None:
            writer.grab_frame()
        if interactive:
            fig.canvas.draw_idle()
            plt.pause(0.001)

    # draw the head through the same raw-polyline path as every later
    # chunk, so the first animation frame doesn't "snap" from a smoothed
    # curve to a raw polyline on the first redraw (QC 2026-07,
    # F22-io-streaming-lsl-005); 1-D head-only streams keep hyp.plot's
    # static rendering (there is nothing to redraw)
    if head_red.shape[1] >= 2:
        _redraw()
    elif writer is not None:
        writer.grab_frame()

    def _n_clamped(pts):
        # how many of pts land outside the frozen display box (and are
        # therefore drawn clamped to its surface)
        t = 2.0 * ((pts - head_mu) - box_m1) / box_m2 - 1.0
        return int(np.any((t < -1.0) | (t > 1.0), axis=1).sum())

    # consume until the stream is exhausted, stream_max is reached, the
    # user interrupts, or the stream errors out -- an infinite stream
    # renders continually, and the animation (if any) is finalized
    # whenever streaming stops
    truncated = False
    stream_error = None
    clamped = post_head = 0
    clamp_warned = False

    def _consume(rows):
        # project + draw one (possibly partial) chunk of samples
        nonlocal n_seen, clamped, post_head, clamp_warned
        if not rows:
            return
        chunk = np.vstack([row_to_vector(r) for r in rows])
        projected = project(chunk)
        # append raw only after projection succeeds, so 'data' and
        # 'xform_data' always describe the same samples
        raw.append(chunk)
        accum.append(projected)
        n_seen += len(rows)
        clamped += _n_clamped(projected)
        post_head += len(projected)
        if not clamp_warned and post_head >= 20 \
                and clamped / post_head > 0.25:
            # the stream has drifted out of the head-fitted display box:
            # the plot is visibly distorted (QC 2026-07,
            # F22-io-streaming-lsl-002)
            clamp_warned = True
            warnings.warn(
                f'{clamped} of {post_head} streamed samples '
                f'({100.0 * clamped / post_head:.0f}%) fall outside '
                'the display box fitted on the first stream_init '
                'samples and are drawn clamped to its surface, so '
                'their displayed positions are distorted (the true '
                "projected values are kept in "
                "fig.stream_info['xform_data']). If the early "
                'samples are not representative of the whole stream, '
                'increase stream_init.', RuntimeWarning, stacklevel=2)
        _redraw()

    try:
        # `while head_error is None` (never mutated inside the loop) gates
        # the consume loop entirely when the stream already died during
        # the head phase: the iterator is never touched again, and the
        # salvaged head flows into the writer finalization / stream_info
        # assembly below
        while head_error is None:
            if stream_max is not None and n_seen >= stream_max:
                # stream_max reached: stop WITHOUT touching the stream
                # again. (An earlier version peeked one extra sample here
                # to test whether the stream really had more, silently
                # consuming -- and discarding -- a sample beyond the
                # documented cap, which matters for stateful/costly
                # sources like hardware acquisition or paid APIs;
                # release-1.0 audit, D09-tutorials-applied-006. Exactly
                # stream_max samples are now consumed, and 'truncated'
                # means "streaming was stopped by stream_max, an
                # interrupt, or an error before the stream was observed
                # to end" -- it is True even for a stream holding exactly
                # stream_max samples.)
                truncated = True
                break
            take = stream_chunk
            if stream_max is not None:
                take = min(take, stream_max - n_seen)
            rows = []
            try:
                for r in itertools.islice(it, take):
                    rows.append(r)
            except BaseException:
                # the stream died mid-chunk: salvage the rows it yielded
                # before dying, then let the outer handler finalize
                try:
                    _consume(rows)
                except Exception:
                    pass
                raise
            if not rows:
                break
            _consume(rows)
        if head_error is not None:
            truncated = True
            if not isinstance(head_error, KeyboardInterrupt):
                stream_error = head_error
                warnings.warn(
                    f'streaming stopped early (while consuming the first '
                    f'stream_init samples): {type(head_error).__name__}: '
                    f'{head_error}. Models were fitted on the {n_seen} '
                    'samples received before the error, and the figure '
                    'is returned with those samples (see '
                    'fig.stream_info; the exception is stored under '
                    "fig.stream_info['error']).",
                    RuntimeWarning, stacklevel=2)
    except KeyboardInterrupt:
        truncated = True
    except Exception as e:  # noqa: BLE001 -- deliberately broad: a source
        # error at minute 29 of a 30-minute acquisition must not destroy
        # the figure, the consumed data, and the animation (QC 2026-07,
        # F22-io-streaming-lsl-003)
        truncated = True
        stream_error = e
        warnings.warn(
            f'streaming stopped early: {type(e).__name__}: {e}. '
            f'Returning the figure with the {n_seen} samples consumed so '
            "far (see fig.stream_info; the exception is stored under "
            "fig.stream_info['error']).", RuntimeWarning, stacklevel=2)
    finally:
        if writer is not None:
            try:
                writer.finish()
                if writer_tmp is not None:
                    # mkstemp's private 0600 mode must not leak onto the
                    # saved animation (release-1.0 audit: security
                    # re-review; shared with hyp.save's atomic-write path)
                    from .save import _transfer_file_mode
                    _transfer_file_mode(writer_tmp, save_path)
                    os.replace(writer_tmp, save_path)
            finally:
                if writer_tmp is not None and os.path.exists(writer_tmp):
                    os.remove(writer_tmp)

    fig.stream_info = {
        'data': [np.vstack(raw)],
        'xform_data': [np.vstack(accum)],
        'n_samples': n_seen,
        'reduce_model': model,
        'truncated': truncated,
        'error': stream_error,
    }
    if show:
        plt.show()
    return fig
