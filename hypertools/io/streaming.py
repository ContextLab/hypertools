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

import numpy as np
import pandas as pd



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
            return m
    elif normalize == 'row':
        def norm(m):
            return (m - m.mean(axis=1, keepdims=True)) / \
                _safe_std(m, axis=1, keepdims=True)
    elif normalize in ('across', 'within', True):
        mu = head.mean(axis=0, keepdims=True)
        sd = _safe_std(head, axis=0, keepdims=True)

        def norm(m):
            return (m - mu) / sd
    else:
        raise ValueError(f'unsupported normalize option for streaming data: '
                         f'{normalize!r}')
    head_n = norm(head)

    # reduction spec: name / dict / class / instance, mirroring tools.reduce
    if isinstance(reduce, dict):
        model_spec = reduce.get('model')
        params = dict(reduce.get('params', {}))
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
    samples have been consumed, or the user interrupts (Ctrl-C) -- infinite
    streams render continually, and any animation being saved is finalized
    whenever streaming stops, including on interrupt. ``stream_window``
    optionally limits the *display* to the most recent samples (comet
    style); all consumed data is still retained on the returned geometry.

    Returns a DataGeometry; data holds the raw consumed samples and
    xform_data the projected trajectory, and .stream_info records the
    fitted model and how much of the stream was consumed.
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

    it = iter(stream)
    head_rows = list(itertools.islice(it, int(stream_init)))
    if not head_rows:
        raise ValueError('stream produced no samples')
    head = np.vstack([row_to_vector(r) for r in head_rows])

    head_red, project, model = _fit_stream_models(
        head, reduce, ndims, normalize)

    # initial plot on the head (already normalized/reduced -> disable both)
    geo = hyp_plot(head_red, fmt, reduce=None, normalize=None, ndims=ndims,
                   show=False, **plot_kwargs)
    artist = next(ln for ln in geo.ax.lines if len(ln.get_data()[0]))

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
        t = 2.0 * ((pts - head_mu) - box_m1) / box_m2 - 1.0
        return np.clip(t, -1.0, 1.0)

    is3d = head_red.shape[1] > 2
    raw = [head]
    accum = [head_red]
    n_seen = len(head_rows)

    interactive = show and matplotlib.get_backend().lower() not in (
        'agg', 'pdf', 'svg', 'ps')
    writer = None
    if save_path is not None:
        writer = animation.PillowWriter(fps=frame_rate)
        writer.setup(geo.fig, save_path, dpi=geo.fig.dpi)
        writer.grab_frame()

    def _redraw():
        # fixed head-fitted transform + clamp: the space inside the cube is
        # stable for the whole stream (no per-chunk re-scaling "twitch")
        shown = np.vstack(accum)
        if stream_window is not None:
            shown = shown[-int(stream_window):]
        full = to_box(shown)
        if is3d:
            artist.set_data_3d(full[:, 0], full[:, 1], full[:, 2])
        else:
            artist.set_data(full[:, 0], full[:, 1])
        if writer is not None:
            writer.grab_frame()
        if interactive:
            geo.fig.canvas.draw_idle()
            plt.pause(0.001)

    # consume until the stream is exhausted, stream_max is reached, or the
    # user interrupts -- an infinite stream renders continually, and the
    # animation (if any) is finalized whenever streaming stops
    truncated = False
    try:
        while True:
            if stream_max is not None and n_seen >= int(stream_max):
                truncated = any(True for _ in itertools.islice(it, 1))
                break
            take = int(stream_chunk)
            if stream_max is not None:
                take = min(take, int(stream_max) - n_seen)
            rows = list(itertools.islice(it, take))
            if not rows:
                break
            chunk = np.vstack([row_to_vector(r) for r in rows])
            raw.append(chunk)
            accum.append(project(chunk))
            n_seen += len(rows)
            _redraw()
    except KeyboardInterrupt:
        truncated = True

    if writer is not None:
        writer.finish()

    geo.data = [np.vstack(raw)]
    geo.xform_data = [np.vstack(accum)]
    geo.stream_info = {'n_samples': n_seen, 'reduce_model': model,
                       'truncated': truncated}
    if show:
        plt.show()
    return geo
