# -*- coding: utf-8 -*-
"""Streaming-data support (issue #101): streams are detected from input
structure (no flag), models are fitted on the first stream_init samples,
and later samples are projected through the fitted models."""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

import hypertools as hyp
from hypertools.io.streaming import is_stream, row_to_vector


def walk_gen(n=300, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((3, dim))
    p = np.zeros(3)
    for _ in range(n):
        p = p + 0.1 * rng.standard_normal(3)
        yield p @ W


def test_is_stream_detection():
    assert is_stream(walk_gen())
    assert is_stream(iter([np.zeros(3)]))
    assert not is_stream([np.zeros((10, 3))])
    assert not is_stream(np.zeros((10, 3)))
    assert not is_stream(pd.DataFrame(np.zeros((10, 3))))
    assert not is_stream('some text')
    assert not is_stream({'a': 1})


def test_row_to_vector_formats():
    np.testing.assert_array_equal(row_to_vector([1, 2, 3]), [1., 2., 3.])
    np.testing.assert_array_equal(row_to_vector(np.arange(4)), np.arange(4.))
    # dict rows: numeric fields in insertion order, strings ignored
    row = {'a': 1.5, 'label': 'cat', 'v': [2.0, 3.0], 'flag': None}
    np.testing.assert_array_equal(row_to_vector(row), [1.5, 2.0, 3.0])
    with pytest.raises(ValueError):
        row_to_vector({'text': 'only strings here'})


def test_stream_plot_consumes_and_projects():
    geo = hyp.plot(walk_gen(300), show=False, stream_init=100,
                   stream_chunk=50, stream_max=None)
    assert geo.xform_data[0].shape == (300, 3)
    assert geo.data[0].shape == (300, 6)
    assert geo.stream_info['n_samples'] == 300
    assert not geo.stream_info['truncated']
    # every consumed sample is on the plot
    assert len(geo.ax.lines[0].get_data_3d()[0]) == 300
    plt.close('all')


def test_stream_models_fitted_on_head_only():
    """The reduction model must be fitted on the first stream_init samples
    and only *applied* afterwards (issue #101's core requirement)."""
    geo = hyp.plot(walk_gen(250), show=False, stream_init=80,
                   stream_chunk=40, stream_max=None)
    model = geo.stream_info['reduce_model']
    # IncrementalPCA records how many samples it was fitted on
    assert model.n_samples_seen_ == 80
    # the stored projection reproduces the plotted trajectory exactly
    expected = model.transform(geo.data[0])
    np.testing.assert_allclose(geo.xform_data[0], expected, atol=1e-10)
    plt.close('all')


def test_stream_max_cuts_off_stream():
    geo = hyp.plot(walk_gen(500), show=False, stream_init=50,
                   stream_chunk=50, stream_max=200)
    assert geo.stream_info['truncated']
    assert geo.xform_data[0].shape[0] == 200
    plt.close('all')


def test_infinite_stream_with_stream_max():
    """stream_max is the cutoff that makes an INFINITE stream saveable."""
    def infinite():
        rng = np.random.default_rng(1)
        p = np.zeros(4)
        while True:
            p = p + 0.1 * rng.standard_normal(4)
            yield p

    geo = hyp.plot(infinite(), show=False, stream_init=60, stream_chunk=30,
                   stream_max=150)
    assert geo.stream_info['truncated']
    assert geo.xform_data[0].shape[0] == 150
    plt.close('all')


def test_stream_window_limits_display_not_retention():
    geo = hyp.plot(walk_gen(300), show=False, stream_init=100,
                   stream_chunk=50, stream_max=None, stream_window=80)
    # display: only the trailing window is on the artist
    assert len(geo.ax.lines[0].get_data_3d()[0]) == 80
    # retention: everything consumed is on the geometry
    assert geo.xform_data[0].shape[0] == 300
    plt.close('all')


def test_stream_interrupt_finalizes(tmp_path):
    """Ctrl-C during an infinite stream still returns the geometry and
    finalizes the saved animation."""
    out = str(tmp_path / 'interrupted.gif')

    def interrupting(n_before=120):
        rng = np.random.default_rng(2)
        p = np.zeros(4)
        for i in range(n_before):
            p = p + 0.1 * rng.standard_normal(4)
            yield p
        raise KeyboardInterrupt

    geo = hyp.plot(interrupting(), show=False, stream_init=60,
                   stream_chunk=30, stream_max=None, save_path=out)
    assert geo.stream_info['truncated']
    assert geo.xform_data[0].shape[0] == 120
    with Image.open(out) as im:
        assert im.n_frames >= 2  # head + at least one chunk, finalized
    plt.close('all')


def test_stream_animation_export(tmp_path):
    out = str(tmp_path / 'stream.gif')
    hyp.plot(walk_gen(200), show=False, stream_init=100, stream_chunk=25,
             stream_max=None, save_path=out)
    plt.close('all')
    with Image.open(out) as im:
        # one frame for the head + one per chunk
        assert im.n_frames == 1 + (200 - 100) // 25
        inks = [int((np.asarray(f.convert('L')) < 150).sum())
                for f in ImageSequence.Iterator(im)]
    assert inks[-1] > inks[0], 'trajectory should grow across frames'


def test_stream_low_dim_passthrough():
    """<=3-dimensional samples need no reduction model."""
    geo = hyp.plot(iter(np.random.default_rng(0).standard_normal((120, 3))),
                   show=False, stream_init=40, stream_chunk=40,
                   stream_max=None)
    assert geo.stream_info['reduce_model'] is None
    assert geo.xform_data[0].shape == (120, 3)
    plt.close('all')


def test_stream_normalize_uses_head_stats():
    geo = hyp.plot(walk_gen(200), show=False, stream_init=100,
                   stream_chunk=50, stream_max=None, normalize='across')
    head = geo.data[0][:100]
    mu, sd = head.mean(axis=0), head.std(axis=0)
    model = geo.stream_info['reduce_model']
    expected = model.transform((geo.data[0] - mu) / sd)
    np.testing.assert_allclose(geo.xform_data[0], expected, atol=1e-10)
    plt.close('all')


def test_stream_rejects_nontransformable_reduce():
    with pytest.raises(ValueError, match='transform'):
        hyp.plot(walk_gen(150), show=False, reduce='TSNE', stream_max=None)


def test_stream_rejects_align_and_cluster():
    with pytest.raises(ValueError, match='align'):
        hyp.plot(walk_gen(150), show=False, align='hyper')
    with pytest.raises(ValueError, match='cluster'):
        hyp.plot(walk_gen(150), show=False, n_clusters=3)


def test_huggingface_iterable_dataset_stream():
    """Real Hugging Face streaming (issue #101 + review request):
    load_dataset(..., streaming=True) plots without materializing."""
    datasets = pytest.importorskip('datasets')
    ds = datasets.load_dataset('scikit-learn/iris', split='train',
                               streaming=True)
    ds = ds.select_columns(['SepalLengthCm', 'SepalWidthCm',
                            'PetalLengthCm', 'PetalWidthCm'])
    assert is_stream(ds)
    geo = hyp.plot(ds, '.', show=False, stream_init=50, stream_chunk=25,
                   stream_max=None)
    assert geo.stream_info['n_samples'] == 150
    assert geo.xform_data[0].shape == (150, 3)
    assert geo.data[0].shape == (150, 4)
    plt.close('all')


def test_stream_view_is_frozen_after_head():
    """Round-6.5: once the head sets the space, drawn positions of already-
    plotted points NEVER move as new chunks arrive (no per-chunk rescale
    'twitch'), and the axis limits stay fixed."""
    import hypertools.io.streaming as st

    # drifting walk: later samples leave the head's extent
    def drifting(n=300):
        rng = np.random.default_rng(5)
        p = np.zeros(4)
        for i in range(n):
            p = p + 0.05 * rng.standard_normal(4) + 0.05  # steady drift
            yield p

    geo = hyp.plot(drifting(), show=False, stream_init=100, stream_chunk=50,
                   stream_max=None)
    # reconstruct what the head looked like on the first draw: its first
    # 100 projected rows, pushed through the frozen transform, must equal
    # the first 100 drawn points of the final artist exactly
    xs, ys, zs = geo.ax.lines[0].get_data_3d()
    drawn = np.column_stack([xs, ys, zs])
    model = geo.stream_info['reduce_model']
    head_red = model.transform(geo.data[0][:100])
    mu = head_red.mean(axis=0, keepdims=True)
    c = head_red - mu
    m1, m2 = c.min(), (c - c.min()).max()
    expected_head = 2.0 * ((head_red - mu) - m1) / m2 - 1.0
    np.testing.assert_allclose(drawn[:100], expected_head, atol=1e-12)
    # limits are the standard fixed cube
    assert geo.ax.get_xlim3d()[0] <= -1 and geo.ax.get_xlim3d()[1] >= 1
    plt.close('all')


def test_stream_out_of_range_samples_clamped_to_box():
    def exploding(n=200):
        rng = np.random.default_rng(6)
        p = np.zeros(4)
        for i in range(n):
            step = 10.0 if i > 100 else 0.1  # explode after the head
            p = p + step * rng.standard_normal(4)
            yield p

    geo = hyp.plot(exploding(), show=False, stream_init=100, stream_chunk=50,
                   stream_max=None)
    xs, ys, zs = geo.ax.lines[0].get_data_3d()
    drawn = np.column_stack([xs, ys, zs])
    # every drawn point is inside (or exactly on) the box surface
    assert np.abs(drawn).max() <= 1.0 + 1e-12
    # and the post-explosion points actually hit the surface (clamped)
    assert np.isclose(np.abs(drawn[150:]).max(), 1.0)
    plt.close('all')
