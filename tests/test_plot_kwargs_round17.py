# -*- coding: utf-8 -*-
"""Round17 task 7: `label_alpha=` (#103), `xlabel=`/`ylabel=`/`zlabel=`, and
`animate=` dict form (#154 resolution).

Every assertion below inspects a REAL rendered artifact (a matplotlib bbox
patch's alpha, a plotly annotation's `bgcolor`, `ax.get_xlabel()`/
`get_ylabel()`/`get_zlabel()`, a plotly axis/scene title, or an actual
rendered animation frame/pixel buffer) rather than merely checking that a
call didn't raise. No mocks: every call below goes through the real
`hypertools.plot.plot` dispatcher and the real matplotlib/plotly backends.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp


def _walk(n=30, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, d)), axis=0)


def _sparse_labels(n, every=10):
    return [f"pt{i}" if i % every == 0 else None for i in range(n)]


def _mpl_label_bbox_alphas(fig):
    """Every drawn `labels=` annotation's bbox patch alpha, on a
    matplotlib Figure returned by `hyp.plot`."""
    ax = fig.axes[0]
    alphas = []
    for text in ax.texts:
        patch = text.get_bbox_patch()
        if patch is not None:
            alphas.append(patch.get_alpha())
    return alphas


# --------------------------------------------------------------- label_alpha

def test_label_alpha_default_mpl_is_half():
    data = _walk()
    fig = hyp.plot(data, show=False, labels=_sparse_labels(30))
    alphas = _mpl_label_bbox_alphas(fig)
    assert len(alphas) == 3
    assert all(a == pytest.approx(0.5) for a in alphas)


def test_label_alpha_custom_mpl():
    data = _walk()
    fig = hyp.plot(data, show=False, labels=_sparse_labels(30),
                    label_alpha=0.2)
    alphas = _mpl_label_bbox_alphas(fig)
    assert len(alphas) == 3
    assert all(a == pytest.approx(0.2) for a in alphas)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2, True, "0.5"])
def test_label_alpha_invalid_raises(bad):
    data = _walk(n=10)
    with pytest.raises(ValueError, match="label_alpha"):
        hyp.plot(data, show=False, label_alpha=bad)


def test_label_alpha_default_plotly_bgcolor():
    data = _walk()
    fig = hyp.plot(data, show=False, backend='plotly',
                    labels=_sparse_labels(30))
    annotations = fig.layout.scene.annotations
    assert len(annotations) == 3
    for ann in annotations:
        assert ann.bgcolor == 'rgba(255,255,255,0.5)'


def test_label_alpha_custom_plotly_bgcolor():
    data = _walk()
    fig = hyp.plot(data, show=False, backend='plotly',
                    labels=_sparse_labels(30), label_alpha=0.3)
    annotations = fig.layout.scene.annotations
    assert len(annotations) == 3
    for ann in annotations:
        assert ann.bgcolor == 'rgba(255,255,255,0.3)'


# ------------------------------------------------------------------- xyzlabel

def test_xlabel_ylabel_zlabel_mpl_static_3d():
    data = _walk()
    fig = hyp.plot(data, show=False, xlabel='X axis', ylabel='Y axis',
                    zlabel='Z axis')
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'X axis'
    assert ax.get_ylabel() == 'Y axis'
    assert ax.get_zlabel() == 'Z axis'


def test_xlabel_ylabel_mpl_static_2d():
    data = _walk()[:, :2]
    fig = hyp.plot(data, show=False, ndims=2, xlabel='2D X', ylabel='2D Y')
    ax = fig.axes[0]
    assert ax.get_xlabel() == '2D X'
    assert ax.get_ylabel() == '2D Y'


def test_xlabel_ylabel_zlabel_mpl_animated_3d():
    # matplotlib animation has no 2-D path -- animated plots are always
    # 3-D, so zlabel is valid here.
    data = _walk()
    fig, ani = hyp.plot(data, show=False, animate=True, duration=1,
                         xlabel='AX', ylabel='AY', zlabel='AZ')
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'AX'
    assert ax.get_ylabel() == 'AY'
    assert ax.get_zlabel() == 'AZ'


def test_zlabel_on_2d_raises_valueerror():
    data = _walk()[:, :2]
    with pytest.raises(ValueError, match="zlabel"):
        hyp.plot(data, show=False, ndims=2, zlabel='Z')


def test_xlabel_ylabel_zlabel_plotly_static_3d():
    data = _walk()
    fig = hyp.plot(data, show=False, backend='plotly', xlabel='PX',
                    ylabel='PY', zlabel='PZ')
    assert fig.layout.scene.xaxis.title.text == 'PX'
    assert fig.layout.scene.yaxis.title.text == 'PY'
    assert fig.layout.scene.zaxis.title.text == 'PZ'


def test_xlabel_ylabel_plotly_static_2d():
    data = _walk()[:, :2]
    fig = hyp.plot(data, show=False, backend='plotly', ndims=2,
                    xlabel='P2X', ylabel='P2Y')
    assert fig.layout.xaxis.title.text == 'P2X'
    assert fig.layout.yaxis.title.text == 'P2Y'


def test_xlabel_ylabel_zlabel_plotly_animated_persists_in_frames():
    data = _walk()
    fig = hyp.plot(data, show=False, backend='plotly', animate=True,
                    duration=1, frame_rate=5, xlabel='APX', ylabel='APY',
                    zlabel='APZ')
    assert fig.layout.scene.xaxis.title.text == 'APX'
    assert fig.layout.scene.yaxis.title.text == 'APY'
    assert fig.layout.scene.zaxis.title.text == 'APZ'
    # titles are set once on the base layout (not per-frame), so they
    # apply throughout every animation frame automatically -- there is no
    # separate per-frame layout to check.
    assert len(fig.frames) > 0


# ------------------------------------------------------------- animate= dict

def test_animate_dict_missing_style_raises():
    data = _walk(n=10)
    with pytest.raises(ValueError, match="style"):
        hyp.plot(data, show=False, animate={'rotations': 2})


def test_animate_dict_unknown_key_raises():
    data = _walk(n=10)
    with pytest.raises(ValueError, match="unknown key"):
        hyp.plot(data, show=False, animate={'style': 'spin', 'bogus': 1})


def test_animate_dict_conflict_with_flat_kwarg_raises():
    data = _walk(n=10)
    with pytest.raises(ValueError, match="rotations"):
        hyp.plot(data, show=False,
                 animate={'style': 'spin', 'rotations': 2}, rotations=5)


def test_animate_dict_no_conflict_when_values_match():
    # same value in both places is NOT a conflict -- only a MISMATCHED
    # value raises.
    data = _walk(n=10)
    fig, ani = hyp.plot(data, show=False,
                         animate={'style': 'spin', 'rotations': 2},
                         rotations=2, duration=1)
    assert ani is not None


def test_animate_dict_matches_flat_kwargs_mpl_pixel_identical():
    """A dict-form spin animation must render EXACTLY the same frames as
    the equivalent flat-kwarg call (fixed seed data, same frame index) --
    compared via the actual rendered RGBA pixel buffer, not just "no
    error"."""
    def _render_frame(fig, ani, frame_idx):
        ani._func(frame_idx, *ani._args)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba()).copy()

    data_dict = _walk(n=25, seed=7)
    data_flat = _walk(n=25, seed=7)

    fig_dict, ani_dict = hyp.plot(
        data_dict, show=False, frame_rate=10,
        animate={'style': 'spin', 'rotations': 2, 'duration': 1})
    fig_flat, ani_flat = hyp.plot(
        data_flat, show=False, frame_rate=10,
        animate='spin', rotations=2, duration=1)

    for frame_idx in (0, 5, 9):
        img_dict = _render_frame(fig_dict, ani_dict, frame_idx)
        img_flat = _render_frame(fig_flat, ani_flat, frame_idx)
        assert np.array_equal(img_dict, img_flat), (
            f"frame {frame_idx} differs between animate=dict and "
            "animate=flat-kwargs forms"
        )


def test_animate_dict_matches_flat_kwargs_plotly_identical_frames():
    """Same equivalence check as the matplotlib test above, but for the
    plotly backend: the dict-form and flat-kwarg calls must produce
    IDENTICAL frame data/layout (compared via `to_plotly_json()`, which
    serializes every trace/layout value actually sent to the renderer)."""
    data_dict = _walk(n=25, seed=11)
    data_flat = _walk(n=25, seed=11)

    fig_dict = hyp.plot(
        data_dict, show=False, backend='plotly', frame_rate=10,
        animate={'style': 'spin', 'rotations': 2, 'duration': 1})
    fig_flat = hyp.plot(
        data_flat, show=False, backend='plotly', frame_rate=10,
        animate='spin', rotations=2, duration=1)

    assert fig_dict.to_plotly_json()['data'] == fig_flat.to_plotly_json()['data']
    assert fig_dict.to_plotly_json()['layout'] == fig_flat.to_plotly_json()['layout']
    frames_dict = fig_dict.to_plotly_json()['frames']
    frames_flat = fig_flat.to_plotly_json()['frames']
    assert len(frames_dict) == len(frames_flat) > 0
    assert frames_dict == frames_flat
