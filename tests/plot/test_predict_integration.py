# -*- coding: utf-8 -*-
"""predict= / impute= integration tests for hyp.plot / hyp.analyze (Task 6,
GH #169). Real forecasters/imputers run on real (small) data -- no mocks.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp


def _walk(seed, n=30, d=3, offset=0.0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, d)), axis=0) + offset


def _forecasts(ax, role=None):
    """Forecast artists identify THEMSELVES via `_hyp_forecast_role`.

    Linestyle is NOT a discriminator: since 1.1.0 a forecast INHERITS its
    observed trace's linestyle (so a solid dataset's forecast is solid), and
    user data drawn with fmt='--' was always dashed too. See docs/animation.rst
    ("Identifying forecast artists").
    """
    out = [ln for ln in ax.lines
           if getattr(ln, '_hyp_forecast_role', None) is not None]
    if role is not None:
        out = [ln for ln in out if ln._hyp_forecast_role == role]
    return out


def _plotly_forecasts(fig, role=None):
    """The plotly twin of `_forecasts` -- `meta['hyp_forecast_role']`."""
    out = [tr for tr in fig.data
           if (tr.meta or {}).get('hyp_forecast_role') is not None]
    if role is not None:
        out = [tr for tr in out
               if (tr.meta or {})['hyp_forecast_role'] == role]
    return out


def _plotly_alpha(trace):
    """The alpha baked into a plotly trace's ``rgba(...)`` line colour."""
    return float(trace.line.color.rsplit(',', 1)[1].rstrip(') '))


def _legend_labels(fig):
    lg = fig.axes[0].get_legend()
    assert lg is not None, 'no legend was drawn'
    return [t.get_text() for t in lg.get_texts()]


# --- matplotlib: artist count, style/alpha/color, legend, forecast length ---

@pytest.mark.parametrize('ndims', [2, 3])
def test_predict_adds_one_forecast_per_dataset_styled_like_its_source(ndims):
    """A forecast is the same series projected forward, so it inherits its
    observed trace's colour, linestyle AND linewidth, differing only in
    transparency (observed * 0.5). Replaces the pre-1.1.0 contract, under
    which every forecast was dashed at a hard-coded alpha=0.6."""
    a = _walk(1, d=max(ndims, 2))
    b = _walk(2, d=max(ndims, 2), offset=5.0)
    t = 15

    fig = hyp.plot([a, b], predict='Kalman', t=t, ndims=ndims, show=False)
    ax = fig.axes[0]
    plt.close(fig)

    assert len(ax.lines) == 2 * len([a, b])

    src_lines, fc_lines = ax.lines[:2], ax.lines[2:]
    assert fc_lines == _forecasts(ax, role='static')
    for src, fc in zip(src_lines, fc_lines):
        assert fc.get_linestyle() == src.get_linestyle()
        assert fc.get_linewidth() == pytest.approx(src.get_linewidth())
        # observed alpha is unset (matplotlib's "opaque", i.e. 1.0) -> 0.5
        assert src.get_alpha() is None
        assert fc.get_alpha() == pytest.approx(0.5)
        assert fc.get_color() == src.get_color()
        assert fc.get_label() == '_nolegend_'
        # the forecast is drawn SMOOTHED like any line (PCHIP-densified well
        # beyond the raw t+1 vertices for a short horizon), and its
        # seam-prepended first vertex still joins the source trajectory's end.
        n_pts = len(fc.get_xdata())
        assert n_pts > t + 1
        assert fc.get_xdata()[0] == pytest.approx(src.get_xdata()[-1])
        assert fc.get_ydata()[0] == pytest.approx(src.get_ydata()[-1])


def test_predict_forecast_drawn_smoothed_not_straight_segments():
    """A very short forecast is drawn as a SMOOTH (PCHIP-densified) curve,
    not a handful of straight segments -- the same auto-smoothing any
    static line gets -- while the RETURNED forecast still has exactly t rows
    (only the DRAWN trace is densified)."""
    a = _walk(11)
    t = 4  # raw drawn trace would be only t + 1 = 5 vertices without smoothing
    bundle = hyp.plot(a, predict='Kalman', t=t, show=False, return_model=True)
    ax = bundle['fig'].axes[0]
    fc = _forecasts(ax, role='static')[0]
    plt.close(bundle['fig'])
    # densified far beyond the raw 5 vertices (matches the static-line target)
    assert len(fc.get_xdata()) >= 100
    # ...but the returned forecast array is untouched (exactly t rows)
    assert np.asarray(bundle['predict']['forecasts'][0]).shape[0] == t


def test_predict_legend_unchanged_no_duplicate_entries():
    a = _walk(3)
    b = _walk(4, offset=5.0)
    fig = hyp.plot([a, b], predict='Kalman', t=10, legend=['first', 'second'],
                  show=False)
    labels = _legend_labels(fig)
    plt.close(fig)
    assert labels == ['first', 'second']  # exactly one entry per dataset


def test_predict_return_model_bundle():
    a = _walk(5)
    b = _walk(6, offset=3.0)
    t = 12
    bundle = hyp.plot([a, b], predict='Kalman', t=t, show=False,
                      return_model=True)
    plt.close(bundle['fig'])

    assert bundle['predict']['model'] == 'Kalman'
    assert bundle['predict']['params'] == {'t': t}
    forecasts = bundle['predict']['forecasts']
    assert len(forecasts) == 2
    for fc in forecasts:
        # exactly t forecast rows, matching hyp.predict (X1-api-016: the
        # bundle used to include the drawn overlay's prepended seam row,
        # an off-by-one vs. hyp.predict; the DRAWN trace still has t + 1
        # vertices -- see test_predict_overlay_traces above)
        assert np.asarray(fc).shape == (t, 3)
    assert bundle['models']['impute'] is None


def test_predict_with_gaussian_process_no_extra_dependency():
    """GaussianProcess ships with scikit-learn (no [predict] extra needed)."""
    a = _walk(7)
    b = _walk(8, offset=2.0)
    fig = hyp.plot([a, b], predict='GaussianProcess', t=6, show=False)
    ax = fig.axes[0]
    plt.close(fig)
    assert len(ax.lines) == 4


# --- forecasts must stay inside the drawn frame (square/cube) --------------
# Regression guard: forecasts used to be scaled AFTER helpers.scale mapped
# the observed data into [-1, 1], so a forecast extending beyond the observed
# range rendered OUTSIDE the black square/cube frame (axes are off, nothing
# clips). The frame must contain everything drawn: center/scale statistics
# are now computed from the FULL stacked data (observed + forecasts).

def _spiral(phase):
    s = np.linspace(0, 4 * np.pi, 90)
    return np.column_stack([np.cos(s + phase), np.sin(s + phase), s / 4,
                            0.5 * np.cos(2 * s + phase),
                            0.5 * np.sin(2 * s + phase)])


def _line_pts(line, ndims):
    if ndims == 3:
        xs, ys, zs = line.get_data_3d()
        return np.column_stack([xs, ys, zs])
    return np.column_stack([line.get_xdata(), line.get_ydata()])


# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures,
# and its lbfgs optimizer can stop early on the same contrived data (only
# the GaussianProcess parameter case emits either)
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
@pytest.mark.filterwarnings(
    'ignore:lbfgs failed to converge:sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('ndims,model', [(2, 'ARIMA'), (3, 'GaussianProcess')])
def test_forecast_vertices_stay_inside_frame(ndims, model):
    if model == 'ARIMA':
        pytest.importorskip('statsmodels')
    data = [_spiral(0.0), _spiral(2.0)]

    fig = hyp.plot(data, ndims=ndims, predict=model, t=30,
                   legend=['x', 'y'], show=False)
    ax = fig.axes[0]
    plt.close(fig)

    fc_lines = [l for l in ax.lines if l.get_label() == '_nolegend_']
    assert len(fc_lines) == len(data)
    # square/cube frame spans [-1, 1]: every forecast vertex must be inside
    for line in fc_lines:
        pts = _line_pts(line, ndims)
        assert pts.min() >= -1.0 - 1e-9
        assert pts.max() <= 1.0 + 1e-9
    # observed data stays inside the frame too (joint center/scale)
    for line in [l for l in ax.lines if l.get_label() != '_nolegend_']:
        pts = _line_pts(line, ndims)
        assert pts.min() >= -1.0 - 1e-9
        assert pts.max() <= 1.0 + 1e-9


# --- animate + predict: 'morph' has no time axis, so it still refuses -------
# (True/'parallel'/'serial'/'window' became supported in 1.1: the forecast is
# precomputed per frame from the history revealed so far -- see
# tests/plot/test_predict_animation.py. 'spin' was always allowed; see below.)

@pytest.mark.parametrize('mode', ['morph', ['morph', 'morph']])
def test_morph_animate_and_predict_raises_not_implemented(mode):
    # a morph interpolates between point CLOUDS rather than progressing along
    # a time axis, so there is no history to forecast from. The list form is
    # covered explicitly because `_resolve_animate_mode` does not run until
    # long after the refusal, so at the check `animate` is still a raw list.
    rng = np.random.default_rng(0)
    a, b = (rng.normal(size=(120, 3)) + off for off in (0.0, 4.0))
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot([a, b], '.', predict='Kalman', animate=mode,
                 morph_samples=120, duration=1, frame_rate=2, show=False)


def test_predict_with_spin_renders_static_forecast_overlay(tmp_path):
    # animate='spin' only rotates the camera around the STATIC scene, so the
    # fixed forecast overlay is coherent: it is drawn once and rotates
    # with everything else (GH #169 follow-up).
    a = _walk(9)
    b = _walk(10, offset=1.0)
    t = 12
    fig, ani = hyp.plot([a, b], predict='Kalman', t=t, animate='spin',
                        rotations=1, duration=1, frame_rate=5, show=False)
    ax = fig.axes[0]

    # one forecast overlay per dataset, same styling as the static path
    fc_lines = _forecasts(ax, role='static')
    assert len(fc_lines) == len([a, b])
    for fc in fc_lines:
        assert fc.get_alpha() == pytest.approx(0.5)
        assert fc.get_label() == '_nolegend_'
        assert len(fc.get_xdata()) > t + 1  # smoothed (densified) beyond raw t+1
        # unclipped like the other 3-D line artists, so a rotated camera
        # never crops the overlay (matches animate_plot3D's set_clip_on(False))
        assert fc.get_clip_on() is False

    # the animation must actually render end-to-end (camera-only spin)
    out = tmp_path / 'predict_spin.gif'
    ani.save(str(out))
    assert out.stat().st_size > 0
    plt.close(fig)


def test_predict_with_spin_return_model_bundle_carries_forecasts():
    a = _walk(9)
    b = _walk(10, offset=1.0)
    t = 12
    bundle = hyp.plot([a, b], predict='Kalman', t=t, animate='spin',
                      rotations=1, duration=1, frame_rate=5, show=False,
                      return_model=True)
    assert bundle['animation'] is not None
    assert bundle['predict']['model'] == 'Kalman'
    forecasts = bundle['predict']['forecasts']
    assert len(forecasts) == 2
    for fc in forecasts:
        assert np.asarray(fc).shape[0] == t  # unprepended: exactly t rows
    plt.close(bundle['fig'])


# --- plotly backend parity: trace count / style / showlegend --------------

def test_plotly_predict_trace_parity():
    pytest.importorskip('plotly')
    a = _walk(11, d=2)
    b = _walk(12, d=2, offset=4.0)
    t = 8

    fig = hyp.plot([a, b], predict='Kalman', t=t, ndims=2, backend='plotly',
                   show=False)

    # 2D plot: 2 data traces + 2 forecast traces, no wireframe cube (3D only)
    assert len(fig.data) == 4
    src_traces, fc_traces = fig.data[:2], fig.data[2:4]
    assert list(fc_traces) == _plotly_forecasts(fig, role='static')
    for src, tr in zip(src_traces, fc_traces):
        # inherits the observed trace's dash and width, half its opacity
        assert tr.line.dash == src.line.dash
        assert tr.line.width == pytest.approx(src.line.width)
        assert _plotly_alpha(tr) == pytest.approx(0.5 * _plotly_alpha(src))
        assert tr.meta['hyp_forecast_alpha'] == pytest.approx(0.5)
        assert tr.showlegend is False


# --- impute= smoke test through plot() -------------------------------------

def test_impute_smoke_through_plot():
    rng = np.random.default_rng(13)
    data = np.cumsum(rng.standard_normal((40, 3)), axis=0)
    data[5, :] = np.nan
    data[10, 1] = np.nan

    fig = hyp.plot(data, impute='KNNImputer', show=False)
    plt.close(fig)
    assert fig is not None


def test_impute_through_analyze_direct():
    rng = np.random.default_rng(14)
    data = rng.standard_normal((30, 4))
    data[3, :] = np.nan

    out = hyp.analyze(data, normalize='within', impute='KNNImputer')
    assert not np.isnan(np.vstack(out if isinstance(out, list) else [out])).any()
