# tests/plot/test_predict_animation.py
"""`predict=` with time-progressing animations (matplotlib backend)."""

import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pytest

import hypertools as hyp


def _series(n=3, rows=60, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _forecasts(ax, role=None):
    """Forecast artists identify THEMSELVES (Contract 5). Linestyle is not a
    discriminator: user data drawn with fmt='--' is dashed too."""
    out = [ln for ln in ax.lines
           if getattr(ln, '_hyp_forecast_role', None) is not None]
    if role is not None:
        out = [ln for ln in out if ln._hyp_forecast_role == role]
    return out


def test_predict_with_animate_true_no_longer_raises():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=2, frame_rate=4, show=False)
    assert ani is not None


@pytest.mark.parametrize('mode', [True, 'parallel', 'serial', 'window'])
def test_time_progressing_animation_draws_no_static_full_history_overlay(mode):
    """plot.py:4907 had no `animate` guard, so a time-progressing animation
    would draw BOTH a frozen full-history overlay AND the per-frame one.
    Measured before the fix on animate='spin': 3 dashed 901-vertex overlays,
    landing FIRST in ax.lines."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=mode,
                        duration=2, frame_rate=4, show=False)
    assert _forecasts(_ax(fig), role='static') == []


def test_spin_still_draws_the_static_overlay():
    """Regression: 'spin' only rotates the camera, so its fixed overlay is
    correct and must be untouched -- including alpha/label/clip."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate='spin',
                        duration=2, frame_rate=4, show=False)
    ax = _ax(fig)
    static = _forecasts(ax, role='static')
    solid = _solid(ax)
    assert len(static) == 3
    for fc, src in zip(static, solid):
        # 1.0.1: inherits the observed trace's style, at half its alpha
        assert fc.get_linestyle() == src.get_linestyle()
        assert fc.get_alpha() == pytest.approx(0.5)
        assert fc.get_label() == '_nolegend_'
        assert fc.get_clip_on() is False
    ani._func(1, *ani._args)
    first = [np.array(ln.get_data_3d()) for ln in static]
    ani._func(6, *ani._args)
    for a, ln in zip(first, static):
        assert np.allclose(a, np.array(ln.get_data_3d())), \
            'spin forecast overlay must stay fixed'


def test_static_plot_still_draws_the_static_overlay():
    fig = hyp.plot(_series(), '-', predict='Kalman', t=3, show=False)
    assert len(_forecasts(_ax(fig), role='static')) == 3


def test_scalar_morph_still_refuses_predict():
    """A morph interpolates between point clouds; there is no time axis."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot(clouds, '.', predict='Kalman', t=3, animate='morph',
                 morph_samples=120, duration=1, frame_rate=2, show=False)


def test_list_form_morph_still_refuses_predict():
    """`_resolve_animate_mode` runs at plot.py:4158, ~1400 lines AFTER the
    refusal at plot.py:2740 -- so at the check `animate` is still a raw list
    and `animate == 'morph'` is False. Naive narrowing would silently ACCEPT
    a per-dataset morph list into the forecast path."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot(clouds, '.', predict='Kalman', t=3,
                 animate=['morph', 'morph'], morph_samples=120,
                 duration=1, frame_rate=2, show=False)


def _solid(ax):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None
            and ln.get_linestyle() in ('-', 'solid')]


def test_a_live_forecast_artist_exists_per_dataset():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=2, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(6, *ani._args)
    assert len(_forecasts(ax, role='live')) == 3


# --- the internal-phase contract, from the consumer's side (Task 0) -------

def test_a_user_callback_sees_this_frames_forecast_not_the_last_ones():
    """Regression for the ordering defect: the forecast updater is a LIBRARY
    updater and must run before user `on_frame=` callbacks. Registered on the
    same list, it lands after them and every callback reads the PREVIOUS
    frame's forecast geometry -- an off-by-one nothing raises about.

    Compares what the callback SAW at frame f against what the artist holds
    after frame f is fully drawn. Stale-by-one fails on the first frame that
    moves the forecast.
    """
    seen = []

    def watch(ctx):
        art = _forecasts(_ax(ctx.figure), role='live')[0]
        seen.append(np.array(art.get_data_3d()))

    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3,
                        animate=True, duration=4, frame_rate=4, show=False,
                        on_frame=watch)
    ax = _ax(fig)
    for frame in (4, 8, 12):
        ani._func(frame, *ani._args)
        after = np.array(_forecasts(ax, role='live')[0].get_data_3d())
        assert np.allclose(seen[-1], after), (
            'the user callback observed a different forecast than the one '
            'drawn for this frame -- internal updaters ran too late')
    # and the forecast really did move, so the assertion had something to bite
    assert not np.allclose(seen[0], seen[-1])


def test_the_forecast_updater_runs_with_no_user_callback_registered():
    """The common case: animated `predict=` and no `on_frame=` at all. If
    dispatch's early return still asks only about USER callbacks, the
    forecast silently freezes at frame 0."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3,
                        animate=True, duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    seen = []
    for frame in (2, 10):
        ani._func(frame, *ani._args)
        seen.append(np.array(_forecasts(ax, role='live')[0].get_data_3d()))
    assert not np.allclose(seen[0], seen[1]), 'forecast never advanced'


@pytest.mark.parametrize('frames', [
    (12, 8, 4, 0),           # strictly backwards
    (0, 12, 4, 8, 4),        # shuffled, with a repeat
])
def test_frames_drawn_out_of_order_give_the_same_geometry(frames):
    """A frame must be a pure function of its index. matplotlib re-delivers
    frame indices on loop and on `save()`, so any hidden accumulation shows
    up as a different artist for the same index."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3,
                        animate=True, duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    forward = {}
    for frame in sorted(set(frames)):
        ani._func(frame, *ani._args)
        forward[frame] = np.array(_forecasts(ax, role='live')[0].get_data_3d())
    for frame in frames:
        ani._func(frame, *ani._args)
        got = np.array(_forecasts(ax, role='live')[0].get_data_3d())
        assert np.allclose(got, forward[frame]), (
            f'frame {frame} drew differently out of order')


def _plot_ax(fig):
    """The axes the trajectories are drawn on, for EITHER dimensionality.

    `_ax` above hard-selects the 3-D axes (`hasattr(a, 'zaxis')`) and
    raises `IndexError` on a 2-D figure, which has no zaxis at all -- so a
    2-D test cannot use it.
    """
    solid = [a for a in fig.axes if hasattr(a, 'zaxis')]
    return solid[0] if solid else fig.axes[0]


@pytest.mark.parametrize('ndims', [2, 3])
def test_the_live_forecast_updates_in_both_2d_and_3d(ndims):
    """A 3-D forecast artist is a `Line3D`: `set_data` alone leaves its
    z-data untouched, so a 3-D forecast would silently draw in the wrong
    place. Both branches of the updater's `_ndims >= 3` dispatch are
    exercised here, and each is checked on every axis it owns.

    Note `dims=` (the INPUT feature count) is separate from `ndims=` (the
    display dimensionality): `_series` names it `dims`, and it must stay
    >= 2 so there is something to reduce.
    """
    fig, ani = hyp.plot(_series(n=1, dims=max(ndims, 2)), '-',
                        predict='Kalman', t=3, animate=True, duration=4,
                        frame_rate=4, ndims=ndims, show=False)
    ax = _plot_ax(fig)
    got = []
    for frame in (4, 12):
        ani._func(frame, *ani._args)
        art = _forecasts(ax, role='live')[0]
        got.append(np.array(art.get_data_3d() if ndims >= 3
                            else art.get_data()))
    assert got[0].shape[0] == (3 if ndims >= 3 else 2)
    for axis in range(got[0].shape[0]):
        assert not np.allclose(got[0][axis], got[1][axis]), (
            f'axis {axis} never moved -- 3-D artists need set_data_3d')


def test_forecast_head_tracks_the_animation():
    """The forecast must start at the CURRENT head, not at the final point."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    heads = []
    for frame in (4, 8, 12):
        ani._func(frame, *ani._args)
        heads.append(np.array(_forecasts(ax, role='live')[0].get_data_3d())[:, 0])
    assert not np.allclose(heads[0], heads[1]), 'forecast head did not move'
    assert not np.allclose(heads[1], heads[2]), 'forecast head did not move'


def test_forecast_is_anchored_near_the_drawn_head():
    """Contract 2. `t` is in RAW analyze-space samples, but the drawn head
    sits on the FRAME GRID: plot.py:4460-4478 resamples a 60-row input to
    round(frame_rate*duration) rows, which matplotlib_backend then densifies.
    This test runs duration=4/frame_rate=4, so 60 raw rows -> **16** grid
    rows. (The review's "60 -> 8 grid rows -> 904 drawn vertices, ~15.1x" was
    measured at duration=2/frame_rate=4 -- half the grid, a different
    configuration from this one; the densification ratio is quoted there, not
    here.) So the forecast anchors on the last revealed RAW sample, which is
    at most ONE raw step behind the drawn head -- an exact atol=1e-6 anchor is
    impossible by construction.

    The tolerance is DERIVED, not guessed. With antialias=False the drawn head
    line's vertices are consecutive FRAME-GRID rows, and one frame-grid step
    spans 59/15 ~= 3.9 raw steps here -- comfortably more than the <= 1 raw
    step of anchor separation -- so the largest drawn vertex spacing is a
    valid upper bound that needs no magic number.

    The discriminating assertion is the second one: anchoring on the FINAL
    observation (what the static overlay does) puts the gap many raw steps
    away and fails."""
    data = _series(n=1)
    fig, ani = hyp.plot(data, '-', predict='Kalman', t=3, animate=True,
                        antialias=False, duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(8, *ani._args)
    drawn = np.array(_solid(ax)[0].get_data_3d())
    fc = np.array(_forecasts(ax, role='live')[0].get_data_3d())
    head = drawn[:, -1]
    gap = np.linalg.norm(fc[:, 0] - head)

    one_grid_step = np.linalg.norm(np.diff(drawn, axis=1), axis=0).max()
    assert gap <= one_grid_step, (
        f'anchor gap {gap} exceeds one frame-grid step {one_grid_step}')

    # the same data drawn statically: its forecast hangs off the FINAL
    # observation, which at frame 8 of 16 is far from the current head
    static = hyp.plot(data, '-', predict='Kalman', t=3, antialias=False,
                      show=False)
    static_fc = np.array(
        _forecasts(_ax(static), role='static')[0].get_data_3d())
    assert gap < np.linalg.norm(static_fc[:, 0] - head), (
        'forecast is anchored on the FINAL observation, not the current head')


def test_forecast_stays_inside_the_axes_limits():
    """Contract 4: the box is built from data + the WHOLE schedule, so this
    holds by construction and nothing is clamped. Measured before the fix:
    1 of 7 partial-history Kalman forecasts fell outside the fixed [-1, 1]
    animated cube."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=5, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    assert _forecasts(ax, role='live'), (
        'no live forecast artists, so the per-frame assertions below would '
        'iterate an empty list and pass vacuously')
    for frame in range(16):
        ani._func(frame, *ani._args)
        for fc in _forecasts(ax, role='live'):
            pts = np.array(fc.get_data_3d())
            if pts.size == 0:
                continue
            assert (pts.min(axis=1) >= lims[:, 0] - 1e-6).all()
            assert (pts.max(axis=1) <= lims[:, 1] + 1e-6).all()


def test_t_is_measured_in_raw_samples_not_frames_or_vertices():
    """antialias=False draws the raw vertices, so the count is checkable."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                        antialias=False, duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(8, *ani._args)
    assert np.array(_forecasts(ax, role='live')[0].get_data_3d()).shape[1] == 4


def test_t_equals_one_is_the_next_raw_step():
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=1, animate=True,
                        antialias=False, duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(8, *ani._args)
    assert np.array(_forecasts(ax, role='live')[0].get_data_3d()).shape[1] == 2


def test_antialias_true_smooths_the_forecast_like_any_other_line():
    """plot.py:2255-2259 documents this as contract ('Forecast overlays drawn
    by predict= are smoothed the same way'), and the spin overlay is pinned to
    it at test_predict_integration.py:198. Measured today: t=1 draws 900
    vertices at antialias=True, 2 at antialias=False."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=1, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(8, *ani._args)
    assert np.array(_forecasts(ax, role='live')[0].get_data_3d()).shape[1] > 2


def test_frames_are_idempotent():
    """Contract 6: ani.save()/to_jshtml() replay from frame 0, and these tests
    drive frames out of order."""
    fig, ani = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(9, *ani._args)
    first = np.array(_forecasts(ax, role='live')[0].get_data_3d())
    for f in (0, 3, 15, 2, 9):
        ani._func(f, *ani._args)
    assert np.allclose(first,
                       np.array(_forecasts(ax, role='live')[0].get_data_3d()))


def test_forecast_composes_with_order_serial():
    """Requires animation-core Task 5 (order=)."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        order='serial', duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(15, *ani._args)
    drawn = [fc for fc in _forecasts(ax, role='live')
             if np.array(fc.get_data_3d()).size]
    assert len(drawn) == 3, 'every dataset is fully revealed by the last frame'


def test_a_dataset_with_too_little_history_hides_its_forecast():
    """Frame 0 reveals one raw row; a forecaster cannot be fitted to it, and
    an empty/garbage trace must not be drawn instead."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(0, *ani._args)
    live = _forecasts(ax, role='live')
    assert live, (
        'no live forecast artists, so "the forecast is hidden" would hold '
        'vacuously -- this must distinguish hidden from absent')
    for fc in live:
        assert not fc.get_visible() or np.array(fc.get_data_3d()).size == 0


def test_forecast_artists_are_not_identified_by_linestyle():
    """T5: user data drawn with fmt='--' is dashed but is NOT a forecast."""
    fig, ani = hyp.plot(_series(), '--', predict='Kalman', t=3, animate=True,
                        duration=2, frame_rate=4, show=False)
    ax = _ax(fig)
    ani._func(6, *ani._args)
    dashed = [ln for ln in ax.lines if ln.get_linestyle() not in ('-', 'solid')]
    assert len(dashed) > len(_forecasts(ax)), \
        'the dashed-linestyle heuristic would have swept up the data lines'
    assert len(_forecasts(ax, role='live')) == 3


def test_hue_regrouping_now_ANIMATES_its_forecasts_like_the_static_plot():
    """The static/animated asymmetry this test used to pin is GONE.

    It formerly asserted the refusal: an animated `hue=`/`cluster=` plot drew
    no forecast and warned, because the schedule mapped frame-grid rows onto
    each DATASET's raw rows while regrouping had replaced the per-dataset
    traces with per-RUN ones. The regrouped-reveal work supplies the missing
    mapping -- `TraceOwnership` says which dataset each run came from, and
    `DatasetRevealSchedule` reads each frame's visible rows off the very
    `RunWindow`s the backend sliced its artists with -- so the animated path
    now draws what the static path draws, silently.

    The `IndexError` that motivated the old refusal (1 history zipped against
    60 run lengths) is still the thing to guard: driving a real frame below
    is what would raise it.

    Runs are two observations long on purpose: single-observation runs raise
    a SEPARATE "a pure line format cannot render a single point" warning."""
    data = _series(n=1, rows=60)
    labels = np.array(['a', 'a', 'b', 'b'] * 15)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(data, '-', predict='Kalman', t=3, hue=labels,
                            animate=True, duration=2, frame_rate=4,
                            show=False)
    assert not caught, [str(w.message) for w in caught]
    ax = _ax(fig)
    ani._func(6, *ani._args)          # must not raise
    assert [a for a in _forecasts(ax, role='live') if a.get_visible()]

    # ...and the STATIC plot of the same data draws them too -- the two
    # paths agree again, which is what this test now pins
    static = hyp.plot(data, '-', predict='Kalman', t=3, hue=labels,
                      show=False)
    assert [ln for ln in static.axes[0].lines
            if getattr(ln, '_hyp_forecast_role', None) == 'static']


def test_the_plotly_backend_DRAWS_the_same_case_the_same_way():
    """Backend parity, in its new direction.

    This used to pin that plotly REFUSED the case matplotlib refused (it had
    warned "no forecast is drawn" and then drawn two, showing the final
    forecast from frame 0, because its static block fires whenever there is
    no schedule). Now that a schedule exists for regrouped animations, parity
    means plotly draws per-frame forecast traces too -- and its animated
    trace loop had the SAME per-trace-vs-per-dataset defect the matplotlib
    side did, raising IndexError from `_forecast_frame_data` on the first
    frame until it was fixed to loop over the forecasts."""
    data = _series(n=1, rows=60)
    labels = np.array(['a', 'a', 'b', 'b'] * 15)
    with hyp.set_interactive_backend('plotly'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig = hyp.plot(data, '-', predict='Kalman', t=3, hue=labels,
                           animate=True, duration=2, frame_rate=4,
                           show=False)
    assert not caught, [str(w.message) for w in caught]
    roles = [(tr.meta or {}).get('hyp_forecast_role') for tr in fig.data]
    assert 'live' in [r for r in roles if r is not None], roles
    # one live forecast trace per DATASET, not per drawn run
    assert len([r for r in roles if r == 'live']) == len(data)


def test_a_refused_forecast_is_still_REPORTED_even_though_it_is_not_drawn():
    """`return_model=True` describes MODEL output. The fit succeeded; only
    the rendering combination is unsupported, so throwing the arrays away
    would lose a valid result to a drawing limitation. The bundle says both
    things instead: the forecasts, and that they were not drawn."""
    # MARKER-only categorical regrouping is what is still refused: it groups
    # globally by category through `reshape_data`, so 3 datasets under 2
    # categories become 2 traces that are not datasets at all. (The LINE
    # regrouping this test used to use now draws -- see the test above.)
    data = _series(n=3, rows=20)
    labels = np.array(['a'] * 10 + ['b'] * 10)
    labels = np.tile(labels, 3)
    with pytest.warns(UserWarning, match='no per-dataset trace'):
        bundle = hyp.plot(data, 'o', predict='Kalman', t=3, hue=labels,
                          animate=True, duration=2, frame_rate=4,
                          show=False, return_model=True)
    reported = bundle['predict']
    assert reported is not None
    assert len(reported['forecasts']) == 3
    # t=3 forecast rows, in the REDUCED (3-D) analyze space
    assert np.asarray(reported['forecasts'][0]).shape == (3, 3)
    assert reported['drawn'] is False
    assert reported['draw_reason'] and 'hue' in reported['draw_reason']

    # and when they ARE drawn, the bundle says so without a reason to give
    drawn = hyp.plot(data, '-', predict='Kalman', t=3, show=False,
                     return_model=True)
    assert drawn['predict']['drawn'] is True
    assert drawn['predict']['draw_reason'] is None


def test_bundle_forecasts_are_the_full_history_forecast():
    """Unchanged from static/spin: exactly t rows, analyze space, one per
    input dataset (plot.py:2289-2295)."""
    data = _series(n=2)
    out = hyp.plot(data, '-', predict='Kalman', t=4, animate=True,
                   duration=2, frame_rate=4, show=False, return_model=True)
    assert out['animation'] is not None
    assert out['predict']['model'] == 'Kalman'
    assert out['predict']['params'] == {'t': 4}
    forecasts = out['predict']['forecasts']
    assert len(forecasts) == 2
    for fc in forecasts:
        assert np.asarray(fc).shape == (4, 3)


def test_bundle_forecast_matches_hyp_predict_on_the_returned_xform_data():
    """Contract 7: the bundle stays interchangeable with hyp.predict, exactly
    as the static path promises and test_predict_return_model_bundle pins."""
    out = hyp.plot(_series(n=1), '-', predict='Kalman', t=4, animate=True,
                   duration=2, frame_rate=4, show=False, return_model=True)
    direct = np.asarray(hyp.predict(np.asarray(out['xform_data'][0]),
                                    model='Kalman', t=4), dtype=float)
    assert np.allclose(np.asarray(out['predict']['forecasts'][0]), direct,
                       rtol=1e-6, atol=1e-6)


def test_the_final_frame_draws_exactly_the_bundled_forecast():
    """The final frame reveals the whole history, so the drawn per-frame
    forecast IS the bundled full-history one -- which is why the bundle needs
    no redefinition for animated plots."""
    out = hyp.plot(_series(n=1), '-', predict='Kalman', t=4, animate=True,
                   antialias=False, duration=4, frame_rate=4, show=False,
                   return_model=True)
    fig, ani = out['fig'], out['animation']
    ani._func(15, *ani._args)
    ax = _ax(fig)
    drawn = np.array(_forecasts(ax, role='live')[0].get_data_3d()).T
    # t + 1 vertices: the anchor plus t forecast steps
    assert drawn.shape == (5, 3)
    # and the t forecast steps advance in the same directions as the bundle
    bundled = np.asarray(out['predict']['forecasts'][0], dtype=float)
    assert np.allclose(np.sign(np.diff(drawn[1:], axis=0)),
                       np.sign(np.diff(bundled, axis=0)))


def test_return_model_xform_data_is_untouched_by_the_schedule():
    """The schedule snapshots analyze-space copies; it must not alias or
    mutate what the user gets back.

    Compare VALUES, not shapes. An earlier version of this test asserted
    only that the two `xform_data` arrays had the same SHAPE -- which every
    mutation in place also satisfies, since mutating an array does not
    resize it. It could not detect the defect named in its own docstring.
    """
    plain = hyp.plot(_series(n=1), '-', animate=True, duration=2,
                     frame_rate=4, show=False, return_model=True)
    forecast = hyp.plot(_series(n=1), '-', predict='Kalman', t=3,
                        animate=True, duration=2, frame_rate=4, show=False,
                        return_model=True)
    a = np.asarray(plain['xform_data'][0], dtype=float)
    b = np.asarray(forecast['xform_data'][0], dtype=float)
    assert a.shape == b.shape
    assert np.allclose(a, b), (
        'predict= changed the returned xform_data; the schedule must take '
        'its own copies (np.array(..., copy=True)) and never write back')
    # ...and driving the animation must not mutate it either: the updater
    # READS the schedule, so a frame render cannot move the user's data
    before = np.array(forecast['xform_data'][0], dtype=float, copy=True)
    ani = forecast['animation']
    for f in (0, 4, 7):
        ani._func(f, *ani._args)
    assert np.allclose(np.asarray(forecast['xform_data'][0], dtype=float),
                       before), 'rendering frames mutated the returned data'


def test_a_SINGLE_FRAME_animation_forecasts_from_the_whole_trajectory():
    """The schedule's clock is the RENDERER's clock, at every frame count.

    `plot()` floors the interpolation grid at 2 rows because PCHIP needs two
    samples; it used to floor the forecast SCHEDULE's frame count with the
    same expression, while both backends pace with `max(1, ...)`. The two
    differ at exactly one setting -- `round(frame_rate * duration) == 1` --
    and there the renderer draws its single frame holding the WHOLE
    trajectory while a 2-frame schedule reports one row revealed, so the
    finished animation showed all the data and no forecast at all. Measured
    before the fix on this fixture: renderer 8 raw rows, schedule 1.
    """
    data = np.random.default_rng(0).normal(size=(8, 3)).cumsum(axis=0)
    fig, ani = hyp.plot([data], '-', predict='Kalman', t=3, animate=True,
                        duration=1, frame_rate=1, show=False)
    ani._func(0, *ani._args)
    drawn = _forecasts(_ax(fig), role='live')
    assert drawn, 'the only frame shows the whole trajectory, so it must ' \
                  'also show the forecast that trajectory implies'
    # `get_data_3d`, not `get_xdata`: a Line3D keeps its vertices in
    # `_verts3d`, and `get_xdata()` reports the stale 2-D projection
    pts = np.array(drawn[0].get_data_3d())
    assert pts.shape[1] > 1, pts.shape
