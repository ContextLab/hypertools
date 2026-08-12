"""Forecasts over an animation whose data hue=/cluster= regrouped."""
import contextlib
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp

HUE = ['A'] * 10 + ['B'] * 10 + ['A'] * 10


@contextlib.contextmanager
def no_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield caught
    assert not caught, [str(w.message) for w in caught]


def _walks(n=2, rows=30, seed=0):
    rng = np.random.RandomState(seed)
    return [np.cumsum(rng.randn(rows, 3), 0) for _ in range(n)]


def _artists(fig, role=None):
    return [ln for ln in fig.axes[0].lines
            if getattr(ln, '_hyp_forecast_role', None) is not None
            and (role is None or ln._hyp_forecast_role == role)]


def _endpoint(artist):
    """The last DRAWN vertex of a forecast artist.

    `get_data_3d()`, not `get_xdata()`: a `Line3D` keeps its vertices in
    `_verts3d`, and `get_xdata()` reports a stale 2-D projection -- empty
    for an artist only ever filled through `set_data_3d`, which reads as
    "no forecast" no matter what is on screen.
    """
    return np.asarray(artist.get_data_3d())[:, -1]


def _rgb(artist):
    return tuple(np.round(matplotlib.colors.to_rgb(artist.get_color()), 5))


def _animate(data=None, hue=None, **kwargs):
    data = _walks() if data is None else data
    hue = (HUE * len(data)) if hue is None else hue
    with no_warnings():
        return hyp.plot(data, '-', hue=hue, predict='Kalman', t=4,
                        animate=True, duration=2, frame_rate=6, show=False,
                        **kwargs)


def test_a_regrouped_animation_now_DRAWS_its_forecasts():
    fig, ani = _animate()
    ani._func(11, *ani._args)
    live = [a for a in _artists(fig, 'live') if a.get_visible()]
    assert live, 'no live forecast artist was drawn'


def test_it_no_longer_warns_that_it_cannot_draw_them():
    """`no_warnings()` already asserts this, but name it: the refusal warning
    disappearing is the observable half of the feature."""
    fig, ani = _animate()
    assert _artists(fig, 'live')


def test_the_bundle_reports_drawn_True():
    with no_warnings():
        out = hyp.plot(_walks(), '-', hue=HUE * 2, predict='Kalman', t=4,
                       animate=True, duration=2, frame_rate=6, show=False,
                       return_model=True)
    info = out['predict']
    assert info['drawn'] is True
    assert info['draw_reason'] is None


def _displacement(artist):
    """Anchor -> tip of a drawn forecast, in its own figure's coordinates."""
    pts = np.asarray(artist.get_data_3d())
    return pts[:, -1] - pts[:, 0]


def test_the_final_frame_forecast_equals_the_STATIC_one():
    """The animation's last frame has the whole history, so its forecast must
    be the one a static plot of the same data draws -- otherwise the animated
    and static paths disagree about the same model on the same rows.

    Compared as DIRECTION plus a single shared magnitude ratio, because raw
    display coordinates legitimately differ: `plot()` builds the [-1, 1] box
    from everything it will draw, and an animation draws EVERY frame's
    forecast while a static plot draws only the final one, so the two figures
    have different centre/scale statistics. Measured on this fixture, the
    observed DATA endpoints differ between the figures too, on identical
    data -- an equality of coordinates would therefore be testing the box,
    not the forecast. The transform is isotropic (one scalar `m2` for every
    dimension), so a shared ratio across datasets is exactly the statement
    "same forecast, different zoom": measured 0.85883598 for both.
    """
    data = _walks()
    fig, ani = _animate(data)
    with no_warnings():
        sfig = hyp.plot(data, '-', hue=HUE * 2, predict='Kalman', t=4,
                        show=False)
    ani._func(11, *ani._args)
    live = [_displacement(a) for a in _artists(fig, 'live')
            if a.get_visible()]
    static = [_displacement(a) for a in _artists(sfig, 'static')]
    assert live and len(live) == len(static)
    ratios = []
    for a, b in zip(live, static):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        assert na > 0 and nb > 0
        assert np.allclose(a / na, b / nb, atol=1e-8), (a, b)
        ratios.append(na / nb)
    assert np.allclose(ratios, ratios[0], atol=1e-9), ratios


def test_an_EARLY_forecast_is_not_the_full_history_one():
    """A smoke test only -- the exact 'fit on precisely these rows' assertion
    lives in `test_for_regrouped_fits_EXACTLY_the_visible_rows`, where the
    expected fit can be computed directly. Here it guards the WIRING: that
    `plot()` handed the animation the reveal schedule and not the static
    forecast."""
    fig, ani = _animate()
    ani._func(2, *ani._args)
    early = [tuple(_endpoint(a)) for a in _artists(fig, 'live')
             if a.get_visible()]
    ani._func(11, *ani._args)
    late = [tuple(_endpoint(a)) for a in _artists(fig, 'live')
            if a.get_visible()]
    assert early and late
    assert not np.allclose(sorted(early), sorted(late))


def test_the_live_forecast_takes_the_HEAD_run_colour():
    """Decision R3: the forecast continues the head, so it wears that run's
    colour and changes with it at a category boundary."""
    fig, ani = _animate(_walks(n=1), hue=HUE)
    seen = set()
    for frame in range(12):
        ani._func(frame, *ani._args)
        seen.update(_rgb(a) for a in _artists(fig, 'live')
                    if a.get_visible())
    assert len(seen) > 1, 'the forecast kept one colour across a boundary'


def test_a_RETAINED_trail_keeps_the_colour_it_was_FIT_with():
    """Decision R3's second half. A fan member drawn before the boundary must
    stay in category A when the live forecast moves to B; repainting the whole
    fan would make a saved animation differ from a played one."""
    fig, ani = _animate(_walks(n=1), hue=HUE, forecast_trail=8)
    boundary = None
    prev = None
    for frame in range(12):
        ani._func(frame, *ani._args)
        live = [a for a in _artists(fig, 'live') if a.get_visible()]
        if not live:
            continue
        now = _rgb(live[0])
        if prev is not None and now != prev:
            boundary = frame
            break
        prev = now
    assert boundary is not None, 'no category boundary was crossed'
    ani._func(boundary, *ani._args)
    fan = sorted((a._hyp_forecast_age, _rgb(a)) for a in _artists(fig, 'trail')
                 if a.get_visible())
    assert fan, 'no retained forecast was drawn at the boundary'
    assert len({c for _, c in fan}) > 1 or fan[0][1] == prev, (
        f'the fan was repainted to the new category: {fan}')


def test_replaying_frames_does_not_MUTATE_the_fan():
    """`save()`/`to_jshtml()` replay from 0 and may deliver frames out of
    order; a colour that depended on frame HISTORY would differ between a
    saved and a played animation."""
    fig, ani = _animate(_walks(n=1), hue=HUE, forecast_trail=8)

    def snapshot(frame):
        ani._func(frame, *ani._args)
        return sorted((a._hyp_forecast_age, _rgb(a),
                       np.asarray(a.get_data_3d()).shape[1])
                      for a in _artists(fig, 'trail') if a.get_visible())
    first = snapshot(9)
    for frame in (0, 4, 11, 2, 7):
        snapshot(frame)
    assert snapshot(9) == first


def test_forecast_cluster_still_holds_ONE_colour_across_frames():
    """The override path is unchanged by Decision R3: an explicit grouping is
    resolved once from the full-history forecasts and fixed for every frame."""
    fig, ani = _animate(_walks(n=4, rows=20),
                        hue=(['A'] * 10 + ['B'] * 10) * 4,
                        forecast_cluster='KMeans', forecast_n_clusters=2)
    seen = {}
    for frame in (0, 4, 9, 11, 2, 11, 7):
        ani._func(frame, *ani._args)
        for a in _artists(fig, 'live'):
            if a.get_visible():
                seen.setdefault(id(a), set()).add(_rgb(a))
    assert seen
    assert all(len(v) == 1 for v in seen.values()), seen


def test_MARKER_only_regrouping_still_refuses_and_says_so():
    """The other refusal is untouched, and this is what actually reaches it.

    Marker-only categorical regrouping goes through `reshape_data`, which
    groups GLOBALLY by category: 3 datasets under 2 categories become 2
    traces that are not datasets at all, so there is no per-dataset trace to
    anchor a forecast to. Measured, because the plan (and a comment in
    `plot.py`) named a CONTINUOUS hue as the example and that is not one --
    see the test below.
    """
    rng = np.random.RandomState(0)
    data = [np.cumsum(rng.randn(20, 3), 0) for _ in range(3)]
    with pytest.warns(UserWarning, match='no per-dataset trace'):
        hyp.plot(data, 'o', hue=(['A'] * 10 + ['B'] * 10) * 3,
                 predict='Kalman', t=4, animate=True, duration=2,
                 frame_rate=6, show=False)


def test_a_CONTINUOUS_hue_DRAWS_its_forecasts_and_never_refused():
    """A continuous hue colours ONE line artist per dataset through a
    `LineCollection` overlay; it does not change the trace count, so the
    per-dataset correspondence holds and the forecast draws. Pinned because
    the plan assumed the opposite, and an assumption that a working case
    refuses is how a real refusal gets written for the wrong reason.
    """
    with no_warnings():
        fig, ani = hyp.plot(_walks(n=1), '-', hue=list(range(30)),
                            predict='Kalman', t=4, animate=True, duration=2,
                            frame_rate=6, show=False)
    ani._func(11, *ani._args)
    assert [a for a in _artists(fig, 'live') if a.get_visible()]
