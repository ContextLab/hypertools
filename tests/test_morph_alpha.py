"""GH #284: ``alpha=`` reaches the ``animate='morph'`` traveling cloud.

Before the fix, ``hyp.plot(..., animate='morph', alpha=0.25)`` set the alpha
on the per-dataset line artists -- which the morph keeps hidden for the
whole animation -- and never on the ONE artist that is actually drawn (the
traveling point cloud), so ``examples/animate_surface_morph.py`` had to
reach into the figure with ``ax.get_lines()[-1].set_alpha(0.25)``.

The rule (``hypertools.plot.morph.morph_alpha``, shared by both backends):
a HOLD draws the held dataset's own alpha; a TRANSITION eases (smoothstep,
on the same schedule as the colour lerp) from the departing dataset's alpha
to the arriving one's -- so its first frame is the departing dataset's
alpha and its last is the arriving dataset's. A scalar ``alpha=`` gives
every dataset the same value, so the cloud is constant. When no alpha was
asked for at all, the artist is left at matplotlib's default (``None``) --
nothing about a default morph changes.

Every test here renders a real figure (no mocks); the raster tests draw
through a real Agg canvas.
"""

import re

import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot import morph


def _blobs(n=30, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, 3)) + 6.0 * i for i in range(k)]


def _mpl(data, **kwargs):
    kwargs.setdefault('duration', 2)
    kwargs.setdefault('frame_rate', 10)
    return hyp.plot(data, '.', animate='morph', show=False, **kwargs)


def _agg_rgba(fig):
    """Rasterize `fig` on a real Agg canvas and return its RGBA buffer."""
    if not hasattr(fig.canvas, 'buffer_rgba'):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _segment_frames(frame_counts):
    """(first frame of hold_1, first/mid/last frame of morph_1->2, first
    frame of hold_2) for a schedule `frame_counts`."""
    h0 = 0
    t_first = frame_counts[0]
    t_last = frame_counts[0] + frame_counts[1] - 1
    t_mid = frame_counts[0] + frame_counts[1] // 2
    h1 = frame_counts[0] + frame_counts[1]
    return h0, t_first, t_mid, t_last, h1


# ---------------------------------------------------------------------------
# the shared rule
# ---------------------------------------------------------------------------

class TestMorphAlphaRule:
    def test_all_none_returns_none(self):
        assert morph.morph_alpha([None, None], 0, 0, 5) is None
        assert morph.morph_alpha([None, None], 1, 2, 5) is None
        assert morph.morph_alpha(None, 1, 2, 5) is None

    def test_hold_is_held_datasets_alpha(self):
        assert morph.morph_alpha([0.2, 0.6, 1.0], 0, 3, 5) == 0.2
        assert morph.morph_alpha([0.2, 0.6, 1.0], 2, 0, 5) == 0.6
        assert morph.morph_alpha([0.2, 0.6, 1.0], 4, 4, 5) == 1.0

    def test_transition_eases_departing_to_arriving(self):
        alphas = [0.2, 0.6]
        assert morph.morph_alpha(alphas, 1, 0, 5) == pytest.approx(0.2)
        assert morph.morph_alpha(alphas, 1, 4, 5) == pytest.approx(0.6)
        t = float(morph.smoothstep(2 / 4))
        assert morph.morph_alpha(alphas, 1, 2, 5) == pytest.approx(
            0.2 + t * 0.4)

    def test_unset_entry_counts_as_opaque_when_others_are_set(self):
        assert morph.morph_alpha([0.3, None], 2, 0, 5) == 1.0
        assert morph.morph_alpha([0.3, None], 1, 4, 5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# matplotlib backend
# ---------------------------------------------------------------------------

class TestMatplotlibMorphAlpha:
    def test_scalar_alpha_lands_on_the_cloud_every_frame(self):
        fig, ani = _mpl(_blobs(), alpha=0.25)
        state = ani._args[0]
        artist = state['artist']
        assert artist.get_alpha() == 0.25
        for k in range(sum(state['frame_counts'])):
            ani._func(k, *ani._args)
            assert artist.get_alpha() == 0.25, f"frame {k}"

    def test_scalar_alpha_2d(self):
        data = [d[:, :2] for d in _blobs()]
        fig, ani = _mpl(data, alpha=0.4)
        state = ani._args[0]
        artist = state['artist']
        assert artist.get_alpha() == 0.4
        h0, t_first, t_mid, t_last, h1 = _segment_frames(
            state['frame_counts'])
        for k in (h0, t_first, t_mid, t_last, h1):
            ani._func(k, *ani._args)
            assert artist.get_alpha() == 0.4

    def test_per_dataset_list_follows_the_hold_and_departing_rule(self):
        alphas = [0.2, 0.6, 1.0]
        fig, ani = _mpl(_blobs(), alpha=alphas)
        state = ani._args[0]
        artist = state['artist']
        fc = state['frame_counts']
        h0, t_first, t_mid, t_last, h1 = _segment_frames(fc)

        ani._func(h0, *ani._args)
        assert artist.get_alpha() == 0.2          # hold on dataset 0
        ani._func(t_first, *ani._args)
        assert artist.get_alpha() == pytest.approx(0.2)  # departing
        ani._func(t_mid, *ani._args)
        step = t_mid - fc[0]
        expect = morph.morph_alpha(alphas, 1, step, fc[1])
        assert 0.2 < expect < 0.6
        assert artist.get_alpha() == pytest.approx(expect)
        ani._func(t_last, *ani._args)
        assert artist.get_alpha() == pytest.approx(0.6)  # arriving
        ani._func(h1, *ani._args)
        assert artist.get_alpha() == 0.6          # hold on dataset 1
        ani._func(sum(fc) - 1, *ani._args)
        assert artist.get_alpha() == 1.0          # final hold, dataset 2

    def test_default_alpha_unchanged(self):
        """Regression: no `alpha=` -> the artist stays at matplotlib's
        default (`None`) at construction and after every frame update."""
        fig, ani = _mpl(_blobs())
        state = ani._args[0]
        artist = state['artist']
        assert state['alphas'] == [None, None, None]
        assert artist.get_alpha() is None
        for k in range(sum(state['frame_counts'])):
            ani._func(k, *ani._args)
            assert artist.get_alpha() is None, f"frame {k}"

    def test_alpha_kwarg_renders_like_manual_set_alpha(self):
        """Real render: `alpha=0.25` on the call must rasterize a mid-
        transition frame IDENTICALLY to the pre-fix workaround (default
        call + `set_alpha(0.25)` on the visible cloud), and differently
        from the default (opaque) render."""
        data = _blobs(seed=4)
        fig_a, ani_a = _mpl(data, alpha=0.25, markersize=8)
        fig_b, ani_b = _mpl(data, markersize=8)
        fig_c, ani_c = _mpl(data, markersize=8)
        fc = ani_a._args[0]['frame_counts']
        assert fc == ani_b._args[0]['frame_counts']
        mid = fc[0] + fc[1] // 2

        ani_b._args[0]['artist'].set_alpha(0.25)   # the old workaround
        for ani in (ani_a, ani_b, ani_c):
            ani._func(mid, *ani._args)
        rgba_a = _agg_rgba(fig_a)
        rgba_b = _agg_rgba(fig_b)
        rgba_c = _agg_rgba(fig_c)
        assert rgba_a.shape == rgba_b.shape == rgba_c.shape
        assert np.array_equal(rgba_a, rgba_b)
        assert not np.array_equal(rgba_a, rgba_c)

    def test_alpha_with_surface_and_keep_points(self):
        """The example's configuration: a lit hull plus a faint point
        texture -- the cloud (not the mesh) takes the alpha."""
        spec = {'alpha': 0.97, 'color': '#2E86AB', 'smoothing': 1,
                'keep_points': True}
        fig, ani = _mpl(_blobs(n=40, seed=5), alpha=0.25, color='k',
                        surface=spec)
        state = ani._args[0]
        assert state['artist'].get_alpha() == 0.25
        ani._func(state['frame_counts'][0] + 1, *ani._args)
        assert state['artist'].get_alpha() == 0.25
        assert state['artist'].get_visible()


# ---------------------------------------------------------------------------
# plotly backend
# ---------------------------------------------------------------------------

_RGBA = re.compile(r'rgba\((\d+),(\d+),(\d+),([0-9.]+)\)')


def _plotly(data, **kwargs):
    kwargs.setdefault('duration', 2)
    kwargs.setdefault('frame_rate', 10)
    return hyp.plot(data, '.', animate='morph', backend='plotly',
                    show=False, **kwargs)


def _alpha_of(color_str):
    m = _RGBA.fullmatch(color_str)
    assert m, f"expected an rgba(...) colour, got {color_str!r}"
    return float(m.group(4))


class TestPlotlyMorphAlpha:
    def test_scalar_alpha_on_initial_trace_and_every_frame(self):
        fig = _plotly(_blobs(), alpha=0.25)
        morph_idx = fig.frames[0].traces[0]
        assert _alpha_of(fig.data[morph_idx].marker.color) == 0.25
        for k, frame in enumerate(fig.frames):
            assert _alpha_of(frame.data[0].marker.color) == 0.25, f"frame {k}"

    def test_scalar_alpha_2d(self):
        data = [d[:, :2] for d in _blobs()]
        fig = _plotly(data, alpha=0.4)
        morph_idx = fig.frames[0].traces[0]
        assert _alpha_of(fig.data[morph_idx].marker.color) == 0.4
        for frame in fig.frames:
            assert _alpha_of(frame.data[0].marker.color) == 0.4

    def test_per_dataset_list_follows_the_hold_and_departing_rule(self):
        alphas = [0.2, 0.6, 1.0]
        fig = _plotly(_blobs(), alpha=alphas)
        fc, _, _ = morph.morph_schedule(3, len(fig.frames), 1, -60)
        h0, t_first, t_mid, t_last, h1 = _segment_frames(fc)
        col = lambda k: fig.frames[k].data[0].marker.color  # noqa: E731
        assert _alpha_of(col(h0)) == 0.2
        assert _alpha_of(col(t_first)) == pytest.approx(0.2)
        expect = morph.morph_alpha(alphas, 1, t_mid - fc[0], fc[1])
        assert 0.2 < expect < 0.6
        assert _alpha_of(col(t_mid)) == pytest.approx(expect)
        assert _alpha_of(col(t_last)) == pytest.approx(0.6)
        assert _alpha_of(col(h1)) == 0.6
        assert _alpha_of(col(sum(fc) - 1)) == 1.0

    def test_default_alpha_unchanged(self):
        """Regression: no `alpha=` -> the plain opaque `rgb(...)` colour
        string, exactly as before, on the initial trace and every frame."""
        fig = _plotly(_blobs())
        morph_idx = fig.frames[0].traces[0]
        assert fig.data[morph_idx].marker.color.startswith('rgb(')
        for frame in fig.frames:
            assert frame.data[0].marker.color.startswith('rgb(')

    def test_surface_mesh_keeps_its_own_alpha(self):
        """`alpha=` fades the CLOUD only: the hull mesh's opacity comes
        from `surface['alpha']` and is identical with and without it."""
        spec = {'alpha': 0.97, 'color': '#2E86AB', 'smoothing': 1,
                'keep_points': True}
        data = _blobs(n=40, seed=5)
        fig = _plotly(data, alpha=0.25, color='k', surface=spec)
        ref = _plotly(data, color='k', surface=spec)
        cloud_idx, mesh_idx = fig.frames[0].traces
        assert list(ref.frames[0].traces) == [cloud_idx, mesh_idx]
        assert _alpha_of(fig.data[cloud_idx].marker.color) == 0.25
        assert ref.data[cloud_idx].marker.color.startswith('rgb(')
        assert fig.data[mesh_idx].opacity == ref.data[mesh_idx].opacity
        for frame, rframe in zip(fig.frames, ref.frames):
            assert _alpha_of(frame.data[0].marker.color) == 0.25
            assert frame.data[1].opacity == rframe.data[1].opacity
