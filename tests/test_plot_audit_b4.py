# -*- coding: utf-8 -*-
"""Regression tests for release-1.0 audit batch B4 (unit
F07-plot-density-surface): the ``surface=`` / ``density=`` plot() kwargs.

Findings covered (all CONFIRMED by the independent verifier):

- F07-001 (major): the plotly backend NaN'd out every data point enclosed
  by its own surface hull regardless of ``surface['alpha']``, so translucent
  plotly surfaces showed no data at all (matplotlib, the reference behavior,
  shows all points through a translucent hull). Fixed: surfaces with
  ``alpha < 0.999`` are now rendered genuinely translucent (real Mesh3d
  opacity) and keep every data point; only a fully-opaque surface
  (``alpha >= 0.999``) hides the points it encloses (they would be invisible
  behind it anyway, and hiding them avoids plotly's WebGL punch-through
  defect).
- F07-002 (minor): ``density=`` on coplanar/collinear data silently drew
  nothing instead of the docstring-promised UserWarning (fixed by the
  rank check in ``fit_kde``; these tests pin the contract).
- F07-005 (minor): animated (spin/parallel/serial) surfaces dropped the
  per-vertex hue coloring static plots use, rendering a hue'd hull flat
  gray. Fixed on both backends.
- F07-006 (minor): ``surface=`` dict values were never validated eagerly --
  bad ``alpha`` crashed late and cryptically inside matplotlib,
  ``pre_inflate=0`` leaked a raw QhullError, ``pre_inflate<0`` silently
  built an inside-out mesh. Fixed: eager ValueErrors mirroring density='s
  validation.
- F07-008 (style): numpy bools (``np.True_``/``np.False_``) were rejected
  for the top-level ``surface=``/``density=`` while ``per_group`` accepted
  them. Fixed: accepted everywhere a Python bool is.

No mocks -- real data, real mesh/KDE computation, seeded RNGs, Agg backend.
"""
import warnings

import matplotlib as mpl
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import hypertools as hyp
from hypertools.plot.meshutil import smooth_hull_3d

mpl.rcParams['figure.max_open_warning'] = 40


def _blob_3d(n=80, seed=0, center=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) + np.asarray(center)


def _two_blobs_3d(n=150):
    rng = np.random.default_rng(42)
    return [rng.normal(size=(n, 3)),
            rng.normal(size=(n, 3)) + np.array([4.0, 0.0, 0.0])]


def _hue_helix(n=200):
    t = np.linspace(0, 4 * np.pi, n)
    traj = np.column_stack([np.cos(t), np.sin(t), t / (4 * np.pi)])
    return traj, t


def _scatter3d_nonnan_counts(fig, n):
    return [int(np.sum(~np.isnan(np.asarray(t.x, dtype=float))))
            for t in fig.data
            if t.type == 'scatter3d' and t.x is not None and len(t.x) == n]


def _parse_rgb_strings(strings):
    return np.array([[float(v) for v in c[4:-1].split(',')] for c in strings])


def _largest_hull_facecolors(fig):
    ax = fig.axes[0]
    hulls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    assert hulls, 'no Poly3DCollection (surface hull) found'
    hull = max(hulls, key=lambda c: len(c.get_facecolor()))
    return np.asarray(hull.get_facecolor())[:, :3]


class TestPlotlyTranslucentSurfaceKeepsPoints:
    """F07-001: translucent plotly surfaces must show the data through the
    hull, exactly like the matplotlib reference behavior."""

    @pytest.mark.parametrize('surface_arg', [True, {'alpha': 0.2}])
    def test_translucent_surface_keeps_every_point(self, surface_arg):
        data = _two_blobs_3d()
        fig = hyp.plot(data, '.', surface=surface_arg, backend='plotly',
                       show=False)
        counts = _scatter3d_nonnan_counts(fig, len(data[0]))
        assert counts == [len(data[0]), len(data[1])], (
            'translucent plotly surfaces must not hide (NaN out) the data '
            f'points they enclose; got non-NaN counts {counts}')

    def test_translucent_mesh_has_real_opacity(self):
        alpha = 0.6
        fig = hyp.plot(_two_blobs_3d(), '.', surface={'alpha': alpha},
                       backend='plotly', show=False)
        meshes = [t for t in fig.data if t.type == 'mesh3d']
        assert len(meshes) == 2
        # every face is emitted twice (double winding), so each layer gets
        # 1 - sqrt(1 - alpha), compositing to exactly `alpha` total
        expected = 1.0 - np.sqrt(1.0 - alpha)
        for m in meshes:
            assert m.opacity == pytest.approx(expected, abs=1e-9)

    def test_opaque_surface_still_hides_enclosed_points(self):
        """alpha=1.0 keeps the historical artifact-free opaque path: the
        mesh is fully opaque and points it encloses are hidden (NaN'd) --
        they would be invisible behind it anyway, and drawing them would
        punch WebGL holes through the mesh."""
        data = _two_blobs_3d()
        fig = hyp.plot(data, '.', surface={'alpha': 1.0}, backend='plotly',
                       show=False)
        meshes = [t for t in fig.data if t.type == 'mesh3d']
        assert all(m.opacity == 1.0 for m in meshes)
        counts = _scatter3d_nonnan_counts(fig, len(data[0]))
        assert all(c < len(data[0]) for c in counts)

    def test_translucent_vertexcolor_not_whitened(self):
        """The translucent mesh must shade the dataset's TRUE base color
        (real alpha compositing handles the lightening, as in matplotlib) --
        not the toward-white pre-blended color the opaque path bakes in.
        With identical geometry, the translucent trace's vertexcolor must
        therefore match the alpha=1.0 trace's vertexcolor exactly."""
        pts = _blob_3d(n=100, seed=3)
        fig_translucent = hyp.plot(pts, '.', surface={'alpha': 0.6},
                                   backend='plotly', show=False)
        fig_opaque = hyp.plot(pts, '.', surface={'alpha': 1.0},
                              backend='plotly', show=False)
        vc_t = _parse_rgb_strings(
            [t for t in fig_translucent.data if t.type == 'mesh3d'][0].vertexcolor)
        vc_o = _parse_rgb_strings(
            [t for t in fig_opaque.data if t.type == 'mesh3d'][0].vertexcolor)
        assert vc_t.shape == vc_o.shape
        assert np.allclose(vc_t, vc_o, atol=1.0)


class TestDensityDegenerateWarns:
    """F07-002: the docstring-promised UserWarning must fire for
    coplanar/collinear (rank-deficient) data instead of silently drawing
    nothing."""

    def test_coplanar_3d_density_warns_and_skips(self):
        rng = np.random.default_rng(3)
        data = np.column_stack([rng.normal(size=(80, 2)), np.zeros(80)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = hyp.plot(data, '.', density=True, reduce=None, show=False)
        msgs = [str(x.message) for x in w]
        assert any('density' in m and 'degenerate' in m for m in msgs), msgs
        # nothing beyond the cube edges + data line should have been drawn
        ax = fig.axes[0]
        assert not [c for c in ax.collections if isinstance(c, Poly3DCollection)]

    def test_collinear_2d_density_warns_and_skips(self):
        x = np.linspace(0.0, 1.0, 60)
        data = np.column_stack([x, 2.0 * x])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = hyp.plot(data, '.', density=True, reduce=None, show=False)
        msgs = [str(x.message) for x in w]
        assert any('density' in m and 'degenerate' in m for m in msgs), msgs
        assert not fig.axes[0].images  # no imshow density layer

    def test_plotly_coplanar_3d_density_warns(self):
        rng = np.random.default_rng(3)
        data = np.column_stack([rng.normal(size=(80, 2)), np.zeros(80)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = hyp.plot(data, '.', density=True, reduce=None,
                           backend='plotly', show=False)
        msgs = [str(x.message) for x in w]
        assert any('density' in m and 'degenerate' in m for m in msgs), msgs
        assert not [t for t in fig.data if t.type == 'volume']


class TestAnimatedSurfaceKeepsHue:
    """F07-005: animated surfaces must keep the per-vertex hue coloring
    static plots use (a rainbow-hue hull must not render flat gray)."""

    CHROMA = 0.25  # max |r-g| across faces; flat-base before-fix was ~0.11

    def test_mpl_spin_hue_surface_is_chromatic(self):
        traj, t = _hue_helix()
        res = hyp.plot(traj, '.', hue=t, animate='spin',
                       surface={'alpha': 0.5}, frame_rate=5, duration=2,
                       return_model=True, show=False)
        ani = res['animation']
        ani._init_draw()
        ani._draw_frame(next(iter(ani.new_frame_seq())))
        fc = _largest_hull_facecolors(res['fig'])
        assert np.abs(fc[:, 0] - fc[:, 1]).max() > self.CHROMA, (
            'animated (spin) hue surface rendered (near-)achromatic: '
            f'max |r-g| = {np.abs(fc[:, 0] - fc[:, 1]).max():.3f}')

    def test_mpl_parallel_hue_surface_is_chromatic(self):
        traj, t = _hue_helix()
        res = hyp.plot(traj, '.', hue=t, animate=True,
                       surface={'alpha': 0.5}, frame_rate=5, duration=2,
                       return_model=True, show=False)
        ani = res['animation']
        ani._init_draw()
        frames = list(ani.new_frame_seq())
        for frame in frames[:max(2, len(frames) // 2)]:
            ani._draw_frame(frame)
        fc = _largest_hull_facecolors(res['fig'])
        assert np.abs(fc[:, 0] - fc[:, 1]).max() > self.CHROMA

    @pytest.mark.parametrize('mode', ['spin', True])
    def test_plotly_animated_hue_surface_is_chromatic(self, mode):
        traj, t = _hue_helix()
        fig = hyp.plot(traj, '.', hue=t, animate=mode,
                       surface={'alpha': 0.5}, frame_rate=2, duration=2,
                       backend='plotly', show=False)
        frame = fig.frames[len(fig.frames) // 2]
        meshes = [d for d in frame.data
                  if d.type == 'mesh3d' and d.vertexcolor is not None]
        assert meshes, 'no Mesh3d update in the animation frame'
        rgb = _parse_rgb_strings(meshes[0].vertexcolor) / 255.0
        assert np.abs(rgb[:, 0] - rgb[:, 1]).max() > 0.2, (
            'animated plotly hue surface rendered (near-)achromatic: '
            f'max |r-g| = {np.abs(rgb[:, 0] - rgb[:, 1]).max():.3f}')


class TestSurfaceValueValidation:
    """F07-006: surface dict VALUES are validated eagerly (before the
    analyze/reduce pipeline runs), with errors naming the key, the
    constraint, and the received value -- mirroring density='s validation."""

    @pytest.mark.parametrize('alpha', [1.5, -0.2, 0, 'red', True])
    def test_bad_alpha_raises_eagerly(self, alpha):
        with pytest.raises(ValueError, match=r"surface\['alpha'\]"):
            hyp.plot(_blob_3d(), '.', surface={'alpha': alpha}, show=False)

    @pytest.mark.parametrize('pre_inflate', [0, -1, 'big', np.inf, np.nan])
    def test_bad_pre_inflate_raises_eagerly(self, pre_inflate):
        with pytest.raises(ValueError, match=r"surface\['pre_inflate'\]"):
            hyp.plot(_blob_3d(), '.', surface={'pre_inflate': pre_inflate},
                     show=False)

    @pytest.mark.parametrize('smoothing', [-1, 7, '3', 2.5, True])
    def test_bad_smoothing_raises_eagerly(self, smoothing):
        with pytest.raises(ValueError, match=r"surface\['smoothing'\]"):
            hyp.plot(_blob_3d(), '.', surface={'smoothing': smoothing},
                     show=False)

    def test_bad_keep_points_raises_eagerly(self):
        with pytest.raises(ValueError, match=r"surface\['keep_points'\]"):
            hyp.plot(_blob_3d(), '.', surface={'keep_points': 'yes'},
                     show=False)

    def test_list_entries_are_value_validated_too(self):
        with pytest.raises(ValueError, match=r"surface\['alpha'\]"):
            hyp.plot(_blob_3d(), '.', surface=[{'alpha': 2.0}], show=False)

    @pytest.mark.parametrize('spec', [
        {'alpha': 1.0}, {'alpha': np.float64(0.5)}, {'smoothing': 0},
        {'smoothing': np.int64(2)}, {'pre_inflate': 1.3},
        {'keep_points': np.True_},
    ])
    def test_valid_values_still_accepted(self, spec):
        fig = hyp.plot(_blob_3d(n=60, seed=5), '.', surface=spec, show=False)
        assert fig is not None

    def test_smooth_hull_3d_never_leaks_qhullerror(self):
        """Direct meshutil callers with a degenerate (collapsed) hull get
        the documented ValueError, not a raw scipy QhullError dump."""
        pts = _blob_3d(n=50, seed=7)
        with pytest.raises(ValueError):
            smooth_hull_3d(pts, pre_inflate=0.0)


class TestNumpyBoolTopLevel:
    """F07-008: numpy bools behave like Python bools for the top-level
    surface=/density= flags (matching per_group's tolerance)."""

    def test_density_numpy_true_accepted(self):
        pts = _blob_3d(n=60, seed=3)
        fig = hyp.plot(pts, '.', density=np.True_, show=False)
        fig_off = hyp.plot(pts, '.', density=np.False_, show=False)
        assert fig is not None and fig_off is not None

    def test_surface_numpy_true_accepted(self):
        pts = _blob_3d(n=60, seed=3)
        fig = hyp.plot(pts, '.', surface=np.True_, show=False)
        hulls = [c for c in fig.axes[0].collections
                 if isinstance(c, Poly3DCollection)]
        assert hulls
        fig_off = hyp.plot(pts, '.', surface=np.False_, show=False)
        assert not [c for c in fig_off.axes[0].collections
                    if isinstance(c, Poly3DCollection)]

    def test_surface_list_of_numpy_bools_accepted(self):
        data = _two_blobs_3d(n=60)
        fig = hyp.plot(data, '.', surface=[np.True_, np.False_], show=False)
        hulls = [c for c in fig.axes[0].collections
                 if isinstance(c, Poly3DCollection)]
        assert len(hulls) == 1
