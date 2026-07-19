# -*- coding: utf-8 -*-
"""Tests for the `density=` plot() kwarg (GH #108, #191): subtle KDE
density shading, off by default, for the matplotlib and plotly backends,
static and animated. No mocks -- real `scipy.stats.gaussian_kde`, real
matplotlib image/Poly3DCollection/scatter artists, real plotly
Contour/Volume traces. The scikit-image-absent fallback path is exercised
via a REAL import-system meta-path blocker in a subprocess (not a mock of
hypertools itself), leaving this process's already-imported skimage alone.
"""
import subprocess
import sys
import textwrap
import warnings

import matplotlib as mpl
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import hypertools as hyp
from hypertools.plot.density import (
    DENSITY_BOOST_MAX,
    DENSITY_DEFAULTS,
    HAS_SKIMAGE,
    MAX_VOLUME_OPACITY,
    bbox_extent,
    density_alpha_boost,
    resolve_plotly_volume_params,
)

mpl.rcParams['figure.max_open_warning'] = 25


def _blob_3d(n=150, seed=0, center=(0.0, 0.0, 0.0), scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * scale + np.asarray(center)


def _blob_2d(n=150, seed=1, center=(0.0, 0.0), scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2)) * scale + np.asarray(center)


def _two_datasets_3d():
    return [_blob_3d(seed=0, center=(-3, 0, 0)), _blob_3d(seed=1, center=(3, 0, 0))]


def _two_datasets_2d():
    return [_blob_2d(seed=0, center=(-3, 0)), _blob_2d(seed=1, center=(3, 0))]


def _two_identical_datasets_3d():
    """Two datasets built from the exact SAME underlying points (just
    duplicated) -- guarantees each dataset's own bounding box exactly
    equals the whole scene's bounding box, so the GH #108 round-2 density
    auto-boost (see `density_alpha_boost`) is EXACTLY `1.0` (a no-op) for
    both. Used by tests that pin the BASE, pre-boost `density=` formula
    constants; `_two_datasets_3d` (genuinely separated) is used instead
    wherever a test wants the boost engaged."""
    pts = _blob_3d(n=200, seed=0, center=(0, 0, 0))
    return [pts, pts.copy()]


def _two_blobs_3d_sep(sep, n=200):
    """Two well-separated 3-D blobs, `sep` units apart (center-to-center,
    in RAW pre-transform units) along x -- used to exercise the density
    auto-boost at a range of separations (GH #108 round 2)."""
    return [_blob_3d(n=n, seed=0, center=(-sep / 2, 0, 0)),
            _blob_3d(n=n, seed=1, center=(sep / 2, 0, 0))]


class TestStaticMatplotlib2D:
    def test_image_artist_present_one_per_dataset(self):
        fig = hyp.plot(_two_datasets_2d(), '.', density=True, show=False)
        ax = fig.axes[0]
        assert len(ax.get_images()) == 2

    def test_peak_alpha_matches_requested(self):
        fig = hyp.plot(_two_datasets_2d(), '.', density={'alpha': 0.35},
                       show=False)
        ax = fig.axes[0]
        for im in ax.get_images():
            cmap = im.get_cmap()
            assert cmap(1.0)[3] == pytest.approx(0.35, abs=1e-9)
            assert cmap(0.0)[3] == pytest.approx(0.0, abs=1e-9)

    def test_default_alpha(self):
        fig = hyp.plot(_two_datasets_2d(), '.', density=True, show=False)
        ax = fig.axes[0]
        for im in ax.get_images():
            assert im.get_cmap()(1.0)[3] == pytest.approx(
                DENSITY_DEFAULTS['alpha'], abs=1e-9)

    def test_kde_peak_location_matches_blob_mean(self):
        data = [_blob_2d(n=300, seed=7, center=(2.5, -1.5))]
        fig = hyp.plot(data, '.', density=True, show=False)
        ax = fig.axes[0]
        im = ax.get_images()[0]
        Z = np.asarray(im.get_array())
        xmin, xmax, ymin, ymax = im.get_extent()
        xs = np.linspace(xmin, xmax, Z.shape[1])
        ys = np.linspace(ymin, ymax, Z.shape[0])
        iy, ix = np.unravel_index(np.argmax(Z), Z.shape)
        peak = np.array([xs[ix], ys[iy]])
        # the plotted data is centered/scaled into [-1, 1] by plot(), so
        # compare the peak against the (post-transform) data's own mean
        # rather than the pre-transform blob center.
        drawn_pts = np.column_stack(ax.lines[0].get_data())
        assert np.allclose(peak, drawn_pts.mean(axis=0), atol=0.4)

    def test_per_group_colors_distinct(self):
        fig = hyp.plot(_two_datasets_2d(), '.', density=True, show=False)
        ax = fig.axes[0]
        cmaps = [im.get_cmap() for im in ax.get_images()]
        colors = [cmap(1.0)[:3] for cmap in cmaps]
        assert not np.allclose(colors[0], colors[1])

    def test_per_group_false_pools_into_one_layer(self):
        fig = hyp.plot(_two_datasets_2d(), '.',
                       density={'per_group': False}, show=False)
        ax = fig.axes[0]
        assert len(ax.get_images()) == 1

    def test_default_off_no_density_artists(self):
        fig = hyp.plot(_two_datasets_2d(), '.', show=False)
        ax = fig.axes[0]
        assert len(ax.get_images()) == 0


class TestStaticMatplotlib3D:
    @pytest.mark.skipif(not HAS_SKIMAGE, reason="requires scikit-image")
    def test_iso_surfaces_three_collections_expected_alphas(self):
        fig = hyp.plot([_blob_3d(n=200, seed=0, center=(0, 0, 0))], '.',
                       density=True, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 3
        alphas = sorted(float(c.get_facecolor()[0][3]) for c in colls)
        assert alphas == pytest.approx([0.03, 0.05, 0.07], abs=1e-9)

    @pytest.mark.skipif(not HAS_SKIMAGE, reason="requires scikit-image")
    def test_alpha_scales_iso_surface_alphas(self):
        fig = hyp.plot([_blob_3d(n=200, seed=0, center=(0, 0, 0))], '.',
                       density={'alpha': 0.4}, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 3
        alphas = sorted(float(c.get_facecolor()[0][3]) for c in colls)
        # alpha=0.4 is 2x the 0.2 default -> alphas should double too
        assert alphas == pytest.approx([0.06, 0.10, 0.14], abs=1e-9)

    @pytest.mark.skipif(not HAS_SKIMAGE, reason="requires scikit-image")
    def test_per_dataset_two_datasets_six_collections(self):
        fig = hyp.plot(_two_datasets_3d(), '.', density=True, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 6

    @pytest.mark.skipif(not HAS_SKIMAGE, reason="requires scikit-image")
    def test_levels_one_gives_one_collection(self):
        fig = hyp.plot([_blob_3d(n=200, seed=0, center=(0, 0, 0))], '.',
                       density={'levels': 1}, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 1

    @pytest.mark.skipif(not HAS_SKIMAGE, reason="requires scikit-image")
    def test_levels_five_gives_five_monotonic_alphas(self):
        fig = hyp.plot([_blob_3d(n=200, seed=0, center=(0, 0, 0))], '.',
                       density={'levels': 5}, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 5
        alphas = sorted(float(c.get_facecolor()[0][3]) for c in colls)
        assert alphas == pytest.approx(list(np.linspace(0.03, 0.07, 5)),
                                       abs=1e-9)
        assert all(a2 > a1 for a1, a2 in zip(alphas, alphas[1:]))

    @pytest.mark.skipif(not HAS_SKIMAGE, reason="requires scikit-image")
    def test_levels_three_explicit_matches_legacy_exactly(self):
        fig = hyp.plot([_blob_3d(n=200, seed=0, center=(0, 0, 0))], '.',
                       density={'levels': 3}, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 3
        alphas = sorted(float(c.get_facecolor()[0][3]) for c in colls)
        assert alphas == pytest.approx([0.03, 0.05, 0.07], abs=1e-9)

    def test_default_off_no_density_collections(self):
        fig = hyp.plot(_two_datasets_3d(), '.', show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 0


class TestFogFallbackSubprocess:
    """Exercises the scikit-image-absent 3-D fallback via a REAL meta-path
    finder that blocks `import skimage` -- a genuine import-system behavior,
    not a mock of hypertools -- run in a subprocess so this test process's
    (already-imported, really-installed) scikit-image is untouched."""

    SCRIPT = textwrap.dedent("""
        import sys, warnings
        import importlib.abc, importlib.machinery

        class BlockLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise ImportError("skimage blocked for test")

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == 'skimage' or name.startswith('skimage.'):
                    return importlib.machinery.ModuleSpec(name, BlockLoader())
                return None

        sys.meta_path.insert(0, Blocker())

        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import hypertools as hyp
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from hypertools.plot import density as density_mod

        assert density_mod.HAS_SKIMAGE is False, "blocker did not take effect"

        rng = np.random.default_rng(0)
        data = [rng.normal(size=(150, 3))]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = hyp.plot(data, '.', density=True, show=False)
            msgs = [str(x.message) for x in w
                    if issubclass(x.category, UserWarning)]

        density_warnings = [m for m in msgs if 'density' in m]
        assert len(density_warnings) == 1, density_warnings
        msg = density_warnings[0]
        assert 'scikit-image' in msg
        assert 'fog' in msg
        assert 'hypertools[density3d]' in msg
        assert "backend='plotly'" in msg

        ax = fig.axes[0]
        polys = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(polys) == 0, "no iso-surfaces should be drawn"

        # the fog fallback is a 3-D scatter (ax.scatter -> Path3DCollection)
        fog = [c for c in ax.collections
               if not isinstance(c, Poly3DCollection)
               and hasattr(c, '_offsets3d')]
        assert len(fog) == 1, "expected exactly one fog scatter collection"

        print("SUBPROCESS_OK")
    """)

    def test_fog_fallback_warns_and_draws_scatter(self):
        result = subprocess.run(
            [sys.executable, '-c', self.SCRIPT],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert 'SUBPROCESS_OK' in result.stdout


class TestStaticPlotly3D:
    def test_volume_trace_params_exact(self):
        # identical-points fixture -> boost == 1.0 (no-op): this pins the
        # BASE, pre-boost formula constants (GH #108 round 2 dedicated
        # tests below cover the boosted, genuinely-separated case).
        fig = hyp.plot(_two_identical_datasets_3d(), '.', density=True,
                       backend='plotly', show=False)
        volumes = [t for t in fig.data if t.type == 'volume']
        assert len(volumes) == 2
        for v in volumes:
            assert v.isomin == pytest.approx(0.05)
            assert v.isomax == pytest.approx(1.0)
            assert v.surface.count == 15
            assert v.opacity == pytest.approx(0.4, abs=1e-9)  # 2.0 * 0.2
            assert tuple(tuple(p) for p in v.opacityscale) == (
                (0, 0), (0.3, 0.27), (1, 0.53))
            assert v.showscale is False
            assert v.hoverinfo == 'skip'
            # solid (2-stop, same color) colorscale
            assert len(v.colorscale) == 2
            assert v.colorscale[0][1] == v.colorscale[1][1]

    def test_opacity_scales_with_alpha_capped(self):
        # identical-points fixture -> boost == 1.0 (no-op): pins the BASE
        # opacity-capping formula.
        fig = hyp.plot(_two_identical_datasets_3d(), '.',
                       density={'alpha': 0.3}, backend='plotly', show=False)
        volumes = [t for t in fig.data if t.type == 'volume']
        for v in volumes:
            # min(2.0 * 0.3, 0.4) == 0.4 (capped)
            assert v.opacity == pytest.approx(0.4, abs=1e-9)

    def test_per_group_colors_distinct(self):
        fig = hyp.plot(_two_datasets_3d(), '.', density=True,
                       backend='plotly', show=False)
        volumes = [t for t in fig.data if t.type == 'volume']
        colors = [v.colorscale[0][1] for v in volumes]
        assert colors[0] != colors[1]

    def test_surface_count_scales_with_levels(self):
        # identical-points fixture -> boost == 1.0 (no-op): pins the BASE
        # surface_count formula.
        fig = hyp.plot(_two_identical_datasets_3d(), '.',
                       density={'levels': 5}, backend='plotly', show=False)
        volumes = [t for t in fig.data if t.type == 'volume']
        assert len(volumes) == 2
        for v in volumes:
            assert v.surface.count == 25  # 5 * levels

    def test_surface_count_default_levels_unchanged(self):
        fig = hyp.plot(_two_identical_datasets_3d(), '.', density=True,
                       backend='plotly', show=False)
        volumes = [t for t in fig.data if t.type == 'volume']
        for v in volumes:
            assert v.surface.count == 15  # 5 * default levels (3)

    def test_default_off_no_volume_traces(self):
        fig = hyp.plot(_two_datasets_3d(), '.', backend='plotly', show=False)
        assert not [t for t in fig.data if t.type == 'volume']


class TestStaticPlotly2D:
    def test_contour_trace_params_exact(self):
        fig = hyp.plot(_two_datasets_2d(), '.', density=True,
                       backend='plotly', show=False)
        contours = [t for t in fig.data if t.type == 'contour']
        assert len(contours) == 2
        for c in contours:
            assert c.contours.coloring == 'heatmap'
            assert c.contours.showlines is False
            assert c.line.width == 0
            assert c.showscale is False
            assert c.hoverinfo == 'skip'
            assert 'rgba(' in c.colorscale[0][1]
            assert c.colorscale[0][1].rstrip(')').endswith(',0')

    def test_alpha_ramp_scales_to_1_5x(self):
        fig = hyp.plot(_two_datasets_2d(), '.', density={'alpha': 0.2},
                       backend='plotly', show=False)
        contours = [t for t in fig.data if t.type == 'contour']
        top_alpha = float(contours[0].colorscale[1][1].rstrip(')').split(',')[-1])
        assert top_alpha == pytest.approx(0.3, abs=1e-9)

    def test_default_off_no_contour_traces(self):
        fig = hyp.plot(_two_datasets_2d(), '.', backend='plotly', show=False)
        assert not [t for t in fig.data if t.type == 'contour']

    def test_density_below_data_traces(self):
        fig = hyp.plot(_two_datasets_2d(), '.', density=True,
                       backend='plotly', show=False)
        types = [t.type for t in fig.data]
        first_scatter = types.index('scatter')
        first_contour = types.index('contour')
        assert first_contour < first_scatter


class TestAnimatedMatplotlib:
    def test_density_background_present_and_static_across_frames(self):
        fig, ani = hyp.plot(_two_datasets_3d(), '.', density=True,
                            animate=True, frame_rate=5, duration=1,
                            show=False)
        ax = fig.axes[0]
        colls_before = [c for c in ax.collections
                        if isinstance(c, Poly3DCollection)]
        assert len(colls_before) == 6
        ids_before = set(id(c) for c in colls_before)

        for k in range(5):
            ani._func(k, *ani._args)

        colls_after = [c for c in ax.collections
                       if isinstance(c, Poly3DCollection)]
        ids_after = set(id(c) for c in colls_after)
        # density artists are drawn ONCE and never touched by frame updates
        assert ids_after == ids_before


class TestAnimatedPlotly:
    @pytest.mark.parametrize('style', [True, 'spin', 'serial'])
    def test_volume_traces_present_and_excluded_from_frames(self, style):
        fig = hyp.plot(_two_datasets_3d(), '.', density=True,
                       backend='plotly', animate=style, frame_rate=5,
                       duration=1, show=False)
        volume_indices = [i for i, t in enumerate(fig.data)
                          if t.type == 'volume']
        assert len(volume_indices) == 2
        for frame in fig.frames:
            touched = set(frame.traces or ())
            assert touched.isdisjoint(volume_indices)

    def test_contour_traces_present_and_excluded_from_frames_2d(self):
        fig = hyp.plot(_two_datasets_2d(), '.', density=True,
                       backend='plotly', animate=True, frame_rate=5,
                       duration=1, show=False)
        contour_indices = [i for i, t in enumerate(fig.data)
                           if t.type == 'contour']
        assert len(contour_indices) == 2
        for frame in fig.frames:
            touched = set(frame.traces or ())
            assert touched.isdisjoint(contour_indices)
        # and the data traces (scatter) are STILL properly animated
        scatter_indices = [i for i, t in enumerate(fig.data)
                           if t.type == 'scatter']
        assert set(fig.frames[0].traces) == set(scatter_indices)


class TestValidation:
    def test_invalid_key_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), density={'bogus': 1}, show=False)

    def test_non_bool_non_dict_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), density='bogus', show=False)

    def test_1d_data_raises_valueerror(self):
        data_1d = [np.random.default_rng(0).normal(size=(20, 1))]
        with pytest.raises(ValueError):
            hyp.plot(data_1d, density=True, ndims=1, show=False)

    def test_alpha_zero_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), density={'alpha': 0}, show=False)

    def test_alpha_above_one_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), density={'alpha': 1.5}, show=False)

    def test_grid_too_small_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), density={'grid': 2}, show=False)

    def test_levels_zero_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), density={'levels': 0}, show=False)

    def test_levels_above_ten_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), density={'levels': 11}, show=False)

    def test_per_group_non_bool_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), density={'per_group': 'yes'},
                     show=False)


class TestDegenerateInputs:
    def test_too_few_points_warns_and_skips(self):
        data = [np.random.default_rng(0).normal(size=(2, 3))]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = hyp.plot(data, '.', density=True, show=False)
        assert any('density' in str(warning.message) for warning in w)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 0

    def test_collinear_points_2d_warns_and_skips(self):
        pts = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = hyp.plot([pts], '.', density=True, show=False)
        assert any('density' in str(warning.message) for warning in w)
        ax = fig.axes[0]
        assert len(ax.get_images()) == 0


class TestDensityAlphaBoostFormula:
    """Direct, numeric tests of the `density_alpha_boost`/`bbox_extent`
    formula (GH #108 round 2: 3-D density was effectively invisible once
    datasets were separated beyond ~4-5 std, at default alpha, for ALL
    `levels` values -- see the through-`hyp.plot` engagement tests below
    for the end-to-end regression)."""

    def test_scene_filling_dataset_boost_is_one(self):
        assert density_alpha_boost(2.0, 2.0) == pytest.approx(1.0)

    def test_small_dataset_boost_exceeds_one(self):
        assert density_alpha_boost(1.0, 3.0) > 1.0

    def test_boost_clamped_at_max(self):
        # scene_extent / dataset_extent = 100 -- far past max_boost
        boost = density_alpha_boost(0.01, 1.0, max_boost=6.0)
        assert boost == pytest.approx(6.0)

    def test_boost_never_below_one(self):
        # a dataset bigger than the whole scene shouldn't happen in
        # practice, but the formula must never DARKEN (return < 1)
        assert density_alpha_boost(5.0, 1.0) >= 1.0

    def test_degenerate_zero_extent_returns_one(self):
        assert density_alpha_boost(0.0, 2.0) == pytest.approx(1.0)
        assert density_alpha_boost(2.0, 0.0) == pytest.approx(1.0)

    def test_bbox_extent_matches_known_box(self):
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        # span (3, 4, 0) -> Euclidean norm 5
        assert bbox_extent(pts) == pytest.approx(5.0)

    def test_gamma_two_matches_quadratic_scaling(self):
        # explicit gamma overrides the module default
        boost = density_alpha_boost(1.0, 2.0, gamma=2.0, max_boost=100.0)
        assert boost == pytest.approx(4.0)


class TestDensityBoostEngagementMatplotlib:
    """Numeric regression: mpl iso-surface alphas are boosted for a
    well-separated (small-in-scene) two-blob 3-D scene, but NOT for a
    single, scene-filling dataset (GH #108 round 2)."""

    def test_separated_blobs_boost_effective_alphas(self):
        fig = hyp.plot(_two_blobs_3d_sep(10), '.', density=True, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 6  # 2 datasets * 3 levels
        alphas = [float(c.get_facecolor()[0][3]) for c in colls]
        base_alphas = (0.03, 0.05, 0.07)
        # every boosted alpha must be at least 3x its own base shell alpha
        assert min(alphas) >= 3 * min(base_alphas)

    def test_scene_filling_single_blob_keeps_base_alphas(self):
        fig = hyp.plot([_blob_3d(n=200, seed=0, center=(0, 0, 0))], '.',
                       density=True, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        alphas = sorted(float(c.get_facecolor()[0][3]) for c in colls)
        assert alphas == pytest.approx([0.03, 0.05, 0.07], abs=1e-9)

    def test_boost_grows_with_separation(self):
        peak_alpha_by_sep = {}
        for sep in (0, 5, 10):
            fig = hyp.plot(_two_blobs_3d_sep(sep), '.', density=True,
                           show=False)
            ax = fig.axes[0]
            colls = [c for c in ax.collections
                    if isinstance(c, Poly3DCollection)]
            peak_alpha_by_sep[sep] = max(
                float(c.get_facecolor()[0][3]) for c in colls)
        assert peak_alpha_by_sep[0] < peak_alpha_by_sep[5] < peak_alpha_by_sep[10]

    def test_levels_one_vs_six_distinguishable_when_separated(self):
        # GH #108 round 2: the review found levels=1 vs levels=6 looked
        # indistinguishable once the boost made both visible -- assert
        # their peak (innermost-shell) alpha genuinely differs.
        peak_alpha_by_levels = {}
        for levels in (1, 6):
            fig = hyp.plot(_two_blobs_3d_sep(10), '.',
                           density={'levels': levels}, show=False)
            ax = fig.axes[0]
            colls = [c for c in ax.collections
                    if isinstance(c, Poly3DCollection)]
            assert len(colls) == 2 * levels
            peak_alpha_by_levels[levels] = max(
                float(c.get_facecolor()[0][3]) for c in colls)
        assert peak_alpha_by_levels[6] > peak_alpha_by_levels[1]


class TestDensityBoostEngagementPlotly:
    """Numeric regression: plotly Volume opacity/surface_count are boosted
    for a well-separated two-blob 3-D scene, capped at
    `MAX_VOLUME_OPACITY`, but unchanged for a single, scene-filling
    dataset (GH #108 round 2)."""

    def test_separated_blobs_boost_opacity_above_base_cap(self):
        fig = hyp.plot(_two_blobs_3d_sep(10), '.', density=True,
                       backend='plotly', show=False)
        volumes = [t for t in fig.data if t.type == 'volume']
        assert len(volumes) == 2
        for v in volumes:
            # base (unboosted) cap is 0.4 -- boost must push past it
            assert v.opacity > 0.4
            assert v.opacity <= MAX_VOLUME_OPACITY + 1e-9

    def test_separated_blobs_boost_surface_count(self):
        fig = hyp.plot(_two_blobs_3d_sep(10), '.', density=True,
                       backend='plotly', show=False)
        volumes = [t for t in fig.data if t.type == 'volume']
        for v in volumes:
            assert v.surface.count > 15  # base is 5 * default levels (3)

    def test_scene_filling_single_blob_keeps_base_opacity(self):
        fig = hyp.plot([_blob_3d(n=200, seed=0, center=(0, 0, 0))], '.',
                       density=True, backend='plotly', show=False)
        volumes = [t for t in fig.data if t.type == 'volume']
        assert len(volumes) == 1
        assert volumes[0].opacity == pytest.approx(0.4, abs=1e-9)
        assert volumes[0].surface.count == 15

    def test_resolve_plotly_volume_params_noop_at_boost_one(self):
        pad, isomin, opacityscale, opacity, surface_count = (
            resolve_plotly_volume_params(0.2, 3, boost=1.0))
        assert pad == pytest.approx(0.15)
        assert isomin == pytest.approx(0.05)
        assert opacityscale == [[0, 0], [0.3, 0.27], [1, 0.53]]
        assert opacity == pytest.approx(0.4, abs=1e-9)
        assert surface_count == 15

    def test_resolve_plotly_volume_params_widens_at_max_boost(self):
        pad, isomin, opacityscale, opacity, surface_count = (
            resolve_plotly_volume_params(0.2, 3, boost=DENSITY_BOOST_MAX))
        assert pad > 0.15
        assert isomin < 0.05
        assert opacity == pytest.approx(MAX_VOLUME_OPACITY, abs=1e-9)
        assert surface_count == 5 * 3 * DENSITY_BOOST_MAX


class TestVolumeMoreTransparentThanR1:
    """Regression (R2 fix, maintainer request): plotly's `go.Volume`
    density shading must be strictly LESS opaque than the original R1
    tuning at every boost level -- both the absolute ceiling
    (`MAX_VOLUME_OPACITY`) and the base (`boost=1`, unboosted) opacity/
    opacityscale constants were lowered so the glow reads as subtle/airy
    like matplotlib's iso-surfaces instead of a heavy, dense blob (see
    `docs/images/v1.0-seven-features/density_3d_plotly.png`)."""

    R1_MAX_VOLUME_OPACITY = 0.95
    R1_BASE_OPACITY_CAP = 0.6
    R1_OPACITYSCALE_NOOP = ((0, 0), (0.3, 0.4), (1, 0.8))

    def test_max_volume_opacity_lowered(self):
        assert MAX_VOLUME_OPACITY < self.R1_MAX_VOLUME_OPACITY

    def test_base_opacity_lowered_at_every_alpha(self):
        for alpha in (0.05, 0.2, 0.5, 1.0):
            _, _, _, opacity, _ = resolve_plotly_volume_params(
                alpha, 3, boost=1.0)
            r1_opacity = min(3.0 * alpha, self.R1_BASE_OPACITY_CAP)
            assert opacity < r1_opacity

    def test_opacityscale_ramp_lowered_at_noop_boost(self):
        _, _, opacityscale, _, _ = resolve_plotly_volume_params(
            0.2, 3, boost=1.0)
        # skip the trivial (0, 0) origin breakpoint shared by both curves
        for (new_x, new_y), (r1_x, r1_y) in zip(
                opacityscale[1:], self.R1_OPACITYSCALE_NOOP[1:]):
            assert new_x == pytest.approx(r1_x)  # x breakpoints unchanged
            assert new_y < r1_y  # y (opacity at that breakpoint) lowered

    def test_boosted_opacity_still_lower_than_r1_ceiling(self):
        # even a maximally-boosted (small-in-scene) cluster must stay
        # below the OLD ceiling -- boosting must never claw back the
        # transparency win.
        _, _, _, opacity, _ = resolve_plotly_volume_params(
            0.2, 3, boost=DENSITY_BOOST_MAX)
        assert opacity < self.R1_MAX_VOLUME_OPACITY
