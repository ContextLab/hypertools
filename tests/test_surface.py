# -*- coding: utf-8 -*-
"""Tests for the `surface=` plot() kwarg (GH #109): smooth convex-hull
surfaces with lighting, for the matplotlib and plotly backends, static and
animated. No mocks -- real ConvexHull/mesh computation, real matplotlib
Poly3DCollection/plotly Mesh3d objects.
"""
import warnings

import matplotlib as mpl
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import hypertools as hyp
from hypertools.plot.meshutil import smooth_hull_3d

mpl.rcParams['figure.max_open_warning'] = 25


def _blob_3d(n=80, seed=0, center=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * 1.0 + np.asarray(center)


def _blob_2d(n=80, seed=1, center=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2)) * 1.0 + np.asarray(center)


def _two_datasets_3d():
    return [_blob_3d(seed=0, center=(-2, 0, 0)), _blob_3d(seed=1, center=(2, 0, 0))]


def _two_datasets_2d():
    return [_blob_2d(seed=0, center=(-2, 0)), _blob_2d(seed=1, center=(2, 0))]


class TestStaticMatplotlib3D:
    def test_collection_exists_with_faces(self):
        fig = hyp.plot(_two_datasets_3d(), '.', surface=True, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 2
        for c in colls:
            assert len(c.get_facecolor()) > 0

    def test_per_face_colors_vary_shading_active(self):
        fig = hyp.plot(_two_datasets_3d(), '.', surface=True, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        for c in colls:
            fc = np.asarray(c.get_facecolor())
            # real shading -> per-face RGB should NOT all be identical
            assert fc[:, :3].std(axis=0).sum() > 1e-6

    def test_verts_within_data_bbox_times_1_3(self):
        data = _two_datasets_3d()
        fig = hyp.plot(data, '.', surface=True, show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        stacked = np.vstack(data)
        center = stacked.mean(axis=0)
        half_range = np.ptp(stacked, axis=0).max() / 2.0 * 1.3
        for c in colls:
            # Poly3DCollection stores its (homogeneous) vertex coordinates
            # in the private `_vec` array (shape (4, n) -- x/y/z/w columns)
            # across the matplotlib versions hypertools supports.
            verts = np.asarray(c._vec[:3]).T
            assert np.all(np.abs(verts - center) <= half_range + 1e-6)

    def test_per_dataset_dict_list_honored_different_alphas(self):
        fig = hyp.plot(_two_datasets_3d(), '.',
                       surface=[{'alpha': 0.25}, {'alpha': 0.85}], show=False)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 2
        alphas = sorted(float(c.get_facecolor()[0][3]) for c in colls)
        assert alphas[0] == pytest.approx(0.25, abs=1e-6)
        assert alphas[1] == pytest.approx(0.85, abs=1e-6)

    def test_keep_points_false_hides_line(self):
        fig = hyp.plot(_two_datasets_3d(), '.',
                       surface={'keep_points': False}, show=False)
        ax = fig.axes[0]
        assert all(not l.get_visible() for l in ax.lines)


class TestStaticMatplotlib2D:
    def test_fill_patch_present_and_closed(self):
        fig = hyp.plot(_two_datasets_2d(), '.', surface=True, show=False)
        ax = fig.axes[0]
        # ax.fill() adds matplotlib.patches.Polygon patches
        polys = [p for p in ax.patches if hasattr(p, 'get_path')
                and p.get_label() == '_nolegend_']
        assert len(polys) == 2
        for p in polys:
            path = p.get_path()
            verts = path.vertices
            # closed: first == last vertex (matplotlib closes fill polygons)
            assert np.allclose(verts[0], verts[-1], atol=1e-6)


class TestValidation:
    def test_invalid_key_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), surface={'bogus': 1}, show=False)

    def test_invalid_lighting_key_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(),
                    surface={'lighting': {'bogus': 1}}, show=False)

    def test_1d_data_raises_valueerror(self):
        data_1d = [np.random.default_rng(0).normal(size=(20, 1))]
        with pytest.raises(ValueError):
            hyp.plot(data_1d, surface=True, show=False)

    def test_mismatched_list_length_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), surface=[{'alpha': 0.5}], show=False)

    def test_non_dict_list_item_raises_valueerror(self):
        with pytest.raises(ValueError):
            hyp.plot(_two_datasets_3d(), surface=['not-a-dict-or-bool'],
                     show=False)


class TestDegenerateInputs:
    def test_too_few_points_3d_warns_and_skips(self):
        data = [np.random.default_rng(0).normal(size=(3, 3))]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = hyp.plot(data, '.', surface=True, show=False)
        assert any('surface' in str(warning.message) for warning in w)
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 0

    def test_collinear_points_2d_warns_and_skips(self):
        pts = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = hyp.plot([pts], '.', surface=True, show=False)
        assert any('surface' in str(warning.message) for warning in w)


class TestStaticPlotly3D:
    def test_mesh3d_fields_and_lighting_roundtrip(self):
        """GH #109 round 3: plotly's OWN lighting engine is always the
        identity (see test_default_lighting_present) -- shading is instead
        precomputed per-vertex (matching the matplotlib renderer's
        Blinn-Phong model) and handed to Mesh3d as `vertexcolor`. A
        `lighting` override (e.g. `ambient`) therefore roundtrips into the
        computed `vertexcolor` values, not into `mesh.lighting`."""
        default_fig = hyp.plot(_two_datasets_3d(), '.', surface=True,
                               backend='plotly', show=False)
        bright_fig = hyp.plot(_two_datasets_3d(), '.',
                              surface={'lighting': {'ambient': 0.9}},
                              backend='plotly', show=False)
        default_meshes = [t for t in default_fig.data if t.type == 'mesh3d']
        bright_meshes = [t for t in bright_fig.data if t.type == 'mesh3d']
        assert len(default_meshes) == len(bright_meshes) == 2
        for m in default_meshes + bright_meshes:
            assert len(m.i) == len(m.j) == len(m.k) > 0
            assert len(m.x) == len(m.y) == len(m.z)
            assert m.lighting.ambient == pytest.approx(1.0)

        def _mean_channel_sum(mesh):
            rgb = np.array([[int(v) for v in c[4:-1].split(',')]
                            for c in mesh.vertexcolor])
            return rgb.mean()

        # a higher ambient term (0.9 vs the 0.45 default) should brighten
        # the precomputed vertex colors on average
        assert _mean_channel_sum(bright_meshes[0]) > _mean_channel_sum(
            default_meshes[0])

    def test_default_lighting_present(self):
        """GH #109 round 3: plotly's Mesh3d lighting engine is always set
        to the identity (ambient=1, diffuse=specular=fresnel=0) so it
        reproduces the precomputed `vertexcolor` values verbatim instead of
        re-shading them (which is what produced round 2's dark jagged
        patches on the doubled-winding faces)."""
        fig = hyp.plot(_two_datasets_3d(), '.', surface=True,
                       backend='plotly', show=False)
        meshes = [t for t in fig.data if t.type == 'mesh3d']
        assert meshes[0].lighting.ambient == pytest.approx(1.0)
        assert meshes[0].lighting.diffuse == pytest.approx(0.0)
        assert meshes[0].lighting.specular == pytest.approx(0.0)
        assert meshes[0].lighting.fresnel == pytest.approx(0.0)
        assert meshes[0].flatshading is False
        assert len(meshes[0].vertexcolor) == len(meshes[0].x)

    def test_mesh3d_gets_full_unculled_face_set(self):
        """Regression (GH #109 rendering-fix): a lone (non-overlapping)
        dataset's Mesh3d trace must carry the FULL face set that
        smooth_hull_3d produces -- not some backface-culled subset (the
        matplotlib renderer's own culled face list would punch exactly the
        kind of hole through the plotly mesh that a fresh visual review
        found).

        GH #109 round 2: plotly's Mesh3d ALSO back-face-culls individual
        triangles against the camera on its own (verified via a live
        Chromium render of a single, non-overlapping mesh with no other
        trace in the scene -- some of smooth_hull_3d's genuinely-concave
        dimple faces, an expected side effect of Taubin-smoothing an
        irregular hull, were culled from certain angles, punching a
        visible hole through to the background). The fix emits every face
        TWICE, once per winding order, so at least one copy is always
        front-facing -- the trace's face count is now exactly DOUBLE
        smooth_hull_3d's."""
        pts = _blob_3d(seed=0, center=(0.0, 0.0, 0.0))
        fig = hyp.plot([pts], '.', surface=True, backend='plotly', show=False)
        mesh = [t for t in fig.data if t.type == 'mesh3d'][0]
        _, expected_faces = smooth_hull_3d(pts, rounds=3, pre_inflate=1.15)
        assert len(mesh.i) == 2 * len(expected_faces)
        assert len(mesh.i) == len(mesh.j) == len(mesh.k)

    def test_mesh3d_has_both_windings_of_every_face_and_no_others(self):
        """Regression (GH #109 round 2): the Mesh3d trace's face set is
        EXACTLY two copies of each smooth_hull_3d face (one per winding
        order, for double-sided rendering -- see
        test_mesh3d_gets_full_unculled_face_set) -- not some other
        accidental duplication or a missing/extra face."""
        pts = _blob_3d(seed=0, center=(0.0, 0.0, 0.0))
        fig = hyp.plot([pts], '.', surface=True, backend='plotly', show=False)
        mesh = [t for t in fig.data if t.type == 'mesh3d'][0]
        faces = np.column_stack([mesh.i, mesh.j, mesh.k])
        sorted_faces = np.sort(faces, axis=1)
        uniq, counts = np.unique(sorted_faces, axis=0, return_counts=True)
        _, expected_faces = smooth_hull_3d(pts, rounds=3, pre_inflate=1.15)
        assert len(uniq) == len(expected_faces)
        assert np.all(counts == 2)

    def test_opaque_alpha_hides_no_mesh_faces_from_overlap_trim(self):
        """Regression: two overlapping datasets' Mesh3d traces must still
        each carry a non-empty, valid triangle set once faces enclosed by
        the OTHER dataset's surface are trimmed away (GH #109's fix for
        the plotly-only "hole" when two surfaces geometrically
        intersect)."""
        fig = hyp.plot(_two_datasets_3d(), '.', surface={'alpha': 1.0},
                       backend='plotly', show=False)
        meshes = [t for t in fig.data if t.type == 'mesh3d']
        assert len(meshes) == 2
        for m in meshes:
            assert len(m.i) > 0
            faces = np.column_stack([m.i, m.j, m.k])
            assert np.all(faces >= 0) and np.all(faces < len(m.x))

    def test_enclosed_marker_points_are_hidden_not_removed(self):
        """Regression: points a dataset's own surface encloses are hidden
        (NaN'd) from its marker trace -- plotly cannot reliably depth-
        composite Scatter3d points enclosed by an opaque Mesh3d surface --
        while the trace itself keeps its full original point count (NaN
        entries, not a shorter array)."""
        data = _two_datasets_3d()
        fig = hyp.plot(data, '.', surface={'alpha': 1.0}, backend='plotly',
                       show=False)
        markers = [t for t in fig.data
                  if t.type == 'scatter3d' and t.mode == 'markers']
        assert len(markers) == 2
        for arr, trace in zip(data, markers):
            assert len(trace.x) == len(arr)
            assert np.isnan(np.asarray(trace.x, dtype=float)).any()


class TestStaticPlotly2D:
    def test_toself_trace_present(self):
        fig = hyp.plot(_two_datasets_2d(), '.', surface=True,
                       backend='plotly', show=False)
        fills = [t for t in fig.data if getattr(t, 'fill', None) == 'toself']
        assert len(fills) == 2
        for t in fills:
            assert len(t.x) > 3
            # explicitly closed
            assert t.x[0] == t.x[-1] and t.y[0] == t.y[-1]


class TestAnimatedMatplotlib:
    @pytest.mark.parametrize('style', [True, 'spin', 'serial'])
    def test_five_frames_swap_collection(self, style):
        fig, ani = hyp.plot(_two_datasets_3d(), '.', surface=True,
                            animate=style, frame_rate=5, duration=1,
                            show=False)
        ax = fig.axes[0]
        face_counts = []
        for k in range(5):
            ani._func(k, *ani._args)
            colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
            # collections must be REPLACED (not accumulated) each frame
            assert len(colls) <= 2
            face_counts.append(sum(len(c.get_facecolor()) for c in colls))
        # vertex/face counts CAN differ across frames (growing window) --
        # just assert we got real data at some point
        assert max(face_counts) > 0


class TestAnimatedPlotly:
    @pytest.mark.parametrize('style', [True, 'spin', 'serial'])
    def test_frames_carry_mesh3d(self, style):
        fig = hyp.plot(_two_datasets_3d(), '.', surface=True,
                       backend='plotly', animate=style, frame_rate=5,
                       duration=1, show=False)
        assert len(fig.frames) == 5
        for frame in fig.frames:
            mesh_updates = [t for t in frame.data if t.type == 'mesh3d']
            assert len(mesh_updates) == 2
            for m in mesh_updates:
                assert len(m.i) == len(m.j) == len(m.k)


def _poly3d_verts(coll):
    """Extract a Poly3DCollection's raw (pre-projection) vertex
    coordinates as an (n, 3) array (GH #109 round 2 containment tests).
    ``_vec`` is matplotlib's internal homogeneous-coordinate cache
    populated whenever the collection's 3-D verts are set; used here
    (rather than the newly differently-signatured, deprecated
    ``get_vector()``) since it is stable across the matplotlib versions
    this project supports."""
    return np.asarray(coll._vec[:3]).T


class TestMeshContainmentInAxesCube:
    """GH #109 round 2: surface meshes (pre_inflate + smoothing overshoot,
    further grown for small point clouds by _rescale_for_containment) must
    never spill outside the drawn axes cube/cube trace -- both backends
    now size the cube to the ACTUAL built mesh extent (see
    `hypertools.plot.surface.surface_cube_scale`) instead of assuming the
    standard [-1, 1] data cube."""

    def test_static_mpl_mesh_within_axes_cube(self):
        fig = hyp.plot(_two_datasets_3d(), '.', surface=True, show=False)
        ax = fig.axes[0]
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(colls) == 2
        for c in colls:
            verts = _poly3d_verts(c)
            assert verts[:, 0].min() >= xlim[0] - 1e-9
            assert verts[:, 0].max() <= xlim[1] + 1e-9
            assert verts[:, 1].min() >= ylim[0] - 1e-9
            assert verts[:, 1].max() <= ylim[1] + 1e-9
            assert verts[:, 2].min() >= zlim[0] - 1e-9
            assert verts[:, 2].max() <= zlim[1] + 1e-9

    def test_static_mpl_cube_grows_past_default_for_this_data(self):
        """Positive control: the two-blob fixture's smoothed hulls DO bulge
        past the standard [-1, 1] cube, so `ax.get_xlim3d()` must have
        actually grown -- otherwise the containment test above would pass
        vacuously (cube big enough by luck, not by the fix)."""
        fig = hyp.plot(_two_datasets_3d(), '.', surface=True, show=False)
        ax = fig.axes[0]
        assert ax.get_xlim3d()[1] > 1.0

    @pytest.mark.parametrize('style', [True, 'spin', 'serial'])
    def test_animated_mpl_mesh_within_axes_cube_3_frames(self, style):
        fig, ani = hyp.plot(_two_datasets_3d(), '.', surface=True,
                            animate=style, frame_rate=5, duration=1,
                            show=False)
        ax = fig.axes[0]
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        assert xlim[1] > 1.0  # positive control: cube actually grew
        for k in [0, 2, 4]:
            ani._func(k, *ani._args)
            colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
            for c in colls:
                verts = _poly3d_verts(c)
                if len(verts) == 0:
                    continue  # too-small window this frame -- no mesh built
                assert verts[:, 0].min() >= xlim[0] - 1e-6
                assert verts[:, 0].max() <= xlim[1] + 1e-6
                assert verts[:, 1].min() >= ylim[0] - 1e-6
                assert verts[:, 1].max() <= ylim[1] + 1e-6
                assert verts[:, 2].min() >= zlim[0] - 1e-6
                assert verts[:, 2].max() <= zlim[1] + 1e-6

    def test_static_plotly_mesh_within_cube_trace_bounds(self):
        fig = hyp.plot(_two_datasets_3d(), '.', surface={'alpha': 1.0},
                       backend='plotly', show=False)
        meshes = [t for t in fig.data if t.type == 'mesh3d']
        assert len(meshes) == 2
        xr = fig.layout.scene.xaxis.range
        yr = fig.layout.scene.yaxis.range
        zr = fig.layout.scene.zaxis.range
        assert xr[1] > 1.0  # positive control: cube actually grew
        for m in meshes:
            x, y, z = np.asarray(m.x), np.asarray(m.y), np.asarray(m.z)
            assert x.min() >= xr[0] - 1e-9 and x.max() <= xr[1] + 1e-9
            assert y.min() >= yr[0] - 1e-9 and y.max() <= yr[1] + 1e-9
            assert z.min() >= zr[0] - 1e-9 and z.max() <= zr[1] + 1e-9
        # the drawn wireframe cube trace itself must match the SAME bound
        cube = [t for t in fig.data
               if t.type == 'scatter3d' and t.mode == 'lines'][0]
        cube_x = [v for v in cube.x if v is not None]
        assert max(abs(v) for v in cube_x) == pytest.approx(xr[1])


class TestPlotlyMeshDoubleSidedAndTrimPriority:
    """GH #109 round 2: re-verification of round 1's plotly seam/hole
    findings turned up two distinct, real defects (confirmed via a live
    Chromium render, not just kaleido export) -- a single mesh's own
    back-face-culled dimples (fixed by emitting both winding orders per
    face) and genuine WebGL depth-fighting between two overlapping OPAQUE
    Mesh3d volumes with no trim at all (so trimming must stay). Switching
    the trim from mutual (every dataset cut against every other) to
    priority (only cut against LOWER-indexed datasets) guarantees the
    first dataset in any overlapping cluster is always a complete, closed
    surface for later datasets' cut edges to sit against."""

    def test_priority_trim_keeps_lowest_index_dataset_whole(self):
        from hypertools.plot.plotly_backend import _trim_faces_inside_other_meshes

        pts0 = _blob_3d(seed=0, center=(-1.2, 0, 0))
        pts1 = _blob_3d(seed=1, center=(1.2, 0, 0))
        mesh0 = smooth_hull_3d(pts0, rounds=2, pre_inflate=1.15)
        mesh1 = smooth_hull_3d(pts1, rounds=2, pre_inflate=1.15)
        meshes = {0: mesh0, 1: mesh1}

        keep0 = _trim_faces_inside_other_meshes(0, meshes)
        keep1 = _trim_faces_inside_other_meshes(1, meshes)

        # dataset 0 (lowest index) is NEVER trimmed against anything
        assert keep0.all()
        # dataset 1 (overlaps dataset 0) loses at least some faces --
        # otherwise this fixture isn't actually exercising an overlap
        assert not keep1.all()
        assert keep1.sum() > 0

    def test_two_overlapping_datasets_no_white_gap_hole(self):
        """Regression: with the double-sided fix + priority trim, neither
        overlapping dataset's Mesh3d trace should be reduced to nothing,
        and the untrimmed (priority) dataset keeps its full, doubled face
        set."""
        data = _two_datasets_3d()
        fig = hyp.plot(data, '.', surface={'alpha': 1.0}, backend='plotly',
                       show=False)
        meshes = [t for t in fig.data if t.type == 'mesh3d']
        assert len(meshes) == 2
        for m in meshes:
            assert len(m.i) > 0
        pts0 = np.atleast_2d(data[0])[:, :3]
        # dataset 0's OWN un-trimmed mesh (built independently here on the
        # same, already-scaled points) has a known face count -- the
        # dataset-0 trace (lowest index, never trimmed) must carry EXACTLY
        # double that (both windings), not some cut-down subset.
        _, expected_faces0 = smooth_hull_3d(pts0, rounds=3, pre_inflate=1.15)
        face_counts = sorted(len(m.i) for m in meshes)
        assert face_counts[-1] == 2 * len(expected_faces0)
