# -*- coding: utf-8 -*-
"""Regression tests for the 2026-07 release audit, batch B1 (static plot
fundamentals, input handling, and kwarg validation).

Covers CONFIRMED findings from audit units F01-plot-static-core,
F03-plot-pipeline-integration, F08-plot-inputs, F10-plot-kwargs-sweep and
D11-docstrings-plot. Real `hyp.plot()` calls only (no mocks): every
assertion inspects actual matplotlib artists, real exceptions, or the real
docstring.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.colors import mat2colors, continuous_colormap
from hypertools.plot.density import fit_kde
from hypertools.plot.plot import _expand_labels


def _line_vertices_3d(line):
    x, y, z = line.get_data_3d()
    return np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])


def _expected_display_transform(data):
    """The documented display transform: center on the stacked mean, then
    rescale to [-1, 1]."""
    c = data - data.mean(axis=0)
    m1 = c.min()
    m2 = (c - m1).max()
    return 2 * ((c - m1) / m2) - 1


# ---------------------------------------------------------------------------
# F01-001 / X3-performance-002: data-faithful static lines
# ---------------------------------------------------------------------------
class TestStaticLineDataFidelity:
    def test_2000_point_spike_and_endpoint_are_drawn(self):
        """A 2000-sample series with a 50-sigma spike: the spike and the
        exact final sample must appear among the drawn line vertices."""
        rng = np.random.default_rng(0)
        n = 2000
        data = rng.standard_normal((n, 3))
        data[777] = [50.0, 50.0, 50.0]  # 50-sigma spike
        fig = hyp.plot(data, "-", reduce=None, show=False)
        verts = _line_vertices_3d(fig.axes[0].lines[0])
        plt.close("all")

        expected = _expected_display_transform(data)
        assert verts.shape[0] >= n  # no decimation: every sample drawn
        # the spike is one of the drawn vertices
        d_spike = np.linalg.norm(verts - expected[777], axis=1).min()
        assert d_spike < 1e-8
        # the exact final sample is drawn, and it is the final vertex
        assert np.allclose(verts[-1], expected[-1], atol=1e-8)

    def test_901_sample_endpoint_not_dropped(self):
        """Historical bug: at n=901 the interpolation grid dropped the final
        sample entirely (and it fell outside the [-1, 1] scaling)."""
        n = 901
        tr = np.column_stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)])
        tr[-1] = [2.0, 1.0, 0.0]
        fig = hyp.plot(tr, "-", reduce=None, show=False)
        verts = _line_vertices_3d(fig.axes[0].lines[0])
        plt.close("all")

        expected = _expected_display_transform(tr)
        assert np.allclose(verts[-1], expected[-1], atol=1e-8)
        # every vertex stays inside the [-1, 1] cube
        assert np.all(verts >= -1 - 1e-9) and np.all(verts <= 1 + 1e-9)

    def test_every_input_vertex_on_interpolated_line(self):
        """For small n the line is still smoothed (adds points) but every
        input sample must remain among the drawn vertices. Verified in
        display space via 'o-' markers, which are drawn at the true sample
        points under the SAME display transform as the line."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((25, 3))
        fig = hyp.plot(data, "o-", reduce=None, show=False)
        lines = fig.axes[0].lines
        line_art = [l for l in lines if l.get_linestyle() == "-"]
        marker_art = [l for l in lines
                      if l.get_linestyle() in ("None", "none")]
        assert len(line_art) == 1 and len(marker_art) == 1
        lverts = _line_vertices_3d(line_art[0])
        mverts = _line_vertices_3d(marker_art[0])
        plt.close("all")

        assert mverts.shape[0] == 25
        assert lverts.shape[0] > 25  # interpolation still adds points
        for row in mverts:  # every true sample is a drawn line vertex
            assert np.linalg.norm(lverts - row, axis=1).min() < 1e-8

    def test_interp_static_line_keeps_knots_exactly(self):
        from hypertools.plot.plot import _interp_static_line
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((37, 3))
        out = _interp_static_line(arr)
        assert out.shape[0] >= 900
        # every original sample appears exactly among the output vertices
        for row in arr:
            assert np.linalg.norm(out - row, axis=1).min() == 0.0
        assert np.array_equal(out[-1], arr[-1])

    def test_combo_markers_inside_cube_and_on_line_scale(self):
        """'o-' markers must live in the SAME [-1, 1] display space as the
        line (historically the dropped endpoint made markers spill outside
        the cube)."""
        n = 901
        tr = np.column_stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)])
        tr[-1] = [2.0, 1.0, 0.0]
        fig = hyp.plot(tr, "o-", reduce=None, show=False)
        lines = fig.axes[0].lines
        marker_art = [l for l in lines if l.get_marker() not in ("None", None, "")]
        assert marker_art
        mverts = _line_vertices_3d(marker_art[0])
        plt.close("all")
        assert np.all(mverts >= -1 - 1e-9) and np.all(mverts <= 1 + 1e-9)

    def test_duration_and_frame_rate_do_not_affect_static_lines(self):
        """F01-007: duration=/frame_rate= are documented '(animation only)'
        and must not change a STATIC line's rendering."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((20, 3))
        counts = []
        for kw in ({}, {"duration": 1}, {"frame_rate": 5}):
            fig = hyp.plot(data, "-", reduce=None, show=False, **kw)
            counts.append(len(fig.axes[0].lines[0].get_xdata()))
            plt.close("all")
        assert counts[0] == counts[1] == counts[2]


# ---------------------------------------------------------------------------
# F01-002 / F01-003: fmt strings behave like matplotlib
# ---------------------------------------------------------------------------
class TestFmtColorContract:
    def test_ro_dash_draws_red_line_and_red_markers(self):
        X = np.random.default_rng(7).standard_normal((15, 3))
        fig = hyp.plot(X, "ro-", reduce=None, show=False)
        colors = [mcolors.to_hex(l.get_color()) for l in fig.axes[0].lines]
        plt.close("all")
        assert colors and all(c == "#ff0000" for c in colors)

    def test_combo_fmt_markers_match_own_line_color(self):
        rng = np.random.default_rng(7)
        L2 = [rng.standard_normal((15, 3)) for _ in range(2)]
        fig = hyp.plot(L2, "o-", reduce=None, show=False)
        lines = fig.axes[0].lines
        line_art = [l for l in lines if l.get_linestyle() == "-"]
        marker_art = [l for l in lines if l.get_linestyle() in ("None", "none")]
        assert len(line_art) == 2 and len(marker_art) == 2
        pair0 = (mcolors.to_hex(line_art[0].get_color()),
                 mcolors.to_hex(marker_art[0].get_color()))
        pair1 = (mcolors.to_hex(line_art[1].get_color()),
                 mcolors.to_hex(marker_art[1].get_color()))
        plt.close("all")
        # markers match their own line...
        assert pair0[0] == pair0[1]
        assert pair1[0] == pair1[1]
        # ...and the two datasets stay distinguishable
        assert pair0[0] != pair1[0]

    def test_bytes_fmt_accepted_like_str(self):
        """F01-017: plain-bytes fmt is decoded like np.bytes_."""
        X = np.random.default_rng(2).standard_normal((10, 3))
        fig = hyp.plot(X, b"-", reduce=None, show=False)
        assert len(fig.axes[0].lines) == 1
        plt.close("all")


# ---------------------------------------------------------------------------
# F01-004 / F08-001: plain python nested numeric lists
# ---------------------------------------------------------------------------
class TestNestedListInputs:
    def test_numeric_matrix_list_is_one_dataset(self):
        """[[1., 2.], [3., 4.]] is ONE dataset (a 2x2 matrix), matching
        np.array input."""
        fig = hyp.plot([[1.0, 2.0], [3.0, 4.0]], show=False)
        lines = [l for l in fig.axes[0].lines
                 if l.get_label() != "_nolegend_" or True]
        n_data_lines = len(fig.axes[0].lines)
        plt.close("all")

        fig2 = hyp.plot(np.array([[1.0, 2.0], [3.0, 4.0]]), show=False)
        n_expected = len(fig2.axes[0].lines)
        plt.close("all")
        assert n_data_lines == n_expected

    def test_numeric_matrix_matches_ndarray_render(self):
        mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        fig = hyp.plot(mat, ".", show=False)
        v1 = np.column_stack(fig.axes[0].lines[0].get_data_3d())
        plt.close("all")
        fig2 = hyp.plot(np.array(mat, dtype=float), ".", show=False)
        v2 = np.column_stack(fig2.axes[0].lines[0].get_data_3d())
        plt.close("all")
        assert np.allclose(v1, v2)

    def test_nested_group_form_unchanged(self):
        """The legit nested-GROUPS form (examples/plot_nested_lists.py) must
        keep its multilevel styling: leaves under the same outer group share
        a color."""
        def walk(seed):
            return np.cumsum(
                np.random.default_rng(seed).standard_normal((50, 6)), axis=0)

        fig = hyp.plot([[walk(0), walk(1)], [walk(2)]], show=False)
        lines = fig.axes[0].lines
        assert len(lines) == 3
        assert lines[0].get_color() == lines[1].get_color()
        assert lines[0].get_color() != lines[2].get_color()
        plt.close("all")


# ---------------------------------------------------------------------------
# F03-001 / F03-002: reduce= edge cases must not crash with NoneType-unpack
# ---------------------------------------------------------------------------
class TestReduceEdgeCases:
    def test_reduce_none_on_high_dim_data_raises_clear_error(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match=r"reduce=None"):
            hyp.plot(rng.standard_normal((30, 10)), reduce=None, show=False)
        plt.close("all")

    def test_reduce_instance_with_more_than_3_components_plots_3d(self):
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(0)
        # the 5-component instance deliberately conflicts with the default
        # ndims, provoking the dims/n_components notice
        with pytest.warns(UserWarning,
                          match='Unequal values passed to dims'):
            fig = hyp.plot(rng.standard_normal((30, 10)),
                           reduce=PCA(n_components=5), show=False)
        assert fig.axes[0].name == "3d"
        plt.close("all")

    def test_fitted_reducer_reused_without_reapplication(self):
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(1)
        model = PCA(n_components=3).fit(rng.standard_normal((40, 10)))
        new_data = rng.standard_normal((30, 10))
        fig = hyp.plot(new_data, reduce=model, show=False)
        assert fig.axes[0].name == "3d"
        plt.close("all")

    def test_legacy_reduce_params_dict_warns_once(self):
        """F03-009 (plot-side): one plot() call should warn once, not
        twice, for the deprecated {'model', 'params'} reduce spec."""
        rng = np.random.default_rng(0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hyp.plot(rng.standard_normal((20, 5)),
                     reduce={"model": "PCA", "params": {}}, show=False)
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)
               and "params" in str(x.message)]
        plt.close("all")
        assert len(dep) == 1


# ---------------------------------------------------------------------------
# F03-003 / F03-004 / D11-001 / D11-002: documented False values
# ---------------------------------------------------------------------------
class TestFalseStageValues:
    def test_align_false_is_no_alignment(self):
        rng = np.random.default_rng(0)
        data = [rng.standard_normal((20, 4)) for _ in range(2)]
        fig = hyp.plot(data, align=False, show=False)
        assert fig is not None
        plt.close("all")

    def test_cluster_false_is_no_clustering(self):
        rng = np.random.default_rng(0)
        fig = hyp.plot(rng.standard_normal((30, 4)), cluster=False, show=False)
        assert fig is not None
        plt.close("all")

    def test_cluster_instance_accepted(self):
        """F03-014: hyp.cluster accepts sklearn instances; plot must too."""
        from sklearn.cluster import KMeans
        rng = np.random.default_rng(0)
        data = np.vstack([rng.standard_normal((30, 4)) + off
                          for off in (0, 10, 20)])
        fig = hyp.plot(data, ".",
                       cluster=KMeans(n_clusters=3, n_init=10,
                                      random_state=0),
                       show=False)
        assert len(fig.axes[0].lines) == 3
        plt.close("all")


# ---------------------------------------------------------------------------
# Validation: clear errors naming the kwarg, before the pipeline runs
# ---------------------------------------------------------------------------
class TestKwargValidation:
    def test_legacy_group_kwarg_names_hue(self):
        X = np.random.default_rng(2).standard_normal((10, 3))
        with pytest.raises(TypeError, match="hue"):
            hyp.plot(X, group=["a"] * 10, show=False)
        plt.close("all")

    def test_misspelled_ndims_gets_did_you_mean(self):
        X = np.random.default_rng(2).standard_normal((10, 4))
        with pytest.raises(TypeError, match=r"n_dims.*ndims"):
            hyp.plot(X, fmt=".", n_dims=2, show=False)
        plt.close("all")

    def test_unknown_kwarg_clear_error(self):
        X = np.random.default_rng(2).standard_normal((10, 3))
        with pytest.raises(TypeError, match="frobnicate"):
            hyp.plot(X, frobnicate=3, show=False)
        plt.close("all")

    def test_fmt_list_too_short_raises_named_error(self):
        rng = np.random.default_rng(7)
        L3 = [rng.standard_normal((15, 3)) for _ in range(3)]
        with pytest.raises(ValueError, match=r"fmt.*2.*3"):
            hyp.plot(L3, fmt=["-", "o"], show=False)
        plt.close("all")

    def test_fmt_list_too_long_raises_named_error(self):
        rng = np.random.default_rng(7)
        L2 = [rng.standard_normal((15, 3)) for _ in range(2)]
        with pytest.raises(ValueError, match=r"fmt.*3.*2"):
            hyp.plot(L2, fmt=["-", "o", "."], show=False)
        plt.close("all")

    def test_labels_too_short_raises_named_error(self):
        X = np.random.default_rng(3).standard_normal((20, 3))
        with pytest.raises(ValueError, match=r"labels.*2.*20"):
            hyp.plot(X, ".", labels=["a", "b"], show=False)
        plt.close("all")

    def test_labels_too_long_raises_named_error(self):
        X = np.random.default_rng(3).standard_normal((20, 3))
        with pytest.raises(ValueError, match=r"labels.*50.*20"):
            hyp.plot(X, ".", labels=list("x" * 50), show=False)
        plt.close("all")

    def test_size_validated_by_name(self):
        X = np.random.default_rng(3).standard_normal((10, 3))
        with pytest.raises(ValueError, match="size"):
            hyp.plot(X, size="big", show=False)
        with pytest.raises(ValueError, match="size"):
            hyp.plot(X, size=[8], show=False)
        plt.close("all")

    def test_elev_azim_validated_by_name(self):
        X = np.random.default_rng(3).standard_normal((10, 3))
        with pytest.raises((TypeError, ValueError), match="elev"):
            hyp.plot(X, elev="up", show=False)
        with pytest.raises((TypeError, ValueError), match="azim"):
            hyp.plot(X, azim="left", show=False)
        plt.close("all")

    def test_ax_validated_by_name(self):
        X = np.random.default_rng(3).standard_normal((10, 3))
        with pytest.raises(TypeError, match="ax"):
            hyp.plot(X, ax="not-an-axes", show=False)
        plt.close("all")

    def test_explore_on_2d_raises_value_error(self):
        X = np.random.default_rng(2).standard_normal((15, 3))
        with pytest.raises(ValueError, match="[Ee]xplore"):
            hyp.plot(X, ndims=2, explore=True, show=False)
        plt.close("all")

    def test_mixed_column_counts_raise_clear_error(self):
        rng = np.random.default_rng(9)
        with pytest.raises(ValueError, match=r"column|feature"):
            hyp.plot([rng.standard_normal((20, 3)),
                      rng.standard_normal((20, 5))], show=False)
        plt.close("all")

    def test_legend_scalar_string_and_wrong_length_named(self):
        rng = np.random.default_rng(1)
        data = [rng.standard_normal((10, 3)) for _ in range(2)]
        with pytest.raises(ValueError, match="legend"):
            hyp.plot(data, legend=["only-one"], show=False)
        with pytest.raises(ValueError, match="legend"):
            hyp.plot(data, legend="groups", show=False)
        plt.close("all")

    def test_names_scalar_string_not_split_into_characters(self):
        rng = np.random.default_rng(1)
        data = [rng.standard_normal((10, 3)) for _ in range(2)]
        with pytest.raises(ValueError, match="names"):
            hyp.plot(data, names="ab", show=False)
        plt.close("all")

    def test_rgb_tuple_color_is_one_color(self):
        """F10-004: a bare (r, g, b) tuple is a single matplotlib color."""
        X = np.random.default_rng(3).standard_normal((10, 3))
        fig = hyp.plot(X, color=(0.2, 0.4, 0.6), show=False)
        col = mcolors.to_hex(fig.axes[0].lines[0].get_color())
        plt.close("all")
        assert col == mcolors.to_hex((0.2, 0.4, 0.6))

    def test_scalar_input_warns(self):
        """D11-014: a bare scalar is plotted as a single point, with a
        warning rather than silently."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = hyp.plot(5, show=False)
        plt.close("all")
        assert fig is not None
        assert any("scalar" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# fmt=None + hue regrouping (F01-005)
# ---------------------------------------------------------------------------
class TestFmtNoneWithHue:
    def test_fmt_none_with_hue_on_list_input(self):
        X = np.random.default_rng(3).standard_normal((20, 3))
        fig = hyp.plot([X], fmt=None, hue=["a"] * 10 + ["b"] * 10, show=False)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Warning hygiene (F01-009, F08-014, F10-007, F10-008)
# ---------------------------------------------------------------------------
class TestWarningHygiene:
    def test_linestyle_kwarg_no_redundant_definition_warning(self):
        X = np.random.default_rng(3).standard_normal((20, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hyp.plot(X, linestyle="--", show=False)
        plt.close("all")
        assert not [x for x in w if "redundantly defined" in str(x.message)]

    def test_linestyles_list_no_redundant_definition_warning(self):
        rng = np.random.default_rng(3)
        data = [rng.standard_normal((20, 3)) for _ in range(2)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hyp.plot(data, linestyles=["-", "--"], show=False)
        plt.close("all")
        assert not [x for x in w if "redundantly defined" in str(x.message)]

    def test_hue_color_conflict_warning_names_hue(self):
        X = np.random.default_rng(3).standard_normal((20, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hyp.plot(X, ".", hue=["a"] * 10 + ["b"] * 10, color="red",
                     show=False)
        plt.close("all")
        msgs = [str(x.message) for x in w if "color" in str(x.message)]
        assert msgs
        assert any("hue" in m for m in msgs)
        assert not any("group" in m for m in msgs)

    def test_alias_warnings_have_no_embedded_whitespace_runs(self):
        X = np.random.default_rng(3).standard_normal((10, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hyp.plot(X, ".", marker="o", markers="s", show=False)
            hyp.plot(X, color="red", colors="blue", show=False)
            hyp.plot(X, linestyle="-", linestyles="--", show=False)
        plt.close("all")
        for x in w:
            assert "   " not in str(x.message)


# ---------------------------------------------------------------------------
# Continuous hue endpoints (F01-013 / D11-015)
# ---------------------------------------------------------------------------
class TestContinuousHueEndpoints:
    def test_hls_continuous_endpoints_distinguishable(self):
        c = np.asarray(mat2colors(np.arange(100).astype(float),
                                  palette="hls"))
        d_ends = np.linalg.norm(c[0] - c[-1])
        assert d_ends > 0.25  # was 0.031 with the full hue circle

    def test_colorbar_colormap_matches_mat2colors(self):
        cmap = continuous_colormap("hls")
        vals = np.arange(100).astype(float)
        cols = np.asarray(mat2colors(vals, palette="hls"))
        # first and last colorbar colors match the drawn extremes
        assert np.allclose(cmap(0.0)[:3], cols[0], atol=1e-6)
        assert np.allclose(cmap(1.0)[:3], cols[-1], atol=1e-6)

    def test_categorical_palette_unchanged(self):
        import seaborn as sns
        cols = np.asarray(mat2colors(["a", "b", "c"], palette="hls"))
        base = np.asarray(sns.color_palette("hls", 3))[:, :3]
        assert np.allclose(cols, base)


# ---------------------------------------------------------------------------
# Density degeneracy detection (D11-009)
# ---------------------------------------------------------------------------
class TestDensityDegenerate:
    @pytest.mark.parametrize("point", [
        [1.0, 2.0, 3.0],
        [-1.26542147, -0.62327446, 0.04132598],
    ])
    def test_identical_points_always_detected(self, point):
        pts = np.tile(np.asarray(point), (30, 1))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kde = fit_kde(pts)
        assert kde is None
        assert any("degenerate" in str(x.message) for x in w)

    def test_full_rank_data_still_fits(self):
        pts = np.random.default_rng(0).standard_normal((50, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kde = fit_kde(pts)
        assert kde is not None
        assert not w


# ---------------------------------------------------------------------------
# Animation label preservation (F10-001)
# ---------------------------------------------------------------------------
class TestAnimationLabels:
    def test_expand_labels_decimation_keeps_labels(self):
        labels = [None] * 30
        labels[0] = "L0"
        labels[15] = "L15"
        out = _expand_labels(labels, [30], [5])
        assert "L0" in out and "L15" in out

    def test_animated_labels_survive_frame_subsampling(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((30, 3))
        labels = [None] * 30
        labels[0] = "L0"
        labels[15] = "L15"
        out = hyp.plot(A, animate=True, duration=1.0, frame_rate=5,
                       labels=labels, show=False)
        fig = out[0]
        texts = [t for ax in fig.axes for t in ax.texts]
        n_annotations = len([t for t in texts
                             if t.get_text() in ("L0", "L15")])
        plt.close("all")
        assert n_annotations == 2


# ---------------------------------------------------------------------------
# Docstring accuracy (F01-008, F01-018, F03-010, F08-004, F10-002/006/016,
# D11-004/005/006/007/012/016/017)
# ---------------------------------------------------------------------------
class TestDocstringAccuracy:
    doc = hyp.plot.__doc__

    def test_no_geo_in_x_type_line(self):
        assert "Geo" not in self.doc

    def test_elev_azim_documented(self):
        assert "elev :" in self.doc
        assert "azim :" in self.doc

    def test_examples_section_present(self):
        assert "Examples" in self.doc
        assert ">>>" in self.doc

    def test_rescale_documented(self):
        assert "[-1, 1]" in self.doc

    def test_no_stale_return_tuple_claim(self):
        assert "axis and data objects will still be returned" not in self.doc

    def test_no_stale_ndims_return_claim(self):
        assert "but return the higher\n        dimensional data" not in self.doc

    def test_no_phantom_params(self):
        assert "vectorizer_params" not in self.doc
        assert "text_params" not in self.doc

    def test_hover_sentence_fixed(self):
        assert ("Displays user defined labels will appear on hover"
                not in self.doc)

    def test_hyper_animation_documented(self):
        assert "HyperAnimation" in self.doc

    def test_palette_list_documented(self):
        import re
        m = re.search(r"palette : ([^\n]*)", self.doc)
        assert m and "list" in m.group(1)

    def test_hue_dtype_rule_documented(self):
        # the docstring must state that NUMERIC hue values take the
        # continuous path (use strings for categorical grouping)
        idx = self.doc.index("hue : ")
        section = self.doc[idx:idx + 2500]
        assert "numeric" in section and "categorical" in section
        assert "string" in section

    def test_examples_run(self):
        """The Examples section is runnable (real code, no mocks)."""
        import doctest
        results = doctest.testmod(
            m=__import__("hypertools.plot.plot", fromlist=["plot"]),
            verbose=False)
        plt.close("all")
        assert results.attempted > 0
        assert results.failed == 0

    def test_plotly_draw_params_documented(self):
        """D11-013: plotly_draw documents its parameters."""
        import inspect
        from hypertools.plot.plotly_backend import plotly_draw
        doc = plotly_draw.__doc__
        sig = inspect.signature(plotly_draw)
        documented = [p for p in sig.parameters if f"{p} :" in doc]
        assert len(documented) >= len(list(sig.parameters)) - 1


# ---------------------------------------------------------------------------
# zlabel / legend overlap (F10-005)
# ---------------------------------------------------------------------------
class TestZlabelLegendLayout:
    def test_zlabel_does_not_overlap_legend(self):
        rng = np.random.default_rng(0)
        data = [rng.standard_normal((20, 3)) for _ in range(2)]
        fig = hyp.plot(data, names=["subject 1", "subject 2"],
                       xlabel="PC1", ylabel="PC2", zlabel="PC3", show=False)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        leg_bbox = legend.get_window_extent(renderer)
        zaxis_label = ax.zaxis.label
        z_bbox = zaxis_label.get_window_extent(renderer)
        plt.close("all")
        assert not leg_bbox.overlaps(z_bbox)
