"""`panels=` follow-ups from converting the gallery (GH #285).

Three things a real gallery conversion turned up:

1. `_plot_panels` called `plt.show()` unconditionally, so every panel grid
   built under a non-interactive backend (Agg -- i.e. every docs build)
   warned "FigureCanvasAgg is non-interactive, and thus cannot be shown".
   The single-axes path never calls `plt.show()`.
2. `panels=` dropped the per-panel axis labels a single
   ``hyp.plot(df, ax=ax)`` derives from DataFrame column names (seen on
   `examples/plot_datasaurus.py`: 13 frames with x/y columns lost every
   label).
3. `panels=` always shared one pipeline fit, which skews panels drawn from
   clouds of very different scale (`examples/plot_shapes_zoo.py`) --
   `panel_fit='independent'` fits per panel instead.

Lives in its own file so it does not collide with `test_plot_panels.py`.
"""

import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

import hypertools as hyp                                        # noqa: E402


def clouds(n_sets=2, n_rows=30, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n_rows, 5)) * (10.0 ** i) + 40.0 * i
            for i in range(n_sets)]


def drawn(ax):
    return np.asarray(ax.lines[0].get_data_3d()).T


class TestNoNonInteractiveShowWarning:

    def test_panels_with_show_true_warns_nothing_under_agg(self):
        assert matplotlib.get_backend().lower() == 'agg'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig = hyp.plot(clouds(), panels=True)
        try:
            messages = [str(w.message) for w in caught]
            assert not any('non-interactive' in m for m in messages), messages
            assert len(fig.axes) == 2
        finally:
            plt.close(fig)

    def test_the_figure_is_still_registered_with_pyplot(self):
        """`show=True` leaves the figure open for the notebook/IPython
        flush, exactly as the single-axes path does."""
        before = set(plt.get_fignums())
        fig = hyp.plot(clouds(), panels=True)
        try:
            assert set(plt.get_fignums()) - before == {fig.number}
        finally:
            plt.close(fig)

    def test_show_false_still_deregisters(self):
        before = set(plt.get_fignums())
        fig = hyp.plot(clouds(), panels=True, show=False)
        try:
            assert set(plt.get_fignums()) == before
            # and the figure is still renderable/savable
            fig.canvas.draw()
        finally:
            plt.close(fig)


class TestPerPanelAxisLabels:

    def frames(self):
        t = np.arange(20.0)
        return [pd.DataFrame({'x': t, 'y': np.sin(t)}),
                pd.DataFrame({'x': t, 'y': np.cos(t)})]

    def test_panels_keep_the_labels_a_single_call_derives(self):
        a, b = self.frames()
        single = hyp.plot(a, ndims=2, show=False)
        try:
            want = (single.axes[0].get_xlabel(),
                    single.axes[0].get_ylabel())
        finally:
            plt.close(single)
        assert want == ('x', 'y')
        fig = hyp.plot([a, b], panels=True, ndims=2, show=False)
        try:
            for ax in fig.axes:
                assert (ax.get_xlabel(), ax.get_ylabel()) == want
        finally:
            plt.close(fig)

    def test_labels_follow_the_column_names(self):
        """Not 'x'/'y' by luck: rename the columns and the labels follow.

        (One list of DataFrames must share its column names -- hypertools
        matches features by name and refuses a mismatched list upstream --
        so this renames BOTH frames.)
        """
        t = np.arange(20.0)
        data = [pd.DataFrame({'year': t, 'temp': np.sin(t)}),
                pd.DataFrame({'year': t, 'temp': np.cos(t)})]
        fig = hyp.plot(data, panels=True, ndims=2, show=False)
        try:
            assert [(ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes] \
                == [('year', 'temp'), ('year', 'temp')]
        finally:
            plt.close(fig)

    def test_explicit_labels_still_win(self):
        a, b = self.frames()
        fig = hyp.plot([a, b], panels=True, ndims=2, xlabel='mine',
                       show=False)
        try:
            assert all(ax.get_xlabel() == 'mine' for ax in fig.axes)
            assert all(ax.get_ylabel() == 'y' for ax in fig.axes)
        finally:
            plt.close(fig)

    def test_plain_arrays_get_no_labels(self):
        fig = hyp.plot(clouds(), panels=True, ndims=2, show=False)
        try:
            assert all(ax.get_xlabel() == '' for ax in fig.axes)
        finally:
            plt.close(fig)

    def test_independent_fit_derives_them_too(self):
        a, b = self.frames()
        fig = hyp.plot([a, b], panels=True, panel_fit='independent',
                       ndims=2, show=False)
        try:
            assert [(ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes] \
                == [('x', 'y'), ('x', 'y')]
        finally:
            plt.close(fig)


class TestPanelFit:

    def test_independent_panels_equal_the_single_call(self):
        """Coordinate-for-coordinate: a panel IS the per-panel
        ``hyp.plot(x[i], ax=ax)`` it replaces."""
        data = clouds()
        fig = hyp.plot(data, panels=True, panel_fit='independent',
                       show=False)
        try:
            got = [drawn(ax) for ax in fig.axes]
        finally:
            plt.close(fig)
        for i, d in enumerate(data):
            single = hyp.plot(d, show=False)
            try:
                want = drawn(single.axes[0])
            finally:
                plt.close(single)
            assert got[i].shape == want.shape
            assert np.abs(got[i] - want).max() < 1e-9

    def test_shared_is_the_default_and_is_not_the_independent_fit(self):
        data = clouds()
        default = hyp.plot(data, panels=True, show=False)
        shared = hyp.plot(data, panels=True, panel_fit='shared', show=False)
        indep = hyp.plot(data, panels=True, panel_fit='independent',
                         show=False)
        try:
            for ax_a, ax_b in zip(default.axes, shared.axes):
                assert np.allclose(drawn(ax_a), drawn(ax_b))
            # clouds two orders of magnitude apart really are drawn
            # differently by the two modes
            assert not np.allclose(drawn(default.axes[0]),
                                   drawn(indep.axes[0]))
        finally:
            for f in (default, shared, indep):
                plt.close(f)

    def test_per_dataset_kwargs_are_still_narrowed(self):
        fig = hyp.plot(clouds(3), panels=True, panel_fit='independent',
                       color=['red', 'green', 'blue'], show=False)
        try:
            # a near-square grid for 3 panels has one hidden spare cell
            panels = [ax for ax in fig.axes if ax.get_visible()]
            assert [ax.lines[0].get_color() for ax in panels] == \
                ['red', 'green', 'blue']
        finally:
            plt.close(fig)

    def test_titles_still_work(self):
        fig = hyp.plot(clouds(), panels=True, panel_fit='independent',
                       title=['left', 'right'], show=False)
        try:
            assert [ax.get_title() for ax in fig.axes] == ['left', 'right']
        finally:
            plt.close(fig)

    def test_return_model_bundle_still_shaped_the_same(self):
        bundle = hyp.plot(clouds(), panels=True, panel_fit='independent',
                          return_model=True, show=False)
        try:
            assert set(bundle) >= {'fig', 'axes', 'panels', 'panel_models',
                                   'xform_data'}
            assert len(bundle['xform_data']) == 2
        finally:
            plt.close(bundle['fig'])

    def test_bad_value_raises(self):
        with pytest.raises(ValueError, match=r"panel_fit= must be"):
            hyp.plot(clouds(), panels=True, panel_fit='nope', show=False)

    def test_independent_with_a_reducer_list_raises(self):
        with pytest.raises(ValueError, match=r'not meaningful'):
            hyp.plot(clouds(), panels=True, panel_fit='independent',
                     reduce=['PCA', 'IncrementalPCA'], show=False)

    def test_panel_fit_is_ignored_without_panels(self):
        fig = hyp.plot(clouds()[0], panel_fit='independent', show=False)
        try:
            assert len(fig.axes) == 1
        finally:
            plt.close(fig)

    def test_plotly_independent_grid_builds(self):
        fig = hyp.plot(clouds(), panels=True, panel_fit='independent',
                       backend='plotly', show=False)
        assert len(fig.data) >= 2


class TestPlotlyPanelBundle:
    """`return_model=True` under `backend='plotly'` used to hand back a bare
    `go.Figure` (every inner call was forced to ``return_model=False``), so
    ``bundle['panels']`` raised a KeyError out of
    ``BaseFigure.__getitem__`` -- a wrong-shape return surfacing as a plotly
    internals error.
    """

    def test_the_plotly_bundle_has_the_matplotlib_bundle_s_keys(self):
        import plotly.io as pio
        pio.renderers.default = 'json'
        data = clouds()
        mpl = hyp.plot(data, panels=True, reduce=['PCA', 'UMAP'],
                       return_model=True, show=False)
        try:
            want = set(mpl)
            n_axes, n_models = len(mpl['axes']), len(mpl['panel_models'])
            grid = mpl['panels']
        finally:
            plt.close(mpl['fig'])
        got = hyp.plot(data, panels=True, reduce=['PCA', 'UMAP'],
                       return_model=True, backend='plotly', show=False)
        assert isinstance(got, dict)
        assert set(got) == want
        assert got['panels'] == grid
        assert len(got['axes']) == n_axes
        assert len(got['panel_models']) == n_models
        assert len(got['xform_data']) == 2
        assert all(np.asarray(a).shape[-1] == 3 for a in got['xform_data'])

    def test_three_d_axes_are_the_scene_objects(self):
        import plotly.io as pio
        pio.renderers.default = 'json'
        bundle = hyp.plot(clouds(), panels=True, return_model=True,
                          backend='plotly', show=False)
        assert bundle['axes'][0] is bundle['fig'].layout.scene
        assert bundle['axes'][1] is bundle['fig'].layout.scene2

    def test_two_d_axes_are_xaxis_yaxis_pairs(self):
        import plotly.io as pio
        pio.renderers.default = 'json'
        bundle = hyp.plot(clouds(), panels=True, ndims=2, return_model=True,
                          backend='plotly', show=False)
        assert bundle['axes'][0] == (bundle['fig'].layout.xaxis,
                                     bundle['fig'].layout.yaxis)

    def test_without_return_model_it_is_still_a_bare_figure(self):
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.renderers.default = 'json'
        fig = hyp.plot(clouds(), panels=True, backend='plotly', show=False)
        assert isinstance(fig, go.Figure)


# --- 1.1 release-review fixes (P1-P3) -----------------------------------

def _walks(seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(30, 2)), axis=0),
            np.cumsum(rng.normal(size=(40, 2)), axis=0)]


def _by_role(ax, role):
    return [line for line in ax.lines
            if getattr(line, '_hyp_forecast_role', None) == role]


@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_P1_panels_with_predict_and_truth_in_both_fit_modes(panel_fit):
    """The shared probe kept `truth=` while dropping `predict=` ("truth=
    ... no forecast was requested"), and the independent mode never
    narrowed the truth list ("1 trace(s) plotted, truth= supplies 2")."""
    x, y = _walks()
    rng = np.random.default_rng(1)
    truths = [x[-1] + np.cumsum(rng.normal(size=(3, 2)), axis=0),
              y[-1] + np.cumsum(rng.normal(size=(3, 2)), axis=0)]
    fig = hyp.plot([x, y], reduce=None, ndims=2, panels=True,
                   panel_fit=panel_fit, predict='Kalman', t=3, truth=truths,
                   forecast_hue=['a', 'b'], axis_scale='data',
                   antialias=False, show=False)
    try:
        assert len(fig.axes) == 2
        for ax, truth in zip(fig.axes, truths):
            (forecast,) = _by_role(ax, 'static')
            drawn_truths = _by_role(ax, 'truth')
            assert len(np.asarray(forecast.get_xdata())) == 4   # seam + 3
            # each panel draws ITS dataset's held-out continuation (the
            # truth overlay is a line plus its marker artist)
            assert len(drawn_truths) >= 1
            for drawn_truth in drawn_truths:
                assert np.allclose(
                    np.asarray(drawn_truth.get_xydata())[1:], truth)
    finally:
        plt.close(fig)


def _dated(cols, seed):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2020-01-01', periods=30, freq='D')
    names = ['val'] if cols == 1 else list('abc')[:cols]
    return pd.DataFrame(np.cumsum(rng.normal(size=(30, cols)), axis=0),
                        index=index, columns=names)


@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_P2_series_mode_panels_keep_dates_and_the_column_name(panel_fit):
    from matplotlib.dates import date2num
    frames = [_dated(1, 2), _dated(1, 3)]
    fig = hyp.plot(frames, ndims=1, panels=True, panel_fit=panel_fit,
                   antialias=False, show=False)
    try:
        for ax, frame in zip(fig.axes, frames):
            (line,) = [ln for ln in ax.lines
                       if getattr(ln, '_hyp_forecast_role', None) is None]
            assert np.allclose(np.asarray(line.get_xdata(), dtype=float),
                               date2num(frame.index.to_pydatetime()))
            assert ax.get_ylabel() == 'val'
    finally:
        plt.close(fig)


@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_P2_a_three_column_frame_in_series_mode_panels(panel_fit):
    """The panel path assigned a 3-column frame's names to x/y/zlabel, and
    the series-mode panel then refused zlabel= for 2-D data."""
    frames = [_dated(3, 4), _dated(3, 5)]
    fig = hyp.plot(frames, ndims=1, reduce=None, panels=True,
                   panel_fit=panel_fit, antialias=False, show=False)
    try:
        for ax in fig.axes:
            lines = [ln for ln in ax.lines
                     if getattr(ln, '_hyp_forecast_role', None) is None]
            assert len(lines) == 3
            assert ax.get_zorder() is not None and not hasattr(ax, 'zaxis')
    finally:
        plt.close(fig)


@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_P3_nested_per_dataset_hue_is_narrowed_per_panel(panel_fit):
    x, y = _walks(seed=6)
    hue = [np.linspace(0.0, 1.0, 30), np.linspace(5.0, 9.0, 40)]
    bundle = hyp.plot([x, y], reduce=None, ndims=2, panels=True,
                      panel_fit=panel_fit, hue=hue, return_model=True,
                      show=False)
    try:
        assert len(bundle['axes']) == 2
        # each panel's colour scale spans ITS OWN hue values
        for model, values in zip(bundle['panel_models'], hue):
            colors = model['colors']
            assert colors['vmin'] == pytest.approx(values.min())
            assert colors['vmax'] == pytest.approx(values.max())
    finally:
        plt.close(bundle['fig'])


@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_P3_labels_are_narrowed_per_panel(panel_fit):
    x, y = _walks(seed=7)
    fig = hyp.plot([x, y], reduce=None, ndims=2, panels=True,
                   panel_fit=panel_fit, labels=['A', 'B'], show=False)
    try:
        texts = [[t.get_text() for t in ax.texts] for ax in fig.axes]
        assert texts == [['A'], ['B']]
    finally:
        plt.close(fig)
    nested = [[None] * 29 + ['end x'], [None] * 39 + ['end y']]
    fig = hyp.plot([x, y], reduce=None, ndims=2, panels=True,
                   panel_fit=panel_fit, labels=nested, show=False)
    try:
        texts = [[t.get_text() for t in ax.texts] for ax in fig.axes]
        assert texts == [['end x'], ['end y']]
    finally:
        plt.close(fig)
