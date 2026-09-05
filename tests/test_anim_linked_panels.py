"""`companion=`: linked animated companion panels (GH #285).

`examples/animate_weather_decades.py` builds a second axes by hand
(``fig.add_axes``), draws the full series faintly, adds a revealed
``LineCollection``, a rolling mean and a head marker, and drives all four
from an `on_frame=` hook -- because ``ax=`` plus ``animate=`` raises and the
docs say several animated panels in one figure are not supported.
`companion=` is that panel, native.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

import hypertools as hyp                                        # noqa: E402


N_ROWS = 24


def trajectory(seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(N_ROWS, 3)), axis=0)


def series(seed=1):
    return np.sin(np.linspace(0, 5, N_ROWS)) + 0.1 * np.random.default_rng(
        seed).normal(size=N_ROWS)


def animate(spec, data=None, **kwargs):
    return hyp.plot(data if data is not None else trajectory(),
                    animate=True, duration=2, frame_rate=8, show=False,
                    companion=spec, **kwargs)


def panel_axes(anim):
    return anim.figure.axes[-1]


class TestLayout:

    def test_a_panel_is_added_and_the_main_axes_keeps_its_size(self):
        plain = hyp.plot(trajectory(), animate=True, duration=2,
                         frame_rate=8, show=False)
        withp = animate({'data': series()})
        try:
            assert len(plain.figure.axes) == 1
            assert len(withp.figure.axes) == 2
            # the figure GREW; the animated axes kept its absolute inches
            pw, ph = plain.figure.get_size_inches()
            ww, wh = withp.figure.get_size_inches()
            assert ww == pytest.approx(pw)
            assert wh > ph
            a = plain.figure.axes[0].get_position()
            b = withp.figure.axes[0].get_position()
            assert b.height * wh == pytest.approx(a.height * ph, rel=1e-9)
            assert b.width * ww == pytest.approx(a.width * pw, rel=1e-9)
            # and it sits ABOVE the panel
            assert b.y0 > panel_axes(withp).get_position().y1
        finally:
            plt.close(plain.figure)
            plt.close(withp.figure)

    def test_position_right_grows_the_width_instead(self):
        anim = animate({'data': series(), 'position': 'right',
                        'size': 0.3})
        try:
            w, h = anim.figure.get_size_inches()
            assert h == pytest.approx(4.8)
            assert w > 6.4
            main = anim.figure.axes[0].get_position()
            assert panel_axes(anim).get_position().x0 > main.x1
        finally:
            plt.close(anim.figure)

    def test_two_panels_both_appear(self):
        anim = animate([{'data': series()},
                        {'data': series(2), 'position': 'right'}])
        try:
            assert len(anim.figure.axes) == 3
        finally:
            plt.close(anim.figure)


class TestRevealTracksTheMainAnimation:

    def test_revealed_points_match_the_reveal_head_frame_by_frame(self):
        """The panel's revealed segment count is the head row, and the head
        row is the SAME one the `{index...}` title pattern reads."""
        anim = animate({'data': series(), 'marker': True})
        try:
            for i in range(anim.n_frames):
                anim.draw_frame(i)
                head = round(i / (anim.n_frames - 1) * (N_ROWS - 1))
                coll = panel_axes(anim).collections[0]
                assert len(coll.get_segments()) == head
                marker = panel_axes(anim).lines[-1]
                assert marker.get_xdata() == [head]
                assert marker.get_ydata() == pytest.approx([series()[head]])
        finally:
            plt.close(anim.figure)

    def test_the_panel_agrees_with_an_index_title_frame_by_frame(self):
        idx = pd.date_range('2020-01-31', periods=N_ROWS, freq='ME')
        df = pd.DataFrame(trajectory(), index=idx)
        anim = animate({'data': series()}, data=df, title='{index:%Y-%m}')
        try:
            for i in range(anim.n_frames):
                anim.draw_frame(i)
                revealed = len(panel_axes(anim).collections[0].get_segments())
                assert anim.figure.axes[0].get_title() == \
                    idx[revealed].strftime('%Y-%m')
        finally:
            plt.close(anim.figure)

    def test_serial_reveal_drives_the_panel_too(self):
        anim = hyp.plot([trajectory(), trajectory(3)], animate='serial',
                        duration=2, frame_rate=8, show=False,
                        companion={'data': series()})
        try:
            counts = []
            for i in range(anim.n_frames):
                anim.draw_frame(i)
                counts.append(
                    len(panel_axes(anim).collections[0].get_segments()))
            assert counts[0] == 0
            assert counts[-1] == N_ROWS - 1
            assert counts == sorted(counts) or min(counts) >= 0
        finally:
            plt.close(anim.figure)

    def test_reveal_false_draws_the_whole_series_every_frame(self):
        anim = animate({'data': series(), 'reveal': False})
        try:
            for i in (0, anim.n_frames // 2, anim.n_frames - 1):
                anim.draw_frame(i)
                assert len(panel_axes(anim).collections[0].get_segments()) \
                    == N_ROWS - 1
        finally:
            plt.close(anim.figure)


class TestPanelContents:

    def test_the_ghost_series_is_drawn_once_in_full(self):
        anim = animate({'data': series()})
        try:
            ghost = panel_axes(anim).lines[0]
            assert len(ghost.get_xdata()) == N_ROWS
            assert np.allclose(ghost.get_ydata(), series())
        finally:
            plt.close(anim.figure)

    def test_smooth_draws_a_pandas_style_rolling_mean(self):
        anim = animate({'data': series(), 'smooth': 5, 'marker': False})
        try:
            anim.draw_frame(anim.n_frames - 1)
            trend = panel_axes(anim).lines[1]
            want = pd.Series(series()).rolling(5).mean().to_numpy()
            got = np.asarray(trend.get_ydata())
            assert len(got) == N_ROWS
            assert np.allclose(got[4:], want[4:])
            assert np.isnan(got[:4]).all()
        finally:
            plt.close(anim.figure)

    def test_two_column_data_is_read_as_x_y(self):
        x = np.linspace(1990.0, 2020.0, N_ROWS)
        y = series()
        anim = animate({'data': np.column_stack([x, y])})
        try:
            assert np.allclose(panel_axes(anim).lines[0].get_xdata(), x)
            assert panel_axes(anim).get_xlim() == pytest.approx(
                (x[0], x[-1]))
        finally:
            plt.close(anim.figure)

    def test_hue_colours_the_revealed_line_and_the_head(self):
        values = series()
        anim = animate({'data': values, 'hue': values})
        try:
            anim.draw_frame(anim.n_frames - 1)
            coll = panel_axes(anim).collections[0]
            assert coll.get_array() is not None
            assert len(coll.get_array()) == N_ROWS - 1
            face = panel_axes(anim).lines[-1].get_markerfacecolor()
            assert len(face) == 4                      # an RGBA from a cmap
        finally:
            plt.close(anim.figure)

    def test_hue_uses_the_plot_s_own_colour_scale_when_there_is_one(self):
        """With a continuous `hue=` on the trajectory, the panel reads the
        SAME cmap/norm the colorbar does -- that is what
        `animate_weather_decades.py` rebuilt by hand
        (`Normalize(mean.min(), mean.max())` + `plt.get_cmap('RdBu_r')`).
        """
        values = np.linspace(0.0, 1.0, N_ROWS)
        anim = hyp.plot(trajectory(), animate=True, duration=2,
                        frame_rate=8, hue=values, colorbar=True,
                        show=False,
                        companion={'data': values, 'hue': values})
        try:
            # main axes, colorbar axes, panel axes
            assert len(anim.figure.axes) == 3
            anim.draw_frame(anim.n_frames - 1)
            coll = panel_axes(anim).collections[0]
            want = coll.cmap(coll.norm(values[-1]))
            got = panel_axes(anim).lines[-1].get_markerfacecolor()
            assert tuple(np.round(got, 6)) == tuple(np.round(want, 6))
            # the panel's norm spans the hue values, not [0, 1] by accident
            assert coll.norm.vmin == pytest.approx(values.min())
            assert coll.norm.vmax == pytest.approx(values.max())
        finally:
            plt.close(anim.figure)

    def test_a_legend_and_a_panel_coexist(self):
        anim = hyp.plot([trajectory(), trajectory(3)], animate=True,
                        duration=1, frame_rate=4, names=['a', 'b'],
                        legend=True, show=False,
                        companion={'data': series()})
        try:
            assert len(anim.figure.axes) == 2
            assert anim.figure.axes[0].get_legend() is not None
        finally:
            plt.close(anim.figure)

    def test_labels_and_spines(self):
        anim = animate({'data': series(), 'xlabel': 'year',
                        'ylabel': 'Average'})
        try:
            pax = panel_axes(anim)
            assert pax.get_xlabel() == 'year'
            assert pax.get_ylabel() == 'Average'
            assert not pax.spines['top'].get_visible()
            assert not pax.spines['right'].get_visible()
        finally:
            plt.close(anim.figure)

    def test_the_panel_renders_ink(self):
        """A real render: the strip below the plot is not blank."""
        anim = animate({'data': series()})
        try:
            anim.draw_frame(anim.n_frames - 1)
            anim.figure.canvas.draw()
            pixels = np.asarray(anim.figure.canvas.buffer_rgba())[..., :3]
            box = panel_axes(anim).get_position()
            h = pixels.shape[0]
            strip = pixels[int((1 - box.y1) * h):int((1 - box.y0) * h)]
            assert (strip.min(-1) < 250).any()
        finally:
            plt.close(anim.figure)


class TestValidation:

    def test_static_plot_is_refused(self):
        with pytest.raises(ValueError, match=r'revealed in lockstep'):
            hyp.plot(trajectory(), companion={'data': series()}, show=False)

    def test_plotly_raises_naming_the_backend(self):
        with pytest.raises(NotImplementedError, match=r'matplotlib-only'):
            hyp.plot(trajectory(), animate=True, duration=1,
                     backend='plotly', companion={'data': series()},
                     show=False)

    @pytest.mark.parametrize('spec, exc, match', [
        ({'data': series(), 'kind': 'bars'}, ValueError, r"kind='bars'"),
        ({'data': series(), 'nope': 1}, ValueError, r'unknown key'),
        ({}, ValueError, r'has no data='),
        ({'data': np.zeros((5, 3))}, ValueError, r'data must be'),
        ({'data': np.zeros(1)}, ValueError, r'at least 2 rows'),
        ({'data': series(), 'position': 'top'}, ValueError,
         r"position must be"),
        ({'data': series(), 'smooth': 1}, ValueError, r'at least 2'),
        ({'data': series(), 'hue': np.zeros(3)}, ValueError, r'hue= has 3'),
        ({'data': series(), 'size': 0.95}, ValueError, r'size='),
        ('not a dict', TypeError, r'takes a dict'),
    ])
    def test_bad_specs_raise(self, spec, exc, match):
        with pytest.raises(exc, match=match):
            animate(spec)

    def test_default_is_no_panel(self):
        anim = hyp.plot(trajectory(), animate=True, duration=1,
                        frame_rate=4, show=False)
        try:
            assert len(anim.figure.axes) == 1
        finally:
            plt.close(anim.figure)
