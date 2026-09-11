"""Animation-core fidelity (1.1 visual review, findings L8, L9, L11, L12,
L13b).

Every check reads what was actually DRAWN -- artist vertices, plotly frame
payloads, text extents against the canvas, rendered colours, emitted
warnings -- never a re-derivation of the library's own arithmetic.

* L8: an animated line used to be resampled onto exactly
  ``round(frame_rate * duration)`` rows, which DOWNsampled any dataset longer
  than the frame count (a 36-sample helix in a 9-frame animation was drawn
  as a zig-zag star). Every observation must now be an exact vertex of the
  animated line, and ``'spin'`` (which reveals nothing) keeps its rows.
* L9: morph transitions sampled their own endpoints, so a short transition
  showed no motion at all (both of its frames were copies of the hold
  clouds). Every transition frame must now lie strictly between the clouds.
* L11: a title set by an `on_frame` callback on a 3-D animation without
  ``title=`` rendered above the canvas.
* L12: `companion=` panels defaulted to matplotlib's global ``'C0'`` blue
  instead of the colour of the trajectory they accompany.
* L13b: the clamped-samples warning of a streamed plot never fired on a
  short stream (it needed 20 post-head samples).
"""

import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.colors import to_hex, to_rgb                    # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot import morph                               # noqa: E402

BACKENDS = ['matplotlib', 'plotly']


def helix(n=36, turns=3.0, radius=0.5):
    """A helix about the z axis: its true radius is `radius` at every row,
    which makes any shape distortion measurable as a radius spread."""
    t = np.linspace(0.0, 2 * np.pi * turns, n)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t),
                            np.linspace(0.0, 1.0, n)])


def _collect(data, backend, **kwargs):
    """Plot with an `on_frame` recorder; return (figure-like, contexts).

    matplotlib: every frame is DRAWN (so the contexts describe rendered
    frames); plotly: the contexts are recorded while its frames are built.
    """
    seen = []
    out = hyp.plot(data, backend=backend, show=False, on_frame=seen.append,
                   **kwargs)
    if backend == 'matplotlib':
        for f in range(out.n_frames):
            out.draw_frame(f)
    # a SNAPSHOT: a later matplotlib draw_frame() calls the recorder again
    return out, list(seen)


def _close(out):
    fig = getattr(out, 'figure', None)
    if fig is not None:
        plt.close(fig)


def _pairwise(points):
    p = np.asarray(points, dtype=float)
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))


def _assert_observations_are_exact_vertices(grid, source):
    """`grid` holds every row of `source` as an exact vertex, at a uniform
    stride, up to the similarity transform the plot pipeline applies
    (PCA rotation + one isotropic rescale): pairwise distances between the
    grid rows at the observation positions are the source's distances
    times ONE constant."""
    grid = np.asarray(grid, dtype=float)
    n = source.shape[0]
    g = grid.shape[0]
    assert g >= n, (
        f'the animated line has {g} rows for {n} observations: it was '
        'DOWNsampled')
    assert (g - 1) % (n - 1) == 0, (
        f'a {g}-row grid is not a uniform refinement of {n} observations, '
        'so the observations cannot all be grid vertices')
    stride = (g - 1) // (n - 1)
    at_obs = grid[::stride]
    d_grid = _pairwise(at_obs)[np.triu_indices(n, 1)]
    d_src = _pairwise(source)[np.triu_indices(n, 1)]
    ratio = d_grid / d_src
    np.testing.assert_allclose(ratio, np.median(ratio), rtol=1e-6)


def _drawn_radius_spread(xyz):
    """min/max of the drawn distance from the helix axis (1.0 = perfectly
    round). The axis is the midpoint of the drawn x/y extent."""
    x, y = np.asarray(xyz[0], float), np.asarray(xyz[1], float)
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    r = np.hypot(x - cx, y - cy)
    return r.min() / r.max()


# ---------------------------------------------------------------------------
# L8: no downsampling onto the frame grid
# ---------------------------------------------------------------------------

class TestAnimatedLinesKeepEveryObservation:

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_a_long_line_in_a_short_animation_keeps_every_observation(
            self, backend):
        """36 observations, 9 frames: the animated line used to have 9
        rows. Every observation must be an exact vertex now."""
        src = helix(36)
        out, seen = _collect(src, backend, animate=True, duration=1.5,
                             frame_rate=6)
        try:
            assert len(seen) == 9          # the frame count is unchanged
            _assert_observations_are_exact_vertices(seen[-1].datasets[0], src)
        finally:
            _close(out)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_a_short_line_in_a_long_animation_keeps_every_observation(
            self, backend):
        """5 observations, 40 frames: the grid is a refinement that still
        contains each observation exactly (it used to be 40 rows, which
        holds none of the three interior observations)."""
        src = helix(5, turns=0.8)
        out, seen = _collect(src, backend, animate=True, duration=4,
                             frame_rate=10)
        try:
            assert len(seen) == 40
            grid = seen[-1].datasets[0]
            assert grid.shape[0] >= 40
            _assert_observations_are_exact_vertices(grid, src)
        finally:
            _close(out)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_the_reveal_still_starts_at_the_first_row_and_ends_complete(
            self, backend):
        src = helix(36)
        out, seen = _collect(src, backend, animate=True, duration=1.5,
                             frame_rate=6)
        try:
            counts = [ctx.revealed_counts[0] for ctx in seen]
            n_grid = seen[-1].datasets[0].shape[0]
            assert counts == sorted(counts)       # never runs backwards
            assert counts[0] == 1                 # frame 0: the first row
            assert counts[-1] == n_grid           # the final frame is whole
        finally:
            _close(out)

    @pytest.mark.parametrize('backend', BACKENDS)
    @pytest.mark.parametrize('n_rows,n_frames', [(36, 9), (5, 40), (40, 12)])
    def test_reveal_timing_is_the_old_one_row_per_frame_timing(
            self, backend, n_rows, n_frames):
        """The reveal is still paced from the first observation (frame 0)
        to the last (final frame), linearly: frame k shows the observations
        up to k * (n - 1) / (n_frames - 1) -- the timing lines had when
        their grid was exactly one row per frame."""
        src = helix(n_rows, turns=1.0)
        out, seen = _collect(src, backend, animate=True,
                             duration=n_frames / 4, frame_rate=4)
        try:
            assert len(seen) == n_frames
            g = seen[-1].datasets[0].shape[0]
            stride = (g - 1) // (n_rows - 1)
            for ctx in seen:
                head_row = ctx.revealed_counts[0] - 1
                param = head_row / stride
                exact = ctx.frame * (n_rows - 1) / (n_frames - 1)
                # on the grid, never ahead of the exact timing and less
                # than one grid step behind it
                assert exact - 1.0 / stride < param <= exact + 1e-9
        finally:
            _close(out)

    def test_the_drawn_helix_stays_round_on_matplotlib(self):
        """The shape the probe measured (min/max drawn radius 0.228/0.516
        in a 9-frame animation, i.e. 0.44): read off the drawn artist on
        the LAST frame, where the whole line is revealed."""
        src = helix(36)
        out, seen = _collect(src, 'matplotlib', animate=True, duration=1.5,
                             frame_rate=6)
        try:
            line = seen[-1].artists[0]
            spread = _drawn_radius_spread(line.get_data_3d())
            static = hyp.plot(src, backend='matplotlib', show=False)
            ref = [ln for ln in static.axes[0].lines
                   if len(ln.get_data_3d()[0]) > 10][0]
            assert spread > 0.9
            assert spread == pytest.approx(
                _drawn_radius_spread(ref.get_data_3d()), abs=0.02)
            plt.close(static)
        finally:
            _close(out)

    def test_the_drawn_helix_stays_round_on_plotly(self):
        src = helix(36)
        out, _ = _collect(src, 'plotly', animate=True, duration=1.5,
                          frame_rate=6)
        last = out.frames[-1].data[0]
        assert _drawn_radius_spread((last.x, last.y)) > 0.9

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_spin_draws_the_observations_themselves(self, backend):
        """'spin' reveals no rows (only the camera moves), so its datasets
        are never regridded: 36 rows stay 36 rows in a 60-frame spin (they
        used to become 60 PCHIP samples, and 9 in a 9-frame one)."""
        src = helix(36)
        for duration in (1.5, 10):
            out, seen = _collect(src, backend, animate='spin',
                                 duration=duration, frame_rate=6)
            try:
                grid = seen[0].datasets[0]
                assert grid.shape[0] == 36
                _assert_observations_are_exact_vertices(grid, src)
            finally:
                _close(out)

    def test_spin_drawn_shape_matches_the_static_plot(self):
        """The probe's own numbers: a 9-frame spin drew the helix at
        min/max radius 0.44 while the static plot is round."""
        src = helix(36)
        anim = hyp.plot(src, backend='matplotlib', animate='spin',
                        duration=1.5, frame_rate=6, show=False)
        try:
            anim.draw_frame(0)
            line = [ln for ln in anim.figure.axes[0].lines
                    if len(ln.get_data_3d()[0]) > 10
                    and ln.get_color() not in ('black', 'k')][0]
            assert _drawn_radius_spread(line.get_data_3d()) > 0.9
        finally:
            plt.close(anim.figure)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_per_point_labels_are_not_dropped(self, backend):
        """18 labels on a 36-row line in a 9-frame animation: squeezed
        onto 9 grid rows, pairs of labels landed on one row and only 9 of
        the 18 were drawn."""
        labels = [f'L{i}' if i % 2 == 0 else None for i in range(36)]
        out = hyp.plot(helix(36), backend=backend, animate=True,
                       duration=1.5, frame_rate=6, labels=labels,
                       show=False)
        if backend == 'matplotlib':
            try:
                out.draw_frame(out.n_frames - 1)
                drawn = [t.get_text() for t in out.figure.axes[0].texts]
            finally:
                plt.close(out.figure)
        else:
            drawn = [a.text for a in out.layout.scene.annotations]
        assert sorted(drawn) == sorted(lab for lab in labels if lab)

    def test_a_label_is_shown_exactly_while_its_point_is_drawn(self):
        """matplotlib animated labels: visible iff the labelled row is inside
        the head window the trace was drawn over THIS frame. The old rule
        compared the row index with the FRAME index, which only coincided
        while every line had exactly one row per frame."""
        labels = [f'L{i}' if i % 5 == 0 else None for i in range(36)]
        state = []

        def record(ctx):
            s, e = ctx.window_bounds[0]
            shown = {t.get_text() for t in ctx.axes.texts if t.get_visible()}
            state.append((s, e, shown))

        anim = hyp.plot(helix(36), backend='matplotlib', animate='window',
                        focused=0.5, duration=3, frame_rate=6,
                        labels=labels, on_frame=record, show=False)
        try:
            for f in range(anim.n_frames):
                anim.draw_frame(f)
        finally:
            plt.close(anim.figure)
        ever = set()
        for s, e, shown in state[:anim.n_frames]:
            expect = {f'L{i}' for i in range(36) if i % 5 == 0 and s <= i < e}
            assert shown == expect, (s, e, shown)
            ever |= shown
        assert ever == {lab for lab in labels if lab}

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_two_datasets_of_different_lengths_each_keep_their_rows(
            self, backend):
        a, b = helix(36), helix(7, turns=1.0) + [2.0, 0.0, 0.0]
        out, seen = _collect([a, b], backend, animate=True, duration=2,
                             frame_rate=6)
        try:
            grids = seen[-1].datasets
            assert len(seen) == 12
            for grid, src in zip(grids, (a, b)):
                assert grid.shape[0] >= max(12, src.shape[0])
                n, g = src.shape[0], grid.shape[0]
                assert (g - 1) % (n - 1) == 0
        finally:
            _close(out)


# ---------------------------------------------------------------------------
# L9: every morph transition frame moves; dots are visible
# ---------------------------------------------------------------------------

def _two_clouds():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(30, 3)) * 0.3
    b = rng.normal(size=(30, 3)) * 0.3 + [3.0, 0.0, 0.0]
    return [a, b]


def _morph_frames(backend, **kwargs):
    """(positions, rgb, ctx) per frame, read off what was drawn."""
    out, seen = _collect(_two_clouds(), backend, animate='morph',
                         **kwargs)
    frames = []
    if backend == 'matplotlib':
        # draw each frame again and read the SINGLE cloud artist right away
        # (the artist is shared across frames)
        for ctx in seen:
            out.draw_frame(ctx.frame)
            art = ctx.artists[0]
            frames.append((np.column_stack(art.get_data_3d()),
                           to_rgb(art.get_color()), ctx))
        plt.close(out.figure)
    else:
        for ctx, frame in zip(seen, out.frames):
            tr = frame.data[0]
            rgb = tuple(float(v) / 255.0 for v in
                        tr.marker.color[tr.marker.color.index('(') + 1:
                                        tr.marker.color.index(')')]
                        .split(',')[:3])
            frames.append((np.column_stack([tr.x, tr.y, tr.z]), rgb, ctx))
    return frames


class TestMorphTransitionsMove:

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_every_transition_frame_lies_strictly_between_the_clouds(
            self, backend):
        """The probe's call: 9 frames, loop=True -> 2-frame transitions.
        Both frames of each transition used to be exact copies of the hold
        clouds."""
        frames = _morph_frames(backend, duration=1.5, frame_rate=6,
                               loop=True, morph_samples=30)
        transitions = [f for f in frames
                       if f[2].segment_kind == 'transition']
        assert transitions, 'the schedule has no transition frames'
        for pts, rgb, ctx in transitions:
            k = ctx.segment_index // 2
            before, after = ctx.datasets[k], ctx.datasets[k + 1]
            assert pts.shape == before.shape
            assert not np.allclose(pts, before), (
                f'transition frame {ctx.frame} is a copy of the cloud it '
                'leaves')
            assert not np.allclose(pts, after), (
                f'transition frame {ctx.frame} is a copy of the cloud it '
                'reaches')

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_consecutive_transition_frames_differ(self, backend):
        frames = _morph_frames(backend, duration=3, frame_rate=6)
        trans = [f for f in frames if f[2].segment_kind == 'transition']
        assert len(trans) >= 2
        for (p0, _, _), (p1, _, _) in zip(trans, trans[1:]):
            assert not np.allclose(p0, p1)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_transition_colour_is_strictly_between_the_dataset_colours(
            self, backend):
        frames = _morph_frames(backend, duration=1.5, frame_rate=6)
        holds = [f for f in frames if f[2].segment_kind == 'hold']
        c_first = np.array(holds[0][1])
        c_last = np.array(holds[-1][1])
        assert not np.allclose(c_first, c_last)
        for _, rgb, ctx in frames:
            if ctx.segment_kind != 'transition':
                continue
            assert not np.allclose(rgb, c_first, atol=2e-3)
            assert not np.allclose(rgb, c_last, atol=2e-3)

    def test_the_interpolation_parameter_never_reaches_an_endpoint(self):
        clouds = [np.zeros((4, 3)), np.ones((4, 3))]
        for n_steps in (1, 2, 3, 10):
            ts = [morph.morph_positions(clouds, 1, s, n_steps)[0, 0]
                  for s in range(n_steps)]
            assert all(0.0 < t < 1.0 for t in ts), (n_steps, ts)
            assert ts == sorted(ts)
            # symmetric about the midpoint: the easing starts and ends alike
            np.testing.assert_allclose(ts, [1 - t for t in reversed(ts)])


class TestMorphDotsAreVisible:

    def test_both_backends_share_one_4pt_default(self):
        from hypertools.plot.plotly_backend import (
            MORPH_DEFAULT_MARKERSIZE_PT, _marker_size_px)
        assert morph.MORPH_DEFAULT_MARKERSIZE_PT == pytest.approx(4.0)
        assert MORPH_DEFAULT_MARKERSIZE_PT == morph.MORPH_DEFAULT_MARKERSIZE_PT
        anim = hyp.plot(_two_clouds(), backend='matplotlib',
                        animate='morph', duration=1, frame_rate=6,
                        show=False)
        try:
            cloud = [ln for ln in anim.figure.axes[0].lines
                     if ln.get_marker() == '.'][0]
            assert cloud.get_markersize() == pytest.approx(4.0)
        finally:
            plt.close(anim.figure)
        fig = hyp.plot(_two_clouds(), backend='plotly', animate='morph',
                       duration=1, frame_rate=6, show=False)
        sizes = [tr.marker.size for tr in fig.data
                 if getattr(tr, 'mode', None) == 'markers'
                 and tr.marker.size is not None]
        assert sizes and sizes[0] == pytest.approx(
            _marker_size_px(4.0, '.', ndims=3))

    @staticmethod
    def _mpl_dot_ink(**kwargs):
        """Pixels (per dot) rendered in the cloud's own colour, frame 0."""
        rng = np.random.default_rng(0)
        a = rng.uniform(-1, 1, size=(30, 3))
        anim = hyp.plot([a, a[::-1] * 0.5], backend='matplotlib',
                        animate='morph', duration=1, frame_rate=6,
                        show=False, **kwargs)
        try:
            anim.draw_frame(0)
            fig = anim.figure
            fig.set_dpi(100)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
            colour = np.array(to_rgb(
                [ln for ln in fig.axes[0].lines
                 if ln.get_marker() == '.'][0].get_color())) * 255
            return (np.abs(img - colour).sum(axis=-1) < 60).sum() / 30
        finally:
            plt.close(anim.figure)

    def test_matplotlib_morph_dots_render_several_pixels_each(self):
        """Pixel census at 100 dpi: the old 1.5 pt default put about ONE
        pixel per dot in the cloud's colour (measured 1.0); the default must
        be several times that, and the explicit 1.5 pt request still works."""
        default = self._mpl_dot_ink()
        old = self._mpl_dot_ink(markersize=1.5)
        assert default >= 4, f'{default:.1f} px per dot'
        assert default >= 4 * old, (default, old)

    def test_plotly_morph_dots_are_not_sub_pixel(self, tmp_path):
        """A real kaleido render: at the old 1.5 pt default the 30 dots
        covered half a pixel each in the cloud's colour (sub-pixel)."""
        from PIL import Image
        rng = np.random.default_rng(0)
        a = rng.uniform(-1, 1, size=(30, 3))

        def ink(**kwargs):
            fig = hyp.plot([a, a[::-1] * 0.5], backend='plotly',
                           animate='morph', duration=1, frame_rate=6,
                           show=False, **kwargs)
            out = tmp_path / f"dots{kwargs.get('markersize', '')}.png"
            fig.write_image(str(out))
            img = np.asarray(Image.open(out).convert('RGB')).astype(int)
            # the first dataset's palette colour is a red (hls palette)
            red = ((img[..., 0] > 150) & (img[..., 0] - img[..., 1] > 50)
                   & (img[..., 0] - img[..., 2] > 50))
            return red.sum() / 30

        default, old = ink(), ink(markersize=1.5)
        assert default >= 3, f'{default:.1f} px per dot'
        assert default >= 4 * old, (default, old)


# ---------------------------------------------------------------------------
# L11: an on_frame title on a 3-D animation is on the canvas
# ---------------------------------------------------------------------------

class TestOnFrameTitleIsVisible:

    @pytest.mark.parametrize('kwargs', [
        dict(),                                               # docstring
        dict(animate='serial', duration=1.5, frame_rate=6),   # tour call
    ])
    def test_a_callback_title_on_a_3d_animation_is_inside_the_canvas(
            self, kwargs):
        data = [np.cumsum(np.random.default_rng(0).standard_normal(
            (20, 3)), axis=0)]
        kwargs = dict(kwargs)
        animate = kwargs.pop('animate', True)

        def annotate(ctx):
            ctx.axes.set_title(f'frame {ctx.frame} of {ctx.n_frames}')

        anim = hyp.plot(data, animate=animate, on_frame=annotate,
                        show=False, backend='matplotlib', **kwargs)
        try:
            fig = anim.figure
            anim.draw_frame(anim.n_frames // 2)
            fig.canvas.draw()
            box = fig.axes[0].title.get_window_extent(
                fig.canvas.get_renderer())
            _, height = fig.canvas.get_width_height()
            assert fig.axes[0].title.get_text().startswith('frame ')
            assert box.y0 >= 0 and box.y1 <= height, (
                f'title spans y={box.y0:.0f}..{box.y1:.0f} on a '
                f'{height}px canvas')
        finally:
            plt.close(anim.figure)

    def test_a_callback_title_on_a_plotly_3d_animation_is_not_clipped(
            self, tmp_path):
        """plotly's `ctx.figure` is the go.Figure: a callback title set on
        its layout got plotly's 10 px no-title margin and rendered cut off
        at the top edge (ink from pixel row 0). Rendered for real."""
        from PIL import Image
        data = [np.cumsum(np.random.default_rng(0).standard_normal(
            (20, 3)), axis=0)]

        def annotate(ctx):
            ctx.figure.update_layout(
                title=dict(text=f'frame {ctx.frame} of {ctx.n_frames}'))

        fig = hyp.plot(data, backend='plotly', animate=True, duration=1,
                       frame_rate=4, on_frame=annotate, show=False)
        assert fig.layout.title.text == 'frame 3 of 4'
        out = tmp_path / 'title.png'
        fig.write_image(str(out))
        grey = np.asarray(Image.open(out).convert('L'))
        ink_rows = np.where((grey < 100).any(axis=1))[0]
        assert ink_rows.size and ink_rows[0] > 0, (
            'the title is cut off at the top edge of the canvas')

    def test_a_titleless_3d_animation_without_a_callback_keeps_its_canvas(
            self):
        data = np.cumsum(np.random.default_rng(0).standard_normal((20, 3)),
                         axis=0)
        plain = hyp.plot(data, animate=True, duration=1, frame_rate=4,
                         show=False, backend='matplotlib')
        titled = hyp.plot(data, animate=True, duration=1, frame_rate=4,
                          show=False, backend='matplotlib', title='t')
        try:
            assert (plain.figure.get_size_inches()[1]
                    < titled.figure.get_size_inches()[1])
        finally:
            plt.close(plain.figure)
            plt.close(titled.figure)


# ---------------------------------------------------------------------------
# L12: companion panels take the trajectory's colour
# ---------------------------------------------------------------------------

class TestCompanionColour:

    def _companion(self, **kwargs):
        rng = np.random.default_rng(0)
        traj = np.cumsum(rng.normal(size=(24, 3)), axis=0)
        seen = []
        anim = hyp.plot(traj, animate=True, duration=2, frame_rate=6,
                        fmt=kwargs.pop('fmt', '-'), show=False,
                        backend='matplotlib', on_frame=seen.append,
                        companion=[{'data': traj[:, 0], 'smooth': 3},
                                   {'data': traj[:, 1], 'reveal': False,
                                    'position': 'right'}],
                        **kwargs)
        anim.draw_frame(anim.n_frames - 1)
        anim.figure.canvas.draw()
        return anim, seen

    def _panel_colours(self, anim):
        out = []
        for pax in anim.figure.axes[1:]:
            cols = {to_hex(c) for coll in pax.collections
                    for c in coll.get_colors()}
            out.append((cols, to_hex(pax.lines[-1].get_markerfacecolor())))
        return out

    def test_default_colour_is_the_accompanied_trajectory_s(self):
        anim, seen = self._companion()
        try:
            main = to_hex(seen[-1].artists[0].get_color())
            assert main != to_hex('C0')
            for cols, head in self._panel_colours(anim):
                assert cols == {main}
                assert head == main
        finally:
            plt.close(anim.figure)

    def test_a_palette_choice_reaches_the_panel(self):
        anim, seen = self._companion(palette='viridis')
        try:
            main = to_hex(seen[-1].artists[0].get_color())
            for cols, head in self._panel_colours(anim):
                assert cols == {main} and head == main
        finally:
            plt.close(anim.figure)

    def test_an_fmt_colour_letter_reaches_the_panel(self):
        anim, _ = self._companion(fmt='g-')
        try:
            for cols, head in self._panel_colours(anim):
                assert cols == {to_hex('g')} and head == to_hex('g')
        finally:
            plt.close(anim.figure)

    def test_an_explicit_panel_colour_still_wins(self):
        rng = np.random.default_rng(0)
        traj = np.cumsum(rng.normal(size=(24, 3)), axis=0)
        anim = hyp.plot(traj, animate=True, duration=2, frame_rate=6,
                        show=False, backend='matplotlib',
                        companion={'data': traj[:, 0], 'color': 'purple'})
        try:
            anim.draw_frame(anim.n_frames - 1)
            pax = anim.figure.axes[-1]
            assert {to_hex(c) for c in pax.collections[0].get_colors()} == {
                to_hex('purple')}
        finally:
            plt.close(anim.figure)


# ---------------------------------------------------------------------------
# L13b: the clamp warning fires on short streams
# ---------------------------------------------------------------------------

def _stream(rows):
    for row in rows:
        yield row


class TestShortStreamClampWarning:

    def _drifting(self, n_post):
        rng = np.random.default_rng(5)
        head = rng.normal(size=(8, 3)) * 0.1
        tail = rng.normal(size=(n_post, 3)) * 0.1 + 10.0
        return np.vstack([head, tail])

    @pytest.mark.parametrize('n_post', [4, 8, 16])
    def test_a_short_drifting_stream_warns(self, n_post):
        with pytest.warns(RuntimeWarning, match='outside the display box'):
            fig = hyp.plot(_stream(self._drifting(n_post)), stream_init=8,
                           stream_chunk=4, show=False)
        plt.close(fig)

    def test_a_stream_that_dies_after_a_short_drift_warns(self):
        """STREAM-02's shape: 8 head samples, 8 more, then the source
        raises. 7 of the 8 post-head samples were drawn clamped, silently."""
        rows = self._drifting(8)

        def broken():
            yield from rows
            raise RuntimeError('source disconnected')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig = hyp.plot(broken(), stream_init=8, stream_chunk=4,
                           show=False)
        messages = [str(w.message) for w in caught]
        assert any('outside the display box' in m for m in messages)
        assert any('streaming stopped early' in m for m in messages)
        plt.close(fig)

    def test_the_warning_fires_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig = hyp.plot(_stream(self._drifting(40)), stream_init=8,
                           stream_chunk=4, show=False)
        assert sum('outside the display box' in str(w.message)
                   for w in caught) == 1
        plt.close(fig)

    def test_fewer_than_four_post_head_samples_do_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig = hyp.plot(_stream(self._drifting(3)), stream_init=8,
                           stream_chunk=4, show=False)
        assert not [w for w in caught
                    if 'outside the display box' in str(w.message)]
        plt.close(fig)

    def test_a_short_stream_inside_the_box_does_not_warn(self):
        rng = np.random.default_rng(5)
        head = rng.uniform(-1, 1, size=(8, 3))
        head[:2] = [[-1, -1, -1], [1, 1, 1]]     # the head spans the box
        tail = rng.uniform(-0.5, 0.5, size=(8, 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig = hyp.plot(_stream(np.vstack([head, tail])), stream_init=8,
                           stream_chunk=4, reduce=None, show=False)
        assert not [w for w in caught
                    if 'outside the display box' in str(w.message)]
        plt.close(fig)
