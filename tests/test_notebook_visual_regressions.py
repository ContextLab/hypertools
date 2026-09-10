"""Real artist and pixel regressions from the v1.1 notebook visual review."""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hypertools as hyp


@pytest.mark.parametrize("frames", [9, 30])
def test_full_companion_curve_keeps_a_moving_colored_head(frames):
    data = hyp.load("helix", n_samples=30)
    dates = pd.date_range("2026-01-01", periods=30)
    seen = []
    anim = hyp.plot(
        pd.DataFrame(data, index=dates),
        animate=True,
        duration=frames / 6,
        frame_rate=6,
        show=False,
        on_frame=seen.append,
        title="{index:%Y-%m-%d}",
        companion=[
            {"data": data[:, 0]},
            {
                "data": data[:, 1],
                "reveal": False,
                "hue": np.arange(30),
                "smooth": 3,
                "position": "right",
            },
        ],
    )
    try:
        # Seek out of order as well as forward: no stale artist state.
        for f in [0, frames // 2, frames - 1, 1, 0]:
            anim.draw_frame(f)
            row = round(f / (frames - 1) * 29)
            bottom, right = anim.figure.axes[1:]
            assert list(bottom.lines[-1].get_xdata()) == [row]
            assert list(right.lines[-1].get_xdata()) == [row]
            assert list(right.lines[-1].get_ydata()) == [data[row, 1]]
            assert len(right.collections[0].get_segments()) == 29
            assert len(right.lines[1].get_xdata()) == 30
            assert anim.figure.axes[0].get_title() == dates[row].strftime("%Y-%m-%d")
            collection = right.collections[0]
            np.testing.assert_allclose(
                right.lines[-1].get_markerfacecolor(),
                collection.cmap(collection.norm(row)),
            )
            ctx = seen[-1]
            main_head = np.array([v[-1] for v in ctx.artists[0].get_data_3d()])
            np.testing.assert_allclose(
                main_head, ctx.datasets[0][ctx.revealed_counts[0] - 1]
            )
    finally:
        plt.close(anim.figure)


@pytest.mark.parametrize("fmt", ["-", "o", "o-"])
@pytest.mark.parametrize("animation", [False, "parallel", "serial", "spin"])
def test_plotly_uniform_alpha_uses_native_opacity(fmt, animation):
    pytest.importorskip("plotly")
    data = hyp.load("helix", n_samples=12)
    fig = hyp.plot(
        data,
        backend="plotly",
        colors=["steelblue"],
        alpha=0.5,
        fmt=fmt,
        animate=animation,
        duration=1,
        frame_rate=4,
        show=False,
    )
    trace = next(
        t for t in fig.data if isinstance(t.meta, dict) and "hyp_trace_index" in t.meta
    )
    assert trace.opacity == 0.5
    for token, component in [("lines", trace.line), ("markers", trace.marker)]:
        if token in trace.mode:
            assert component.color == "rgb(70,130,180)"


def test_rendered_transparent_steelblue_preserves_hue(tmp_path):
    """JSON alone passed the original bug. Decode actual Chrome-rendered pixels."""
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    from PIL import Image

    data = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20), np.zeros(20)])
    fig = hyp.plot(
        data,
        backend="plotly",
        reduce=None,
        colors=["steelblue"],
        alpha=0.5,
        linewidth=12,
        show=False,
    )
    path = tmp_path / "steelblue.png"
    fig.write_image(str(path), width=500, height=400)
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(float)
    pixels = rgb[(rgb[:, :, 2] > rgb[:, :, 0] + 15) & (rgb[:, :, 2] > rgb[:, :, 1] + 5)]
    assert len(pixels) > 100
    # Alpha blending against white preserves this ratio of channel gaps.
    # The faulty RGBA path makes G approach B (cyan), driving it towards zero.
    ratio = (pixels[:, 2] - pixels[:, 1]) / (pixels[:, 2] - pixels[:, 0])
    assert np.median(ratio) == pytest.approx((180 - 130) / (180 - 70), abs=0.12)


def test_opaque_morph_frame_resets_the_base_trace_opacity():
    go = pytest.importorskip("plotly.graph_objects")
    from tests._plotly_colors import rgba

    rng = np.random.default_rng(71)
    fig = hyp.plot(
        [rng.normal(size=(12, 3)), rng.normal(size=(12, 3))],
        fmt="o",
        backend="plotly",
        animate="morph",
        alpha=[0.2, 1.0],
        duration=2,
        frame_rate=6,
        show=False,
    )
    index = fig.frames[-1].traces[0]
    assert rgba(fig.data[index], "marker")[-1] == pytest.approx(0.2)
    snapshot = go.Figure(fig)
    for trace_index, update in zip(fig.frames[-1].traces, fig.frames[-1].data):
        snapshot.data[trace_index].update(update.to_plotly_json())
    assert rgba(snapshot.data[index], "marker")[-1] == pytest.approx(1.0)


def test_tour_plotly_previews_do_not_allocate_live_plots_or_embed_html(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    pytest.importorskip("ipywidgets")
    from IPython.utils.capture import capture_output
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    ns = dict(SCRATCH=tmp_path, IN_COLAB=False)
    exec(
        compile(
            (root / "scripts/feature_tour_support.py").read_text(),
            str(root / "scripts/feature_tour_support.py"),
            "exec",
        ),
        ns,
    )
    fig = hyp.plot(hyp.load("helix", n_samples=12), backend="plotly", show=False)
    from IPython.core.interactiveshell import InteractiveShell

    had_shell = InteractiveShell.initialized()
    InteractiveShell.instance()
    try:
        with capture_output(display=True) as captured:
            ns["show_result"](fig)
    finally:
        if not had_shell:
            InteractiveShell.clear_instance()
    data = [o.data for o in captured.outputs]
    assert any("image/png" in item for item in data)
    assert all("application/vnd.plotly.v1+json" not in item for item in data)
    assert len(json.dumps(data)) < 1_000_000
    assert len(ns["INTERACTIVE_PLOTS"]) == 1
    path = Path(next(iter(ns["INTERACTIVE_PLOTS"].values())))
    assert (
        path.stat().st_size > 1_000_000
    )  # standalone JS lives on disk, not in every cell
    assert "Plotly.newPlot(" in path.read_text()
