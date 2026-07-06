"""Empirical marker-size-parity measurement helper (R2 fix, maintainer
request: "dots should be smaller in plotly (to match matplotlib)").

Renders a single isolated marker with matplotlib and plotly at the SAME
canvas pixel size, then measures the rendered dot's actual pixel diameter
via a simple non-white-pixel bounding box (no mocks -- real renders,
real rasterized PNGs). Used by:
  - `tests/test_marker_parity.py` (empirical parity regression: plotly's
    converted pixel size must match matplotlib's rendered diameter within
    ~20%, for both 'o' and '.' marker characters, the two calibration
    points documented in `hypertools.plot.plotly_backend`'s `PT_TO_PX`/
    `_DOT_MARKER_SCALE` constants).
  - `scripts/generate_marker_parity_evidence.py` (the committed
    `dot_size_parity.png` side-by-side evidence image).

Calibration notes (see `hypertools/plot/plotly_backend.py`'s module-level
constants for the full derivation):
  - matplotlib renders at dpi=100 by default (never overridden by this
    codebase), so 1 point = 100/72 px. This holds for BOTH 2-D `ax.plot`
    and 3-D `Axes3D.plot` marker rendering (verified empirically -- both
    go through the same `Line2D` marker-drawing code).
  - matplotlib's '.'/',' marker glyphs are defined with HALF the path
    scale of 'o' and most other marker characters (verified via
    `matplotlib.markers.MarkerStyle(ch).get_transform()`), so at the SAME
    `markersize`, '.' renders at half the pixel diameter of 'o'.
  - plotly's `go.Scatter` (2-D, SVG) `marker.size` is (empirically, per
    this module) essentially the literal rendered pixel diameter -- no
    adjustment needed there.
  - plotly's `go.Scatter3d` (3-D, WebGL) `marker.size` is NOT the literal
    rendered pixel diameter -- empirically, `diameter_px ~= 1.776 *
    size_px` (verified independent of camera distance, so this is not a
    perspective effect), i.e. Scatter3d needs an EXTRA ~1.776x correction
    on top of the 2-D conversion above.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CANVAS_SIZE = (640, 480)  # matches hypertools' DEFAULT_FIGSIZE (6.4, 4.8) @ 100dpi
DPI = 100


def render_mpl_marker(markersize, marker, path, color='red'):
    """Render one isolated mpl marker, centered, at `markersize` (points),
    on a `CANVAS_SIZE`-pixel white canvas at `DPI`, and save to `path`."""
    w_in, h_in = CANVAS_SIZE[0] / DPI, CANVAS_SIZE[1] / DPI
    fig = plt.figure(figsize=(w_in, h_in), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # markeredgewidth=0: plotly markers have no stroke by default either
    # (`render_plotly_marker` never sets `marker.line`), so a fair
    # pixel-diameter comparison must exclude mpl's default ~1pt edge
    # stroke, which would otherwise inflate the measured diameter by a
    # fixed pixel amount disproportionately large at small markersizes.
    ax.plot([0.5], [0.5], marker=marker, linestyle='None',
            markersize=markersize, color=color, markeredgecolor=color,
            markeredgewidth=0)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def render_mpl_marker_3d(markersize, marker, path, color='red'):
    """As `render_mpl_marker`, but through `Axes3D.plot` (a single point at
    the origin, camera-angle-independent since mpl markers are NOT
    perspective-scaled) -- confirms 3-D marker rendering matches the 2-D
    calibration exactly (it does: both use `Line2D`'s marker path)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    w_in, h_in = CANVAS_SIZE[0] / DPI, CANVAS_SIZE[1] / DPI
    fig = plt.figure(figsize=(w_in, h_in), dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.axis('off')
    ax.plot([0], [0], [0], marker=marker, linestyle='None',
            markersize=markersize, color=color, markeredgecolor=color,
            markeredgewidth=0)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def render_plotly_marker(size_px, path, color='red'):
    """Render one isolated plotly marker (`marker.size=size_px`), centered,
    on the same `CANVAS_SIZE`-pixel white canvas, and save to `path`."""
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter(
        x=[0.5], y=[0.5], mode='markers',
        marker=dict(color=color, size=size_px, symbol='circle'))])
    fig.update_layout(
        width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white', plot_bgcolor='white')
    fig.write_image(path, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])


def render_plotly_marker_3d(size_px, path, color='red'):
    """Render one isolated plotly `go.Scatter3d` marker (`marker.size=
    size_px`), centered at the scene origin, on the same `CANVAS_SIZE`
    canvas -- `go.Scatter3d` needs `_SCATTER3D_SIZE_FACTOR` correction on
    top of the 2-D conversion (see this module's docstring)."""
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(color=color, size=size_px, symbol='circle'))])
    fig.update_layout(
        width=CANVAS_SIZE[0], height=CANVAS_SIZE[1],
        scene=dict(xaxis=dict(visible=False, range=[-1, 1]),
                  yaxis=dict(visible=False, range=[-1, 1]),
                  zaxis=dict(visible=False, range=[-1, 1]),
                  camera=dict(eye=dict(x=0, y=0, z=2.5))),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white')
    fig.write_image(path, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])


def measure_diameter(path, threshold=250):
    """Return (width_px, height_px) of the non-white bounding box in the
    PNG at `path` -- the rendered marker's pixel diameter along each axis
    (equal for a circular marker, modulo rasterization noise)."""
    img = np.asarray(Image.open(path).convert('RGB'))
    mask = np.any(img < threshold, axis=-1)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
