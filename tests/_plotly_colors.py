"""Effective Plotly colors for parity checks across opacity serializations."""

import re
from matplotlib.colors import to_rgba


def rgba(trace, component="line", index=None):
    style = getattr(trace, component)
    color = style.color if index is None else style.color[index]
    if isinstance(color, str) and color.startswith(("rgb(", "rgba(")):
        parts = [float(v) for v in re.findall(r"[\d.]+", color)]
        assert len(parts) in (3, 4), color
        result = tuple(v / 255 for v in parts[:3]) + (
            parts[3] if len(parts) == 4 else 1.0,
        )
    else:
        result = to_rgba(color)
    opacity = trace.opacity if trace.opacity is not None else 1.0
    if component == "marker" and style.opacity is not None:
        opacity *= style.opacity
    return result[:3] + (result[3] * opacity,)
