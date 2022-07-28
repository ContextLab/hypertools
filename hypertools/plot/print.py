import numpy as np
import plotly.graph_objects as go
import moviepy.editor as mpy
import io
from PIL import Image


def frame2fig(fig, i):
    i = int(np.round(i))
    if i >= len(fig.frames):
        i = len(fig.frames) - 1
    elif i < 0:
        i = 0

    template = list(fig.data).copy()
    frame_data = list(fig.frames[i].data).copy()

    for i in range(len(template)):
        for coord in ['x', 'y', 'z']:
            if hasattr(template[i], coord):
                setattr(template[i], coord, getattr(frame_data[i], coord))

    fig = go.Figure(layout=fig.layout, data=template)
    fig.update_layout(showlegend=False)
    fig = fig.to_dict()
    fig['layout'].pop('sliders')
    fig['layout'].pop('updatemenus')
    return go.Figure(fig)


def fig2array(fig):
    fig_bytes = fig.to_image(format='png')
    buffer = io.BytesIO(fig_bytes)
    img = Image.open(buffer)
    return np.asarray(img)


def save_gif(fig, fname, framerate=30, duration=10):
    def get_frame(t):
        frame = (t / duration) * len(fig.frames)
        return fig2array(frame2fig(fig, frame))

    animation = mpy.VideoClip(get_frame, duration=len(fig.frames))
    animation.write_gif(fname, fps=framerate)


def hypersave(fig, fname, **kwargs):
    if hasattr(fig, 'frames') and len(fig.frames) > 0:
        save_gif(fig, fname, **kwargs)
    else:
        fig.write_image(fname, **kwargs)
