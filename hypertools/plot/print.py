import numpy as np
import plotly.graph_objects as go
import io
from PIL import Image
import sys

def frame2fig(fig, i):
    i = int(np.round(i))
    if i >= len(fig.frames):
        i = len(fig.frames) - 1
    elif i < 0:
        i = 0

    template = list(fig.data).copy()
    frame_data = list(fig.frames[i].data).copy()

    for t in range(len(template)):
        for coord in ['x', 'y', 'z']:
            if hasattr(template[t], coord):
                setattr(template[t], coord, getattr(frame_data[t], coord))
    
    # if self.proj == '3d':
    #         lengths = np.abs(np.diff(get_bounds(self.data), axis=0)).ravel()
    #         fig.update_layout(scene_aspectmode='manual',
    #                           scene_aspectratio={'x': 1, 'y': lengths[1] / lengths[0], 'z': lengths[2] / lengths[0]},
    #                           scene={'camera': init.layout.scene.camera})

    x = go.Figure(layout=fig.layout, data=template)  # FIXME: if the figure is 3d, grab the camera properties here...
    x.update_layout(showlegend=False)

    try:
        x.update_layout(scene_aspectmode=fig.layout.scene.aspectmode,
                        scene_aspectratio=fig.layout.scene.aspectratio,
                        scene={'camera': fig.frames[i].layout.scene.camera})
    except:
        pass
    x = x.to_dict()
    x['layout'].pop('sliders')
    x['layout'].pop('updatemenus')
    return go.Figure(x)


def fig2array(fig):
    fig_bytes = fig.to_image(format='png')
    buffer = io.BytesIO(fig_bytes)
    img = Image.open(buffer)
    return np.asarray(img)


def save_gif(fig, fname, framerate=10, duration=20):
    if not any(m in sys.modules for m in ['moviepy', 'mpy']):
        try:
            exec('import moviepy.editor as mpy', globals())
        except ImportError:
            raise RuntimeError('To enable saving to GIFs, please install ffmpeg')

    def get_frame(t):
        frame = int(np.round((t / duration) * len(fig.frames), decimals=0))
        return fig2array(frame2fig(fig, frame))

    animation = mpy.VideoClip(get_frame, duration=duration)
    animation.write_gif(fname, fps=framerate)


def hypersave(fig, fname, **kwargs):
    if hasattr(fig, 'frames') and len(fig.frames) > 0:
        save_gif(fig, fname, **kwargs)
    else:
        fig.write_image(fname, **kwargs)
