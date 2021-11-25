import gif
import plotly.graph_objects as go


@gif.frame
def frame2fig(fig, i):
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


def save_gif(fig, fname, framerate=30):
    frames = []
    for i in range(len(fig.frames)):
        frames.append(frame2fig(fig, i))

    gif.save(frames, fname, duration=1000 / framerate)


def hypersave(fig, fname, **kwargs):
    if hasattr(fig, 'frames') and len(fig.frames) > 0:
        save_gif(fig, fname, **kwargs)
    else:
        fig.write_image(fname, **kwargs)
