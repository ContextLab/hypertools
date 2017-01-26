#!/usr/bin/env python

"""
Implements animated trajectory plot

INPUTS:
-numpy array(s)
-list of numpy arrays

OUTPUTS:
-None
"""

##PACKAGES##
from __future__ import division
import numpy as np
import matplotlib
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from .._shared.helpers import *
from ..tools.reduce import reduce as reduceD

##MAIN FUNCTION##
def animated_plot(x, *args, **kwargs):

    assert x[0].shape[1]>2, "Hypertools currently only supports animation for data with > 2 dims."

    ## HYPERTOOLS-SPECIFIC ARG PARSING ##

    if 'zoom' in kwargs:
        zoom=kwargs['zoom']
        del kwargs['zoom']
    else:
        zoom=0

    if 'chem_trails' in kwargs:
        chem_trails= kwargs['chem_trails']
        del kwargs['chem_trails']
    else:
        chem_trails=False

    if 'rotations' in kwargs:
        rotations=kwargs['rotations']
        del kwargs['rotations']
    else:
        rotations=2

    if 'duration' in kwargs:
        duration=kwargs['duration']
        del kwargs['duration']
    else:
        duration=30

    if 'frame_rate' in kwargs:
        frame_rate=kwargs['frame_rate']
        del kwargs['frame_rate']
    else:
        frame_rate=50

    if 'tail_duration' in kwargs:
        tail_duration=kwargs['tail_duration']
        del kwargs['tail_duration']
    else:
        tail_duration=2

    if 'return_data' in kwargs:
        return_data = kwargs['return_data']
        del kwargs['return_data']
    else:
        return_data=False

    if 'legend' in kwargs:
        legend=True
        legend_data = kwargs['legend']
        del kwargs['legend']
    else:
        legend=False

    if 'color_palette' in kwargs:
        palette = kwargs['color_palette']
        del kwargs['color_palette']

    if 'save_path' in kwargs:
        save=True
        save_path = kwargs['save_path']
        del kwargs['save_path']
    else:
        save=False

    # handle show flag
    if 'show' in kwargs:
        show=kwargs['show']
        del kwargs['show']
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    else:
        show=True
        import matplotlib.pyplot as plt

    ##SUB FUNCTIONS##
    def plot_cube(scale, const=1):
        cube = {
            "top"    : ( [[-1,1],[-1,1]], [[-1,-1],[1,1]], [[1,1],[1,1]] ),
            "bottom" : ( [[-1,1],[-1,1]], [[-1,-1],[1,1]], [[-1,-1],[-1,-1]] ),
            "left"   : ( [[-1,-1],[-1,-1]], [[-1,1],[-1,1]], [[-1,-1],[1,1]] ),
            "right"  : ( [[1,1],[1,1]], [[-1,1],[-1,1]], [[-1,-1],[1,1]] ),
            "front"  : ( [[-1,1],[-1,1]], [[-1,-1],[-1,-1]], [[-1,-1],[1,1]] ),
            "back"   : ( [[-1,1],[-1,1]], [[1,1],[1,1]], [[-1,-1],[1,1]] )
            }

        plane_list = []
        for side in cube:
            (Xs, Ys, Zs) = (
                np.asarray(cube[side][0])*scale*const,
                np.asarray(cube[side][1])*scale*const,
                np.asarray(cube[side][2])*scale*const
                )
            plane_list.append(ax.plot_wireframe(Xs, Ys, Zs, rstride=1, cstride=1, color='black', linewidth=2))
        return plane_list

    def update_lines(num, data_lines, lines, trail_lines, cube_scale, tail_duration):

        if hasattr(update_lines, 'planes'):
            for plane in update_lines.planes:
                plane.remove()

        update_lines.planes = plot_cube(cube_scale)
        ax.view_init(elev=10, azim=rotations*(360*(num/data_lines[0].shape[0])))
        ax.dist=8-zoom

        for line, data, trail in zip(lines, data_lines, trail_lines):
            if num<=tail_duration:
                    line.set_data(data[0:num+1, 0:2].T)
                    line.set_3d_properties(data[0:num+1, 2])
            else:
                line.set_data(data[num-tail_duration:num+1, 0:2].T)
                line.set_3d_properties(data[num-tail_duration:num+1, 2])
            if chem_trails:
                trail.set_data(data[0:num + 1, 0:2].T)
                trail.set_3d_properties(data[0:num + 1, 2])
        return lines,trail_lines

    args_list = parse_args(x,args)
    kwargs_list = parse_kwargs(x,kwargs)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    if type(x) is not list:
        x = [x]

    interp_val = frame_rate*duration/(x[0].shape[0] - 1)
    x = interp_array_list(x, interp_val=interp_val)
    x = center(x)
    x = scale(x)

    if tail_duration==0:
        tail_duration=1
    else:
        tail_duration = frame_rate*tail_duration

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], linewidth=3, *args_list[idx], **kwargs_list[idx])[0] for idx,dat in enumerate(x)]
    trail = [
        ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], alpha=.3, linewidth=3, *args_list[idx], **kwargs_list[idx])[0]
        for idx, dat in enumerate(x)]

    ax.set_axis_off()

    # Get cube scale from data
    cube_scale = 1

    # Setting the axes properties
    ax.set_xlim3d([-cube_scale, cube_scale])
    ax.set_ylim3d([-cube_scale, cube_scale])
    ax.set_zlim3d([-cube_scale, cube_scale])

    #add legend
    if legend:
        proxies = [plt.Rectangle((0, 0), 1, 1, fc=palette[idx]) for idx,label in enumerate(legend_data)]
        ax.legend(proxies,legend_data)

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, x[0].shape[0], fargs=(x, lines, trail, cube_scale, tail_duration),
                                   interval=1000/frame_rate, blit=False, repeat=False)
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=frame_rate, bitrate=1800)
        line_ani.save(save_path, writer=writer)

    if show:
        plt.show()

    return fig,ax,x,line_ani
