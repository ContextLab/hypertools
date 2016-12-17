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
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from .helpers import *
from .reduce import reduce as reduceD

##MAIN FUNCTION##
def animated_plot(x, *args, **kwargs):

    if 'save_path' in kwargs:
        save=True
        save_path = kwargs['save_path']
        del kwargs['save_path']
    else:
        save=False

    ##SUB FUNCTIONS##
    def get_cube_scale(x, c):
        x = np.vstack(x)
        dists = np.sqrt(np.sum(np.square(x), axis=1))
        return c*np.max(dists)

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
            plane_list.append(ax.plot_wireframe(Xs, Ys, Zs, rstride=1, cstride=1, color='black', linewidth=3))
        return plane_list

    def update_lines(num, data_lines, lines, trail_lines, cube_scale, tail_len=30, tail_style=':', speed=1):

        if hasattr(update_lines, 'planes'):
            for plane in update_lines.planes:
                plane.remove()

        update_lines.planes = plot_cube(cube_scale)
        ax.view_init(elev=10, azim=speed*num/5)
        ax.dist=cube_scale-1

        for line, data, trail in zip(lines, data_lines, trail_lines):
            if num<=tail_len:
                    line.set_data(data[0:num+1, 0:2].T)
                    line.set_3d_properties(data[0:num+1, 2])
            else:
                line.set_data(data[num-tail_len:num+1, 0:2].T)
                line.set_3d_properties(data[num-tail_len:num+1, 2])
        return lines,trail_lines

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    if type(x) is not list:
        x = [x]

    x = reduceD(x,3)
    x = interp_array_list(x)
    x = center(x)

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], linewidth=3, *args, **kwargs)[0] for dat in x]
    trail = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in x]

    ax.set_axis_off()

    # Get cube scale from data
    cube_scale = get_cube_scale(x,1)

    # Setting the axes properties
    ax.set_xlim3d([-cube_scale, cube_scale])
    ax.set_ylim3d([-cube_scale, cube_scale])
    ax.set_zlim3d([-cube_scale, cube_scale])

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, 1000, fargs=(x, lines, trail, cube_scale),
                                   interval=8, blit=False)
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, bitrate=1800)
        line_ani.save(save_path, writer=writer)

    plt.show()
