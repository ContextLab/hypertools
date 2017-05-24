#!/usr/bin/env python

from __future__ import division
from builtins import str
from builtins import range
import sys
import warnings
import itertools
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import seaborn as sns
from .._shared.helpers import *

matplotlib.rcParams['pdf.fonttype'] = 42

def draw(x, return_data=False, legend=None, save_path=False, labels=False,
         explore=False, show=True, mpl_kwargs=None, format_string=None,
         group=False, animate=False, tail_duration=2, rotations=2, zoom=1,
         chemtrails=False, frame_rate=50, elev=30, azim=-60):

    # handle static plots
    def dispatch_static(x):
        if x[0].ndim==1 or x[0].shape[-1]==1:
            return plot1D(x)
        elif x[0].shape[-1]==2:
            return plot2D(x)
        elif x[0].shape[-1]==3:
            return plot3D(x)

    # plot data in 1D
    def plot1D(data):
        n=len(data)
        fig, ax = plt.subplots()
        for i in range(n):
            ikwargs = kwargs_list[i]
            ifmt = format_string[i]
            ax.plot(data[i][:,0], ifmt, **ikwargs)
        return fig, ax, data

    # plot data in 2D
    def plot2D(data):
        n=len(data)
        fig, ax = plt.subplots()
        for i in range(n):
            ifmt = format_string[i]
            ikwargs = kwargs_list[i]
            ax.plot(data[i][:,0], data[i][:,1], ifmt, **ikwargs)
        return fig, ax, data

    # plot data in 3D
    def plot3D(data):
        n=len(data)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n):
            ifmt = format_string[i]
            ikwargs = kwargs_list[i]
            print(ifmt, ikwargs)
            ax.plot(data[i][:,0], data[i][:,1], data[i][:,2], ifmt, **ikwargs)
        return fig, ax, data

    def annotate_plot(data, labels):
        """Create labels in 3d chart
        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            labels (list) - list of labels of shape (numPoints,1)
        Returns:
            None
        """

        global labels_and_points
        labels_and_points = []

        if data[0].shape[-1]>2:
            proj = ax.get_proj()

        for idx,x in enumerate(data):
            if labels[idx] is not None:
                if data[0].shape[-1]>2:
                    x2, y2, _ = proj3d.proj_transform(x[0], x[1], x[2], proj)
                    label = plt.annotate(
                    labels[idx],
                    xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'),family='serif')
                    labels_and_points.append((label,x[0],x[1],x[2]))
                elif data[0].shape[-1]==2:
                    x2, y2 = x[0], x[1]
                    label = plt.annotate(
                    labels[idx],
                    xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'),family='serif')
                    label.draggable()
                    labels_and_points.append((label,x[0],x[1]))
        fig.canvas.draw()

    def update_position(e):
        """Update label positions in 3d chart
        Args:
            e (mouse event) - event handle to update on
        Returns:
            None
        """

        proj = ax.get_proj()
        for label, x, y, z in labels_and_points:
            x2, y2, _ = proj3d.proj_transform(x, y, z, proj)
            label.xy = x2,y2
            label.update_positions(fig.canvas.renderer)
            label._visible=True
        fig.canvas.draw()

    def hide_labels(e):
        """Hides labels on button press
        Args:
            e (mouse event) - event handle to update on
        Returns:
            None
        """

        for label in labels_and_points:
            label[0]._visible=False

    def add_labels(x, labels):
        """Add labels to graph if available
        Args:
            data (np.ndarray) - Array containing the data points
            labels (list) - List containing labels
        Returns:
            None
        """
        # if explore mode is activated, implement the on hover behavior
        if explore:
            X = np.vstack(x)
            if labels:
                if any(isinstance(el, list) for el in labels):
                    labels = list(itertools.chain(*labels))
                fig.canvas.mpl_connect('motion_notify_event', lambda event: onMouseMotion(event, X, labels)) # on mouse motion
                # fig.canvas.mpl_connect('button_press_event', lambda event: onMouseClick(event, X, labels))  # on mouse click
            else:
                fig.canvas.mpl_connect('motion_notify_event', lambda event: onMouseMotion(event, X)) # on mouse motion
                # fig.canvas.mpl_connect('button_press_event', lambda event: onMouseClick(event, X, labels))  # on mouse click

        elif labels:
            X = np.vstack(x)
            if any(isinstance(el, list) for el in labels):
                labels = list(itertools.chain(*labels))
            annotate_plot(X, labels)
            fig.canvas.mpl_connect('button_press_event', hide_labels)
            fig.canvas.mpl_connect('button_release_event', update_position)

    ##EXPLORE MODE##
    def distance(point, event):
        """Return distance between mouse position and given data point

        Args:
            point (np.array) -  np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent) - mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64) - distance (in screen coords) between mouse pos and data point
        """
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)

    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """

        distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)

    def annotate_plot_explore(X, index, labels=False):
        """Create popover label in 3d chart

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
            labels (list or False) - list of data point labels (default is False)
        Returns:
            None
        """

        # save clicked points
        if not hasattr(annotate_plot_explore, 'clicked'):
            annotate_plot_explore.clicked = []

        # If we have previously displayed another label, remove it first
        if hasattr(annotate_plot_explore, 'label'):
            if index not in annotate_plot_explore.clicked:
                annotate_plot_explore.label.remove()

        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())

        if type(labels) is list:
            label = labels[index]
        else:
            label = "Index " + str(index) + ": (" + "{0:.2f}, ".format(X[index, 0]) + "{0:.2f}, ".format(X[index, 1]) + "{0:.2f}".format(X[index, 2]) + ")"

        annotate_plot_explore.label = plt.annotate(
        label,
        xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        fig.canvas.draw()

    def onMouseMotion(event,X,labels=False):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse
        Args:
            event (event) - event triggered when the mous is moved
            X (np.ndarray) - coordinates by datapoints matrix
            labels (list or False) - list of data labels (default is False)
        Returns:
            None
        """

        closestIndex = calcClosestDatapoint(X, event)

        if hasattr(onMouseMotion, 'first'):
            pass
        else:
            onMouseMotion.first = False
            onMouseMotion.closestIndex_prev = calcClosestDatapoint(X, event)

        if closestIndex!=onMouseMotion.closestIndex_prev:
            if type(labels) is list:
                annotate_plot_explore (X, closestIndex, labels)
                closestIndex_prev = closestIndex
            else:
                annotate_plot_explore (X, closestIndex)
                closestIndex_prev = closestIndex

    def plot_cube(scale):
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
                np.asarray(cube[side][0])*scale,
                np.asarray(cube[side][1])*scale,
                np.asarray(cube[side][2])*scale
                )
            plane_list.append(ax.plot_wireframe(Xs, Ys, Zs, rstride=1, cstride=1, color='black', linewidth=1))
        return plane_list

    def update_lines_parallel(num, data_lines, lines, trail_lines, cube_scale, tail_duration=2,
                     rotations=2, zoom=1, chemtrails=False, elev=10):

        if hasattr(update_lines_parallel, 'planes'):
            for plane in update_lines_parallel.planes:
                plane.remove()

        update_lines_parallel.planes = plot_cube(cube_scale)
        ax.view_init(elev=elev, azim=rotations*(360*(num/data_lines[0].shape[0])))
        # ax.dist=8-zoom

        for line, data, trail in zip(lines, data_lines, trail_lines):
            if num<=tail_duration:
                    line.set_data(data[0:num+1, 0:2].T)
                    line.set_3d_properties(data[0:num+1, 2])
            else:
                line.set_data(data[num-tail_duration:num+1, 0:2].T)
                line.set_3d_properties(data[num-tail_duration:num+1, 2])
            if chemtrails:
                trail.set_data(data[0:num + 1, 0:2].T)
                trail.set_3d_properties(data[0:num + 1, 2])
        return lines, trail_lines

    def update_lines_spin(num, data_lines, lines, cube_scale, rotations=2,
                          zoom=1, elev=10):

        if hasattr(update_lines_spin, 'planes'):
            for plane in update_lines_spin.planes:
                plane.remove()

        update_lines_spin.planes = plot_cube(cube_scale)
        ax.view_init(elev=elev, azim=rotations*(360*(num/data_lines[0].shape[0])))
        # ax.dist=8-zoom

        for line, data in zip(lines, data_lines):
            line.set_data(data[:, 0:2].T)
            line.set_3d_properties(data[:, 2])

        return lines

    def dispatch_animate(x, ani_params):
        if x[0].shape[1] is 3:
            return animate_plot(x, **ani_params)

    def animate_plot(data, tail_duration=2, rotations=2, zoom=1, chemtrails=False,
                       frame_rate=50, elev=10, style='parallel'):

        # inialize plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # create lines
        lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], format_string[idx],
                         linewidth=1, **kwargs_list[idx])[0] for idx,dat in enumerate(x)]
        trail = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], format_string[idx],
                         alpha=.3, linewidth=1, **kwargs_list[idx])[0] for idx, dat in enumerate(x)]
        if tail_duration==0:
            tail_duration=1
        else:
            tail_duration = int(frame_rate*tail_duration)

        # get line animation
        if style in ['parallel', True]:
            line_ani = animation.FuncAnimation(fig, update_lines_parallel, x[0].shape[0],
                            fargs=(x, lines, trail, 1, tail_duration, rotations, zoom, chemtrails, elev),
                            interval=1000/frame_rate, blit=False, repeat=False)
        elif style is 'serial':
            line_ani = animation.FuncAnimation(fig, update_lines_parallel, x[0].shape[0],
                            fargs=(x, lines, trail, 1, tail_duration, rotations, zoom, chemtrails, elev),
                            interval=1000/frame_rate, blit=False, repeat=False)
        elif style is 'spin':
            line_ani = animation.FuncAnimation(fig, update_lines_spin, x[0].shape[0],
                            fargs=(x, lines, 1, rotations, zoom, elev),
                            interval=1000/frame_rate, blit=False, repeat=False)

        return fig, ax, data, line_ani

    # handle explore flag
    if explore:
        assert x[0].ndim>1, "Explore mode is currently only supported for 3D plots."
        kwargs['picker']=True

    # turn kwargs into a list
    kwargs_list = parse_kwargs(x, mpl_kwargs)

    # turn format string into a list
    if format_string is not list:
        format_string = [(format_string) for i in range(len(x))]

    # draw the plot
    if animate in [True, 'parallel', 'serial', 'spin']:

        # animation params
        ani_params = dict(tail_duration=tail_duration,
                          rotations=rotations,
                          zoom=zoom,
                          chemtrails=chemtrails,
                          frame_rate=frame_rate,
                          elev=elev,
                          style=animate)

        # dispatch animation
        fig, ax, data, line_ani = dispatch_animate(x, ani_params)

    else:

        # dispatch static
        fig, ax, data = dispatch_static(x)

        # if 3d, plot the cube
        if x[0].shape[1] is 3:

            # Get cube scale from data
            cube_scale = 1

            # plot cube
            plot_cube(cube_scale)

            # Setting the axes properties
            ax.set_xlim3d([-cube_scale, cube_scale])
            ax.set_ylim3d([-cube_scale, cube_scale])
            ax.set_zlim3d([-cube_scale, cube_scale])

            # initialize the view
            ax.view_init(elev=elev, azim=azim)

        # set line_ani to empty
        line_ani = None

    # remove axes
    ax.set_axis_off()

    # add labels
    if labels is not None:
        add_labels(x, labels)

    # add legend
    if legend is not None:
        proxies = [plt.Rectangle((0, 0), 1, 1, fc=sns.color_palette()[idx]) for idx,label in enumerate(legend)]
        ax.legend(proxies,legend)

    return fig, ax, data, line_ani
