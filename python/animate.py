from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from scipy.interpolate import PchipInterpolator as pchip
from sklearn.decomposition import PCA as PCA
import hyperalign as hyp

import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette(palette="Set2", n_colors=3)

# this will be moved to utils.py
def is_list(x):
    if type(x[0][0])==np.ndarray:
        return True
    elif type(x[0][0])==np.int64 or type(x[0][0])==int:
        return False

# #  this will be moved to utils.py
def interp_array(arr,interp_val=10):
    x=np.arange(0, len(arr), 1)
    xx=np.arange(0, len(arr)-1, 1/interp_val)
    q=pchip(x,arr)
    return q(xx)

# #  this will be moved to utils.py
def interp_array_list(arr_list,interp_val=10):
    smoothed= [np.zeros(arr_list[0].shape) for item in arr_list]
    for idx,arr in enumerate(arr_list):
        smoothed[idx] = interp_array(arr,interp_val)
    return smoothed

def get_cube_scale(x):
    x = np.vstack(x)
    x_square = np.square(x)
    x_ss = np.sum(x_square,axis=1)
    idx = [i for i in range(len(x_ss)) if x_ss[i]==np.max(x_ss)]
    print(np.linalg.norm(x[idx,:]))
    return np.linalg.norm(x[idx,:])

def animate(x):

    def update_lines(num, data_lines, lines, trail_lines, cube_scale, tail_len=50, tail_style=':', speed=1):

        if hasattr(update_lines, 'planes'):
            for plane in update_lines.planes:
                plane.remove()

        update_lines.planes = plot_cube(cube_scale)
        ax.view_init(elev=10, azim=speed*num/5)
        ax.dist=6

        for line, data, trail in zip(lines, data_lines, trail_lines):
            if num<=tail_len:
                    line.set_data(data[0:num+1, 0:2].T)
                    line.set_3d_properties(data[0:num+1, 2])
            else:
                line.set_data(data[num-tail_len:num+1, 0:2].T)
                line.set_3d_properties(data[num-tail_len:num+1, 2])
            if num>=tail_len:
                if num>=tail_len*2:
                    trail.set_data(data[num-tail_len*2:1+num-tail_len, 0:2].T)
                    trail.set_3d_properties(data[num-tail_len*2:1+num-tail_len, 2])
                    trail.set_linestyle(tail_style)
                else:
                    trail.set_data(data[0:1+num-tail_len, 0:2].T)
                    trail.set_3d_properties(data[0:1+num-tail_len, 2])
                    trail.set_linestyle(tail_style)
        return lines,trail_lines

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    x = interp_array_list(x)

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], linewidth=3)[0] for dat in x]
    trail = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in x]

    ax.set_axis_off()

    def plot_cube(scale, const=1.25):
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

    # Get cube scale from data
    cube_scale = get_cube_scale(x)

    # Setting the axes properties
    ax.set_xlim3d([-cube_scale, cube_scale])
    ax.set_ylim3d([-cube_scale, cube_scale])
    ax.set_zlim3d([-cube_scale, cube_scale])

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, 1000, fargs=(x, lines, trail, cube_scale),
                                   interval=8, blit=False)
    plt.show()

################################################################################
################################################################################

def Gen_RandLine(length, dims=2):
    """
    Create a line using a random walk algorithm

    length is the number of points for the line.
    dims is the number of dimensions the line has.
    """
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = ((np.random.rand(dims) - 0.5) * 0.1)
        lineData[:, index] = lineData[:, index - 1] + step

    return lineData

# import scipy.io as sio
#
# data = sio.loadmat('extras/example_data/weights.mat')
# test_data=data['weights'][0][1]
# test_data2=data['weights'][0][2]
# test_data3=data['weights'][0][3]
# data = np.array([test_data, test_data2, test_data3])

data = [Gen_RandLine(100,3).T for index in range(3)]
animate(data)
