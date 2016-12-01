import plot_coords as coords
import scipy.io as sio
import numpy as np

data=sio.loadmat('examples/weights.mat')
w=data['weights'][0][0:3]

# labels=[]
# for idx,i in enumerate(w):
#     tmp=[]
#     for iidx,ii in enumerate(i):
#         if iidx==0:
#             tmp.append('Point ' + str(idx))
#         else:
#             tmp.append(None)
#     labels.append(tmp)
#
# coords.plot_coords(w,'o',labels=labels)

# example of point colors
point_colors=[]
for idx,i in enumerate(w):
    tmp=[]
    for iidx,ii in enumerate(i):
        if iidx==0:
            tmp.append('first_point')
        else:
            tmp.append('other_points')
    point_colors.append(tmp)

coords.plot_coords(w,'o',point_colors=point_colors)
