import hypertools as hyp
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+10, np.eye(3), size=100)

data = [cluster1,cluster2]

normalized_across = hyp.tools.normalize(data,normalize='across')
fig,ax,data = hyp.plot(normalized_across, 'o', show=False, return_data=True)
ax.set_title('z-score columns across all lists')
plt.show()

normalized_within = hyp.tools.normalize(data, normalize='within')
fig,ax,data = hyp.plot(normalized_within, 'o', show=False, return_data=True)
ax.set_title('z-score columns within each list')
plt.show()

normalized_row = hyp.tools.normalize(data, normalize='row')
fig,ax,data = hyp.plot(normalized_row, 'o', show=False, return_data=True)
ax.set_title('z-score each row')
plt.show()
