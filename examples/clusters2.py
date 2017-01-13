import hypertools as hyp
import numpy as np
from scipy.stats import multivariate_normal

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+3, np.eye(3), size=100)
data = np.vstack([cluster1,cluster2])

cluster_labels = hyp.tools.cluster(data,n_clusters=2)
print(cluster_labels)
hyp.plot(data,'o',group=cluster_labels)
