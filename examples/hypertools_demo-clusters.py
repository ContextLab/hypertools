import hypertools as hyp
import pandas as pd

data=pd.read_csv('sample_data/mushrooms.csv')

hyp.plot(data,'o',n_clusters=10)

# OR
# cluster_labels = hyp.util.cluster(data, n_clusters=10)
# hyp.plot(data,'o',point_colors=cluster_labels)
