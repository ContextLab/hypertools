import hypertools as hyp
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data=pd.read_csv('sample_data/mushrooms.csv')

df = pd.DataFrame()
df['class'] = data['class']
df['class'] = df['class'].replace('p', 1).replace('e', 0)
for colname in data.columns[1:]:
    df = df.join(pd.get_dummies(data[colname], prefix=colname))

data = df[df.columns[1:]].as_matrix()
hyp.plot(data,'o',n_clusters=10)

# OR
# cluster_labels = hyp.util.cluster(data, n_clusters=10)
# hyp.plot(data,'o',point_colors=cluster_labels)
