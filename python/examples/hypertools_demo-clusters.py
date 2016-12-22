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

ind_vars = df[df.columns[1:]]
m = PCA(n_components=3)
reduced_data = m.fit_transform(ind_vars)

hyp.plot(reduced_data,'o',n_clusters=10)
