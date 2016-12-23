import hypertools as hyp
import pandas as pd

data=pd.read_csv('sample_data/mushrooms.csv')

df = pd.DataFrame()
df['class'] = data['class']
df['class'] = df['class'].replace('p', 1).replace('e', 0)
for colname in data.columns[1:]:
    df = df.join(pd.get_dummies(data[colname], prefix=colname))

data = df[df.columns[1:]]
hyp.plot(data,'o')
