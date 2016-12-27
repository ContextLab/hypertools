import hypertools as hyp
import pandas as pd

data=pd.read_csv('sample_data/mushrooms.csv')

hyp.plot(data,'o', point_colors=data['class'])
