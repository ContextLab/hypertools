import hypertools as hyp
import pandas as pd
import os

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sample_data/')
data=pd.read_csv(datadir + 'mushrooms.csv')

hyp.plot(data,'o', group=data['class'])
