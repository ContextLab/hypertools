import numpy as np
import pandas as pd

import pytest
import hypertools as hyp

def test_io():
    # note: the load function called BOTH load and save internally, so this test checks both load and save
    datasets = ['spiral', 'weights', 'weights_avg', 'weights_sample']
    for d in datasets:
        x = hyp.load(d)
        assert type(x) is list
        assert all([type(i) is np.ndarray for i in x])

    urls = ['https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/home_on_the_range.txt',
            'https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/testdata.csv',
            'https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/wrangler.jpg']
    types = [str, pd.DataFrame, np.ndarray]
    for i, u in enumerate(urls):
        x = hyp.load(u)
        assert type(x) is types[i]
