#!/usr/bin/env python

import pandas as pd

def pandas_to_matrix(data, return_labels=False):
    df_str = data.select_dtypes(include=['object'])
    df_num = data.select_dtypes(exclude=['object'])
    for colname in df_str.columns:
        df_num = df_num.join(pd.get_dummies(data[colname], prefix=colname))
    plot_data = df_num.as_matrix()
    labels=list(df_num.columns.values)+list(df_str.columns.values)
    if return_labels:
        return plot_data,labels
    else:
        return plot_data
