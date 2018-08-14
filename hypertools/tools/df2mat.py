#!/usr/bin/env python

import pandas as pd


def df2mat(data, return_labels=False):
    """
    Transforms a Pandas DataFrame into a Numpy array with binarized text columns

    This function transforms single-level df to an array so it can be plotted
    with HyperTools.  Additionally, it uses the Pandas.Dataframe.get_dummies
    function to transform text columns into binary vectors, or
    'dummy variables'.

    Parameters
    ----------
    data : A single-level Pandas DataFrame
        The df that you want to convert.  Note that this currently only works
        with single-level (not Multi-level indices).

    Returns
    ----------
    plot_data : Numpy array
        A Numpy array where text columns are turned into binary vectors.

    labels : list (optional)
        A list of column labels for the numpy array. To return this, set
        return_labels=True.

    """

    df_str = data.select_dtypes(include=['object'])
    df_num = data.select_dtypes(exclude=['object'])

    for colname in df_str.columns:
        df_num = df_num.join(pd.get_dummies(data[colname], prefix=colname))

    plot_data = df_num.values()

    labels=list(df_num.columns.values)

    if return_labels:
        return plot_data,labels
    else:
        return plot_data
