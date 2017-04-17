import pandas as pd

def summary(df, n=5, pct=[0.1, 0.5, 0.9]):
    '''
    Summarize DataFrame along columns for data type, sample size, numerical statistics and frequency
    
    Parameters
    ----------
    df : DataFrame
        To summary.
    n : int
        The number of foremost frequent categories of frequency table.
    pct : list
        Percentiles for numerical statistics.

    Returns
    -------
    op : DataFrame
        summary of DataFrame along columns.
    '''
    
    def freq(s):
        op = pd.value_counts(s)
        op = pd.concat([pd.Series(op.index[:n]).rename(lambda x: "FreqCat{}".format(x+1)), 
                        pd.Series(op.values[:n]).rename(lambda x: "FreqVal{}".format(x+1)).T, 
                        pd.Series(op.iloc[n:].sum(), index=["Freq_Others"])])
        return(op)
    op = pd.concat([df.dtypes.rename("Type"), 
                    df.notnull().sum().rename("N"), 
                    df.describe(pct).iloc[1:].T, 
                    df.apply(freq).T], axis=1).loc[df.columns]
    return(op)
