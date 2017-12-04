#!/usr/bin/env python
import warnings
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration, SpectralClustering
from sklearn.mixture import GaussianMixture

import numpy as np
from .._shared.helpers import *


class Cluster(object):
    def __init__(self,x,model):
        self.x=x
        self.model=model
    def set_num_clusters(self,n_clusters):
        self.model_params = {
        'n_clusters' : n_clusters
        }
    def set_custom_params(self,custom_params):
        self.model_params = custom_params
    def initialize_model(self):
        self.model = self.model(**self.model_params)
    def fit_model(self):
        self.model.fit(np.vstack(self.x))
    def get_labels(self):
        return list(self.model.labels_)
class KMeansCluster(Cluster):
    def __init__(self,x):
        Cluster.__init__(self,x,KMeans)
class MiniBatchKMeansCluster(Cluster):
    def __init__(self,x):
        Cluster.__init__(self,x,MiniBatchKMeans)
class AgglomerativeClusteringCluster(Cluster):
    def __init__(self,x):
        Cluster.__init__(self,x,AgglomerativeClustering)
class BirchCluster(Cluster):
    def __init__(self,x):
        Cluster.__init__(self,x,Birch)
class FeatureAgglomerationCluster(Cluster):
    def __init__(self,x):
        Cluster.__init__(self,x,FeatureAgglomeration)
class SpectralClusteringCluster(Cluster):
    def __init__(self,x):
        Cluster.__init__(self,x,SpectralClustering)
class GaussianMixtureCluster(Cluster):
    def __init__(self,x):
        Cluster.__init__(self,x,GaussianMixture)
    def set_num_clusters(self,n_clusters):
        self.model_params = {
        'n_components' : n_clusters
        }
    def get_labels(self):
        return list(self.model.predict(np.vstack(self.x)))


@memoize
def cluster(x, cluster='KMeans', n_clusters=3, ndims=None):
    """
    Performs clustering analysis and returns a list of cluster labels

    Parameters
    ----------
    x : A Numpy array, Pandas Dataframe or list of arrays/dfs
        The data to be clustered.  You can pass a single array/df or a list.
        If a list is passed, the arrays will be stacked and the clustering
        will be performed across all lists (i.e. not within each list).

    cluster : str or dict
        Model to use to discover clusters.  Support algorithms are: KMeans,
        MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration,
        SpectralClustering, GaussianMixture (default: KMeans).Can be passed as a string, but for
        finer control of the model parameters, pass as a dictionary, e.g.
        reduce={'model' : 'KMeans', 'params' : {'max_iter' : 100}}. See
        scikit-learn specific model docs for details on parameters supported for
        each model.

    n_clusters : int
        Number of clusters to discover

    ndims : None
        Deprecated argument.  Please use new analyze function to perform
        combinations of transformations

    Returns
    ----------
    cluster_labels : list
        An list of cluster labels

    """

    # if cluster is None, just return data
    if cluster is None:
        return x
    else:

        if ndims is not None:
            warnings.warn('The ndims argument is now deprecated. Ignoring dimensionality reduction step.')

        x = format_data(x, ppca=True)

        # dictionary of models
        models = {
            'KMeans' : KMeansCluster,
            'MiniBatchKMeans' : MiniBatchKMeansCluster,
            'AgglomerativeClustering' : AgglomerativeClusteringCluster,
            'FeatureAgglomeration' : FeatureAgglomerationCluster,
            'Birch' : BirchCluster,
            'SpectralClustering' : SpectralClusteringCluster,
            'GaussianMixture'   : GaussianMixtureCluster
        }


        # if reduce is a string, find the corresponding model
        if type(cluster) is str:
            model = models[cluster](x)
            model.set_num_clusters(n_clusters)
 
        # if its a dict, use custom params
        elif type(cluster) is dict:
            if type(cluster['model']) is str:
                model = models[cluster['model']](x)
                model.set_custom_params(cluster['params'])

        # initialize model
        model.initialize_model()

        # fit the model
        model.fit_model()

        # return the labels
        return model.get_labels()