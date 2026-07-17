# NOTE: the default cluster count below must match hyp.cluster's signature
# default (n_clusters=3, hypertools/cluster/cluster.py) -- 2026-07 audit
# F23-core-config-exceptions-010: these used to be 5, so
# hyp.plot(x, cluster='KMeans') silently produced a different partition (5
# clusters) than hyp.cluster(x) (3 clusters) on identical data. Enforced by
# tests/test_core_audit_fixes.py.
parameters = {
    'KMeans': {'n_clusters': 3},
    'MiniBatchKMeans': {'n_clusters': 3},
    'SpectralClustering': {'n_clusters': 3,
                           'affinity': 'nearest_neighbors',
                           'n_neighbors': 10},
    'AgglomerativeClustering': {'n_clusters': 3, 'linkage' : 'ward'},
    'FeatureAgglomeration': {'n_clusters': 3},
    'Birch': {'n_clusters': 3},
    'HDBSCAN': {'min_samples': 5, 'min_cluster_size': 15},
    'GaussianMixture': {'n_components': 3},
    'BayesianGaussianMixture': {'n_components': 3},
    'CountVectorizer': {},
    'TfidfVectorizer': {},
    'LatentDirichletAllocation': {'n_components': 20, 'learning_method': 'batch'},
    'NMF': {'n_components': 20}
}


def default_params(model, update_dict=None):
    """
    Loads and updates default model parameters

    Parameters
    ----------

    model : str
        The name of a model

    update_dict : dict
        A dict to update default parameters

    Returns
    ----------

    params : dict
        A dictionary of parameters
    """

    if model in parameters:
        params = parameters[model].copy()
    else:
        params = None

    if update_dict:
        if params is None:
            params = {}
        params.update(update_dict)

    return params
