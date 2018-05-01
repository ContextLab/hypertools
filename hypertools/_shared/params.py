parameters = {
    'KMeans': {'n_clusters': 5},
    'MiniBatchKMeans': {'n_clusters': 5},
    'SpectralClustering': {'n_clusters': 5,
                           'affinity': 'nearest_neighbors',
                           'n_neighbors': 10},
    'AgglomerativeClustering': {'n_clusters': 5, 'linkage' : 'ward'},
    'FeatureAgglomeration': {'n_clusters': 5},
    'Birch': {'n_clusters': 5},
    'HDBSCAN': {'min_samples': 5, 'min_cluster_size': 15},
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
