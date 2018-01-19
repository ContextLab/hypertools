parameters = {
    'KMeans': {'n_clusters': 5},
    'MiniBatchKMeans': {'n_clusters': 5},
    'SpectralClustering': {'n_clusters': 5,
                           'affinity' : 'nearest_neighbors',
                           'n_neighbors' : 10},
    'AgglomerativeClustering': {'n_clusters': 5, 'linkage' : 'ward'},
    'FeatureAgglomeration': {'n_clusters': 5},
    'Birch': {'n_clusters': 5},
    'HDBSCAN': {'min_samples': 5, 'min_cluster_size': 15}
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

    params = parameters[model].copy()
    if update_dict:
        params.update(update_dict)

    return params
