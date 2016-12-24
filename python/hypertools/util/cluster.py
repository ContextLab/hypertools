from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster(x, n_clusters=8, ndims=None):
    if type(x) is list:
        x = np.vstack(x)
    if ndims:
        x = PCA(n_components=ndims).fit_transform(x)
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(x)
    return kmeans.labels_
