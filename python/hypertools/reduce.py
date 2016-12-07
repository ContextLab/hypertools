from sklearn.decomposition import PCA

def reduce(arr,ndims=3):
    return PCA(arr,ndims)
