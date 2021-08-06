import numpy as np
import pandas as pd

import hypertools as hyp

cluster1 = np.random.multivariate_normal(np.zeros(5), np.eye(5), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(5)+100, np.eye(5), size=300)
clusters = pd.DataFrame(np.vstack([cluster1, cluster2]))
true_labels = pd.DataFrame(data=np.concatenate([np.zeros(cluster1.shape[0]), np.ones(cluster2.shape[0])]),
                           index=clusters.index)


def test_discrete_clusters():
    def homogeneity_test(estimates, truth, threshold=0.95):
        for x in np.unique(estimates.values):
            if x == -1:
                continue
            inds = np.where(estimates.values == x)[0]
            if len(inds) > 0:
                zeros = truth.iloc[inds].values == 0
                ones = truth.iloc[inds].values == 1

                assert ((np.sum(zeros) / len(inds)) >= threshold) or ((np.sum(ones) / len(inds)) >= threshold)

    models = ['AffinityPropagation', 'AgglomerativeClustering', 'Birch', 'DBSCAN', 'FeatureAgglomeration', 'KMeans',
              'MiniBatchKMeans', 'MeanShift', 'SpectralClustering']

    for m in models:
        labels = hyp.cluster(clusters, model=m)
        homogeneity_test(labels, true_labels)

        labels2 = hyp.cluster([cluster1, cluster2], model=m)
        homogeneity_test(labels2[0], true_labels.iloc[:cluster1.shape[0]])
        homogeneity_test(labels2[1], true_labels.iloc[cluster1.shape[0]:])


def test_cluster_mixture():
    models = ['Birch', 'FeatureAgglomeration']
    pass


test_discrete_clusters()
