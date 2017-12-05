# -*- coding: utf-8 -*-

import pytest

from scipy.stats import multivariate_normal
import numpy as np

from hypertools.tools.cluster import cluster

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+100, np.eye(3), size=100)
data = np.vstack([cluster1,cluster2])
labels_kmeans = cluster(data,n_clusters=2)
labels_kmeans_custom=cluster(data,cluster={'model':'KMeans','params':{'n_clusters':3}})
labels_gaussian_prob=cluster(data, cluster='GaussianMixture', n_clusters=2)
labels_gaussian_prob_custom=cluster(data, cluster={'model':'GaussianMixture','params':{'n_components':3}})
labels_bayesian_gaussian_prob=cluster(data, cluster='BayesianGaussianMixture', n_clusters=2)
labels_bayesian_gaussian_prob_custom=cluster(data, cluster={'model':'BayesianGaussianMixture','params':{'n_components':3}})

def test_cluster_n_clusters():
    assert (len(set(labels_kmeans))==2) and (len(set(labels_kmeans_custom))==3) and (len(labels_gaussian_prob[0])==2) and \
    	(len(labels_gaussian_prob_custom[0])==3) and (len(labels_bayesian_gaussian_prob[0])==2) and (len(labels_bayesian_gaussian_prob_custom[0])==3)

def test_cluster_returns_list():
    assert (type(labels_kmeans) is list) and (type(labels_kmeans_custom) is list) and (type(labels_gaussian_prob) is list) and \
    	(type(labels_gaussian_prob_custom) is list) and (type(labels_bayesian_gaussian_prob) is list) and (type(labels_bayesian_gaussian_prob_custom) is list)

def test_cluster_gaussian_returns_list_of_probs():
	assert (type(labels_gaussian_prob[0]) is np.ndarray) and (type(labels_gaussian_prob_custom[0]) is np.ndarray) and \
		(type(labels_bayesian_gaussian_prob[0]) is np.ndarray) and (type(labels_bayesian_gaussian_prob_custom[0]) is np.ndarray)
