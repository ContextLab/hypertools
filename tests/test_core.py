# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import warnings

import hypertools as hyp
import sklearn
import flair
import importlib

sklearn_models = {
    'calibration': ['CalibratedClassifierCV', 'calibration_curve'],
    'cluster': ['AffinityPropagation', 'AgglomerativeClustering', 'Birch', 'DBSCAN', 'FeatureAgglomeration',
                'KMeans', 'MiniBatchKMeans', 'MeanShift', 'OPTICS', 'SpectralClustering', 'SpectralBiclustering',
                'SpectralCoclustering'],
    'compose': ['ColumnTransformer', 'TransformedTargetRegressor'],
    'covariance': ['EmpiricalCovariance', 'EllipticEnvelope', 'GraphicalLasso', 'GraphicalLassoCV', 'LedoitWolf',
                   'MinCovDet', 'OAS', 'ShrunkCovariance'],
    'cross_decomposition': ['CCA', 'PLSCanonical', 'PLSRegression', 'PLSSVD'],
    'discriminant_analysis': ['LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis'],
    'ensemble': ['AdaBoostClassifier', 'AdaBoostRegressor', 'BaggingClassifier', 'ExtraTreesClassifier',
                 'ExtraTreesRegressor', 'GradientBoostingClassifier', 'GradientBoostingRegressor',
                 'IsolationForest', 'RandomForestClassifier', 'RandomForestRegressor', 'RandomTreesEmbedding',
                 'StackingClassifier', 'StackingRegressor', 'VotingClassifier', 'VotingRegressor',
                 'HistGradientBoostingRegressor', 'HistGradientBoostingClassifier'],
    'feature_extraction': ['DictVectorizer', 'FeatureHasher'],
    'feature_extraction.image': ['PatchExtractor'],
    'feature_extraction.text': ['CountVectorizer', 'HashingVectorizer', 'TfidfTransformer', 'TfidfVectorizer'],
    'feature_selection': ['GenericUnivariateSelect', 'SelectPercentile', 'SelectKBest', 'SelectFpr', 'SelectFdr',
                          'SelectFromModel', 'SelectFwe', 'SequentialFeatureSelector', 'RFE', 'RFECV',
                          'VarianceThreshold'],
    'gaussian_process': ['GaussianProcessClassifier', 'GaussianProcessRegressor'],
    'impute': ['SimpleImputer', 'IterativeImputer', 'MissingIndicator', 'KNNImputer'],
    'isotonic': ['IsotonicRegression'],
    'kernel_approximation': ['AdditiveChi2Sampler', 'Nystroem', 'PolynomialCountSketch', 'RBFSampler',
                             'SkewedChi2Sampler'],
    'kernel_ridge': ['KernelRidge'],
    'linear_model': ['LogisticRegression', 'LogisticRegressionCV', 'PassiveAggressiveClassifier', 'Perceptron',
                     'Ridge', 'RidgeCV', 'SGDRegressor', 'ElasticNet', 'ElasticNetCV', 'Lars', 'LarsCV', 'Lasso',
                     'LassoCV', 'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'OrthogonalMatchingPursuit',
                     'OrthogonalMatchingPursuitCV', 'ARDRegression', 'BayesianRidge', 'MultiTaskElasticNet',
                     'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV', 'HuberRegressor',
                     'RANSACRegressor', 'TheilSenRegressor', 'PoissonRegressor', 'TweedieRegressor',
                     'GammaRegressor', 'PassiveAggressiveRegressor'],
    'manifold': ['Isomap', 'LocallyLinearEmbedding', 'MDS', 'SpectralEmbedding', 'TSNE'],
    'mixture': ['BayesianGaussianMixture', 'GaussianMixture'],
    'model_selection': ['GroupKFold', 'GroupShuffleSplit', 'KFold', 'LeaveOneGroupOut', 'LeavePGroupsOut',
                        'LeaveOneOut', 'LeavePOut', 'PredefinedSplit', 'RepeatedKFold', 'RepeatedStratifiedKFold',
                        'ShuffleSplit', 'StratifiedKFold', 'StratifiedShuffleSplit', 'TimeSeriesSplit',
                        'GridSearchCV', 'HalvingGridSearchCV', 'ParameterGrid', 'ParameterSampler',
                        'RandomizedSearchCV', 'HalvingRandomSearchCV'],
    'multiclass': ['OneVsRestClassifier', 'OneVsOneClassifier', 'OutputCodeClassifier'],
    'multioutput': ['ClassifierChain', 'MultiOutputRegressor', 'MultiOutputClassifier', 'RegressorChain'],
    'naive_bayes': ['BernoulliNB', 'CategoricalNB', 'ComplementNB', 'GaussianNB', 'MultinomialNB'],
    'neighbors': ['BallTree', 'DistanceMetric', 'KDTree', 'KernelDensity', 'KNeighborsClassifier',
                  'KNeighborsRegressor', 'KNeighborsTransformer', 'LocalOutlierFactor', 'RadiusNeighborsClassifier',
                  'RadiusNeighborsRegressor', 'RadiusNeighborsTransformer', 'NearestCentroid', 'NearestNeighbors',
                  'NeighborhoodComponentsAnalysis'],
    'neural_network': ['BernoulliRBM', 'MLPClassifier', 'MLPRegressor'],
    'pipeline': ['FeatureUnion', 'Pipeline'],
    'preprocessing': ['Binarizer', 'FunctionTransformer', 'KBinsDiscretizer', 'KernelCenterer', 'LabelBinarizer',
                      'LabelEncoder', 'MultiLabelBinarizer', 'MaxAbsScaler', 'MinMaxScaler', 'Normalizer',
                      'OneHotEncoder', 'PolynomialFeatures', 'PowerTransformer', 'QuantileTransformer',
                      'RobustScaler', 'StandardScaler'],
    'random_projection': ['GaussianRandomProjection', 'SparseRandomProjection'],
    'semi_supervised': ['LabelPropagation', 'LabelSpreading', 'SelfTrainingClassifier'],
    'svm': ['LinearSVC', 'LinearSVR', 'NuSVC', 'NuSVR', 'OneClassSVM', 'SVC', 'SVR'],
    'tree': ['DecisionTreeClassifier', 'DecisionTreeRegressor', 'ExtraTreeClassifier', 'ExtraTreeRegressor']}


class HyperTest:
    def __init__(self, a, b, c, d, e, test):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.test = test


def test_get_model():
    for module_name in sklearn_models.keys():
        module = importlib.import_module(f'sklearn.{module_name}')
        for x in sklearn_models[module_name]:
            hypertools_model = hyp.core.get_model(x)
            sklearn_model = getattr(module, x)

            assert hypertools_model is sklearn_model


def test_apply_model():
    # single dataset
    m = hyp.core.apply_model(np.random.randn(10, 20), 'Binarizer')
    assert all([i in [0, 1] for i in np.unique(m)])

    pca = {'model': 'PCA', 'args': [], 'kwargs': {'n_components': 5}}
    m = hyp.core.apply_model(np.random.randn(100, 10), model=pca)
    assert m.shape == (100, 5)

    # list of arrays
    m = hyp.core.apply_model([np.random.randn(10, 5) for _ in range(3)], 'MinMaxScaler')
    assert type(m) is list
    assert len(m) == 3
    assert all([i.shape == (10, 5) for i in m])
    assert all([dw.util.btwn(i, -0.0001, 1.0001) for i in m])

    # multiple models, multiple datasets
    incremental_pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 3}}
    m = hyp.core.apply_model([np.random.randn(100, 10) for _ in range(5)], model=[pca, 'MinMaxScaler', 'Binarizer',
                                                                                  incremental_pca])
    assert type(m) is list
    assert len(m) == 5
    assert all([i.shape == (100, 3) for i in m])

    # now apply every model to each of two datasets-- one single array and one list of arrays
    x1 = np.random.randn(50, 4)
    x2 = [np.random.randn(10, 5) for _ in range(4)]

    skip = ['sample', 'ridge', 'perceptron', 'cv', 'net', 'lars', 'match', 'lasso', 'task', 'fold', 'split', 'group',
            'out', 'grid', 'nb', 'tree', 'metric', 'kernel', 'neighbor', 'project', 'feature', 'pipe', 'bins', 'label',
            'encode', 'svc', 'svr', 'regress', 'classif', 'centroid', 'calibration', 'biclustering', 'coclustering',
            'columntransformer', 'cov', 'ledoit', 'oas', 'cca', 'pls', 'discriminant', 'vectorizer', 'extract',
            'select', 'transformer', 'rfe']

    for module_name in sklearn_models.keys():
        for m in sklearn_models[module_name]:
            if any([i in m.lower() for i in skip]):
                continue

            warnings.simplefilter('ignore')
            x1_fit = hyp.core.apply_model(x1, model=m)
            x2_fit = hyp.core.apply_model(x2, model=m)

            assert x1_fit.shape[0] == x1.shape[0]
            assert type(x2_fit) == list
            assert len(x2_fit) == len(x2)
            assert all([x.shape[0] == 10 for x in x2_fit])


def test_has_all_attributes():
    x = HyperTest(1, 2, 3, 4, 5, 6)
    assert hyp.core.has_all_attributes(x, ['b', 'c', 'd'])
    assert hyp.core.has_all_attributes(x, ['a', 'b', 'c', 'd', 'e', 'test'])
    assert not hyp.core.has_all_attributes(x, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])


def test_has_any_attributes():
    x = HyperTest(1, 2, 3, 4, 5, 6)
    assert hyp.core.has_all_attributes(x, ['b', 'c', 'd'])
    assert hyp.core.has_any_attributes(x, ['a', 'b', 'c', 'd', 'e'])
    assert hyp.core.has_any_attributes(x, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    assert hyp.core.has_any_attributes(x, ['e', 'f', 'g', 'h'])
    assert not hyp.core.has_any_attributes(x, ['f', 'g', 'h', 'i', 'j'])
    assert hyp.core.has_any_attributes(x, ['f', 'g', 'h', 'test', 'i', 'j'])


def test_get_default_options():
    defaults = hyp.get_default_options()
    assert type(defaults) is hyp.core.shared.RobustDict
    assert type(defaults['DoesNotExist']) is dict and len(defaults['DoesNotExist']) == 0
    assert defaults['HyperAlign']['n_iter'] == '10'
    assert defaults['CountVectorizer']['stop_words'] == "'english'"


def test_get():
    x = [1, 2, 3, 4, 5]
    for i in range(5):
        assert hyp.get(x, i) == i + 1
        assert hyp.get(x, i + len(x)) == i + 1

    x = np.cumsum(np.cumsum(np.ones([10, 5]), axis=0), axis=1)
    # axis == 0
    for i in range(10):
        assert np.allclose(hyp.get(x, i, axis=0), x[i, :])
        assert np.allclose(hyp.get(x, i + 2 * x.shape[0], axis=0), x[i, :])

    assert np.allclose(hyp.get(x, range(x.shape[0] * 2)), np.vstack([x, x]))

    # axis == 1
    for i in range(5):
        assert np.allclose(hyp.get(x, i, axis=1), x[:, i])
        assert np.allclose(hyp.get(x, i + 3 * x.shape[1], axis=1), x[:, i])

    assert np.allclose(hyp.get(x, range(x.shape[1] * 3), axis=1), np.hstack([x, x, x]))

    # dataframe
    y = pd.DataFrame(x, index=range(0, 20, 2))

    for i in range(x.shape[0]):
        assert np.allclose(hyp.get(x, i), hyp.get(y, i))
        assert np.allclose(hyp.get(x, i), hyp.get(y, i + 20 * x.shape[0]))

    for i in range(x.shape[1]):
        assert np.allclose(hyp.get(x, i, axis=1), hyp.get(y, i, axis=1).values.ravel())
        assert np.allclose(hyp.get(x, i, axis=1), hyp.get(y, i + 5 * x.shape[0], axis=1).values.ravel())

    z = ['test1', 'test2', 'test3']
    for i in range(len(z) * 10):
        assert hyp.get(z, i) == z[i % len(z)]


def test_fullfact():
    def check(ff, x):
        return np.allclose(np.array(ff), np.array(x))

    assert check(hyp.fullfact([1, 1, 1, 1]), [1, 1, 1, 1])

    x1 = [[1, 1],
          [2, 1],
          [1, 2],
          [2, 2]]
    assert check(hyp.fullfact([2, 2]), x1)

    x2 = [[1, 1, 1],
          [1, 2, 1],
          [1, 1, 2],
          [1, 2, 2],
          [1, 1, 3],
          [1, 2, 3]]
    assert check(hyp.fullfact([1, 2, 3]), x2)

    x3 = [[1, 1, 1],
          [2, 1, 1],
          [3, 1, 1],
          [1, 1, 2],
          [2, 1, 2],
          [3, 1, 2],
          [1, 1, 3],
          [2, 1, 3],
          [3, 1, 3]]
    assert check(hyp.fullfact([3, 1, 3]), x3)


def test_eval_dict():
    pass


def test_unpack_model():
    pass


def test_robust_dict():
    pass
