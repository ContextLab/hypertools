import numpy as np
import pandas as pd

import hypertools as hyp
import sklearn
import flair
import importlib


def test_get_model():
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

    for module_name in sklearn_models.keys():
        module = importlib.import_module(f'sklearn.{module_name}')
        for x in sklearn_models[module_name]:
            hypertools_model = hyp.core.get_model(x)
            sklearn_model = getattr(module, x)

            assert hypertools_model is sklearn_model


test_get_model()


def test_apply_model():
    pass


def test_has_all_attributes():
    pass


def test_has_any_attributes():
    pass


def test_get_default_options():
    pass


def test_get():
    pass


def test_fullfact():
    pass


def test_eval_dict():
    pass


def test_unpack_model():
    pass


def test_robust_dict():
    pass
