# noinspection PyPackageRequirements
import datawrangler as dw
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


# noinspection DuplicatedCode
class Manipulator(BaseEstimator):
    def __init__(self, **kwargs):
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.transformer = kwargs.pop('transformer', None)
        self.required = kwargs.pop('required', [])
        self.kwargs = kwargs

    def fit(self, data):
        assert data is not None, ValueError('cannot manipulate an empty dataset')
        self.data = data

        if self.fitter is None:
            NotFittedError('null fit function; returning without fitting manipulator')
            return

        params = self.fitter(data, **self.kwargs)
        assert type(params) is dict, ValueError('fit function must return a dictionary')
        assert all([r in params.keys() for r in self.required]), ValueError('one or more required fields not'
                                                                            'returned')
        for k, v in params.items():
            setattr(self, k, v)

    def transform(self, *_):
        assert self.data is not None, NotFittedError('must fit manipulator before transforming data')
        for r in self.required:
            assert hasattr(self, r), NotFittedError(f'missing fitted attribute: {r}')

        if self.transformer is None:
            RuntimeWarning('null transform function; returning without manipulating data')
            return self.data

        required_params = {r: getattr(self, r) for r in self.required}
        return dw.decorate.apply_stacked(self.transformer(self.data, **dw.core.update_dict(required_params,
                                                                                           self.kwargs)))

    def fit_transform(self, data):
        self.fit(data)
        return self.transform()
