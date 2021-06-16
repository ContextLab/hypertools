class NullAlign:
    '''
    Template for creating new HyperAlignment classes
    '''

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.(k) = v

    def fit(self, data):
        pass

    def transform(self, data):
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)