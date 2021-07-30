class NullAlign:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, data):
        pass

    # noinspection PyMethodMayBeStatic
    def transform(self, data):
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
