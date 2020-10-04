class Plot(object):
    def __init__(self, plotter, **args, **kwargs):
        self.plotter = plotter
        self.args = args
        self.kwargs = kwargs

    def draw(self, data):
        return self.plotter(data, **self.args, **self.kwargs)

    def save(self, **args, **kwargs):
        pass

    def get_args(self):
        return self.args

    def set_args(self, **args):
        self.args = args

    def get_kwargs(self):
        return self.kwargs

    def set_kwargs(self, **kwargs):
        self.kwargs = kwargs

    def update_kwargs(self, **kwargs):
        for key, val in kwargs:
            self.kwargs[key] = val
