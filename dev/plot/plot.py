# TODO: copy relevant stuff from hypertools_revamp notebook.  key things to do:
#  1.) funnel data
#  2.) specify default reduce args if n_dims > 3 after applying other stuff
#  3.) allow optional calls to reduce (overwrite), cluster, manipulate, and align.
#      user specifies order via a list of models (similar to sklearn Pipeline)
#  4.) parse plot-specific arguments (defaults specified in config.ini)
#  5.) handle multi-index dataframes
#  6.) draw bounding box
#  7.) move/position the camera as needed
#  8.) given current plotting backend, generate the plot
#    a.) to generate an animation, do steps 1--6 and then filter data and/or adjust camera for each animation frame

# import holoviews as hv
#
#
# def backend(engine, *args, **kwargs):
#     hv.extension(engine, *args, **kwargs)
#
#
# def init_notebook_mode():
#     hv.notebook_extension()
#
#
# class Plot(object):
#     # noinspection PyShadowingNames
#     def __init__(self, backend, *args, **kwargs):
#         assert (backend in engine.keys()), 'Unknown backend plot engine: ' + backend
#         self.backend = engine[backend]
#         self.args = args
#         self.kwargs = kwargs
#
#     def draw(self, data):
#         return self.plotter(data, *self.args, **self.kwargs)
#
#     def save(self, *args, **kwargs):
#         pass
#
#     def get_args(self):
#         return self.args
#
#     def set_args(self, **args):
#         self.args = args
#
#     def get_kwargs(self):
#         return self.kwargs
#
#     def set_kwargs(self, **kwargs):
#         self.kwargs = kwargs
#
#     def update_kwargs(self, **kwargs):
#         for key, val in kwargs:
#             self.kwargs[key] = val
