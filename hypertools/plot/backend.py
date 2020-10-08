import inspect
import sys
import warnings
from functools import wraps
from os import getenv

import matplotlib as mpl
import matplotlib.pyplot as plt

HYPERTOOLS_BACKEND = None
IS_NOTEBOOK = False
IPYTHON_INSTANCE = None
BACKEND_WARNING = None


def set_backend():
    global HYPERTOOLS_BACKEND
    global IS_NOTEBOOK
    global IPYTHON_INSTANCE
    global BACKEND_WARNING

    curr_backend = mpl.get_backend()
    try:
        # passes if imported from Jupyter notebook
        IPYTHON_INSTANCE = get_ipython()
        assert 'IPKernelApp' in IPYTHON_INSTANCE.config
        IS_NOTEBOOK = True
        # if running in a notebook, should almost always use nbAgg. May
        # eventually let user override this with environment variable
        # (e.g., to use ipympl or WXAgg in JupyterLab), but switching
        # backends in IPython is already a nightmare
        try:
            mpl.use('nbAgg')
            working_backend = 'nbAgg'
        except ImportError:
            BACKEND_WARNING = "Failed to switch to interactive notebook " \
                              "backend ('nbAgg'). Falling back to inline " \
                              "static plots."
            # how "%matplotlib inline" is translated matplotlib
            working_backend = 'module://ipykernel.pylab.backend_inline'

    except (NameError, AssertionError):
        # NameError: imported from script
        # AssertionError: imported from IPython shell
        IS_NOTEBOOK = False
        # excluding WebAgg. No way to tell whether it will work until run
        backends = ('TkAgg', 'Qt5Agg', 'Qt4Agg', 'WXAgg', 'GTK3Agg')
        if sys.platform == 'darwin':
            # prefer cocoa backend on Mac. Pretty much guaranteed to
            # work, and Mac does NOT like Tkinter
            backends = ('MacOSX', *backends)

        # check environment variable
        env_backend = getenv("HYPERTOOLS_BACKEND")
        if env_backend is not None:
            if env_backend.lower() in tuple(map(str.lower, backends)):
                backends = (backends[:backends.index(HYPERTOOLS_BACKEND)],
                            *backends[backends.index(HYPERTOOLS_BACKEND) + 1:])
            backends = (env_backend, *backends)

        for b in backends:
            try:
                mpl.use(b)
                working_backend = b
                break

            except ImportError:
                continue

        else:
            BACKEND_WARNING = "Failed to switch to an interactive backend " \
                              f"({', '.join(backends)}. Falling back to 'Agg'."
            working_backend = 'Agg'

        if (
                env_backend is not None and
                working_backend.lower() != env_backend.lower()
        ):
            # The only time a warning is issued immediately on import is if
            # $HYPERTOOLS_BACKEND env var specifies an incompatible backend,
            # since that will have been set manually.
            warnings.warn("failed to set matplotlib backend to backend "
                          f"specified in environment ('{env_backend}'). "
                          f"Falling back to '{working_backend}'")

    finally:
        # restore backend
        mpl.use(curr_backend)
        HYPERTOOLS_BACKEND = working_backend


def _defer_backend_reset(backend, ipython_instance):
    def _reset_backend_cb():
        plt.switch_backend(_reset_backend_cb.backend)
        _reset_backend_cb.ipython_instance.events.unregister('pre_execute',
                                                             _reset_backend_cb)

    _reset_backend_cb.backend = backend
    _reset_backend_cb.ipython_instance = ipython_instance
    ipython_instance.events.register('pre_execute', _reset_backend_cb)


def manage_backend(plot_func):
    @wraps(plot_func)
    def plot_wrapper(*args, **kwargs):
        global IPYTHON_INSTANCE
        try:
            # get current rcParams to restore after plot
            curr_rcParams = mpl.rcParams.copy()
            main_backend = mpl.get_backend()

            if main_backend.lower() != HYPERTOOLS_BACKEND.lower():
                # some object inspection magic to get arg values passed
                func_signature = inspect.signature(plot_func)
                bound_args = func_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                all_argvalues = bound_args.arguments
                if all_argvalues.get('animate') or all_argvalues.get('interactive'):
                    if BACKEND_WARNING is not None:
                        warnings.warn(BACKEND_WARNING)

                    if IS_NOTEBOOK:
                        from ipykernel.pylab.backend_inline import flush_figures
                        # see (1) below re: unregistering `flush_figures` callback
                        while flush_figures in IPYTHON_INSTANCE.events.callbacks['post_execute']:
                            IPYTHON_INSTANCE.events.unregister('post_execute', flush_figures)

                    plt.switch_backend(HYPERTOOLS_BACKEND)

            return plot_func(*args, **kwargs)

        finally:
            # if the backend was switched
            if mpl.get_backend().lower() != main_backend.lower():
                if IS_NOTEBOOK:
                    # see (2) below re: deferring resetting the backend in notebooks
                    _defer_backend_reset(backend=main_backend,
                                         ipython_instance=IPYTHON_INSTANCE)
                    # we want the `flush_figures` callback for inline
                    # displays, so re-register it
                    if flush_figures not in IPYTHON_INSTANCE.events.callbacks['post_execute']:
                        IPYTHON_INSTANCE.events.register('post_execute', flush_figures)
                else:
                    # TODO: maybe make this blocking instead of immediate?
                    plt.switch_backend(main_backend)

            with warnings.catch_warnings():
                # if the user's matplotlibrc file was cached from v3.3.2
                # or earlier, there are a *ton* of (currently) harmless
                # MatplotlibDeprecationWarnings about axes.Axes3D
                warnings.simplefilter('ignore', mpl.MatplotlibDeprecationWarning)
                mpl.rcParams.update(**curr_rcParams)

    return plot_wrapper

# (1)
# `flush_figures` is a post-cell execution callback that basically runs
# `plt.show(); plt.close('all')`. There's a weird matplotlib/IPython
# interaction bug where:
#   - matplotlib.pyplot uses IPython.core.pylabtools to register
#     `flush_figures` when imported into an ipython env
#   - The `%matplotlib inline` magic command also registers
#     `flush_figures` the first time it's run
#   - IPython runs `%matplotlib inline` if it detects matplotlib.pyplot
#     has been imported and no backend is set in the same cell
#
# Switching to the interactive notebook backend (plt.switch_backend('nbAgg')
# or `%matplotlib notebook`) unregisters *one* `flush_figures` callback, but
# leaves the other(s), so the interactive figure is closed as soon as it's
# rendered and the event loop throws an error.


# (2)
# if we just created an interactive plot in a notebook and `main_backend`
# isn't the current backend, then we're reverting to 'inline'. This will
# kill any currently running animations, so we can't do it as part of the
# function call OR part of the current cell, otherwise the figure will
# close immediately. Instead, we can register an IPython callback
# function that runs *before* execution of the next cell, resets the
# backend, and unregisters itself all at once.
