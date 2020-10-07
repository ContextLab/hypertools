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
                              "backend ('nbAgg'). Falling back to 'inline'."
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


def manage_backend(plot_func):
    @wraps(plot_func)
    def plot_wrapper(*args, **kwargs):
        global IPYTHON_INSTANCE
        try:
            # get current rcParams to restore after plot
            curr_rcParams = mpl.rcParams.copy()
            main_backend = mpl.get_backend()

            if main_backend.lower() != HYPERTOOLS_BACKEND.lower():
                print('main_backend != HYPERTOOLS_BACKEND')
                # some object inspection magic to get arg values passed
                func_signature = inspect.signature(plot_func)
                bound_args = func_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                all_argvalues = bound_args.arguments
                if all_argvalues.get('animate') or all_argvalues.get('interactive'):
                    print('animate is True or interactive is True')
                    if BACKEND_WARNING is not None:
                        warnings.warn(BACKEND_WARNING)
                    if IS_NOTEBOOK:
                        print('IS_NOTEBOOK')
                        from ipykernel.pylab.backend_inline import flush_figures
                        # see (1) below re: unregistering flush_figures callback
                        while flush_figures in IPYTHON_INSTANCE.events.callbacks['post_execute']:
                            IPYTHON_INSTANCE.events.unregister('post_execute', flush_figures)

                        print(IPYTHON_INSTANCE.events.callbacks['post_execute'])

                    print(f'switching backend from {main_backend} to {HYPERTOOLS_BACKEND}')
                    plt.switch_backend(HYPERTOOLS_BACKEND)

            return plot_func(*args, **kwargs)
        finally:
            if mpl.get_backend().lower() != main_backend.lower():
                print(f'switching backend from {HYPERTOOLS_BACKEND} to {main_backend}')
                plt.switch_backend(main_backend)
                if IS_NOTEBOOK and flush_figures not in IPYTHON_INSTANCE.events.callbacks['post_execute']:
                    IPYTHON_INSTANCE.events.register('post_execute', flush_figures)

            print('setting rcParams back')
            with warnings.catch_warnings():
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