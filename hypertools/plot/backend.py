import functools
import inspect
import sys
import warnings
from os import getenv

import matplotlib as mpl
import matplotlib.pyplot as plt


HYPERTOOLS_BACKEND = None
IS_NOTEBOOK = False
IPYTHON_INSTANCE = None
BACKEND_WARNING = None

switch_backend = None
reset_backend = None


def set_backend():
    """
    Runs when hypertools is initially imported and sets the matplotlib
    backend used for animated/interactive plots.

    Returns
    -------
    None

    Notes
    -----
    Sets the following module-scoped variables:

    HYPERTOOLS_BACKEND : str
        The matplotlib backend used for interactive or animated
        plots.
    IS_NOTEBOOK : bool
        True if hypertools was imported into a Jupyter notebook.
        Otherwise, False.
    IPYTHON_INSTANCE : ipykernel.zmqshell.ZMQInteractiveShell or None
        The IPython InteractiveShell instance for the current
        IPython kernel, if any.  Otherwise, None.
    BACKEND_WARNING : str or None
        The warning to be issued upon trying to create an
        interactive or animated plot, if any.  Otherwise, None.  This is
        set under two conditions:
            1. No compatible interactive backends are available
            2. Hypertools was imported into a notebook and the
               notebook-native interactive backend (nbAgg) is not
               available.  Tthis should never happen, but theoretically
               could if the ipython/jupyter/jupyter-core/notebook
               installation is faulty.
    """
    global HYPERTOOLS_BACKEND, \
        IS_NOTEBOOK, \
        IPYTHON_INSTANCE, \
        BACKEND_WARNING, \
        switch_backend, \
        reset_backend


    curr_backend = mpl.get_backend()

    try:
        # function exists in namespace if hypertools was imported from IPython shell or Jupyter notebook
        IPYTHON_INSTANCE = get_ipython()
        assert 'IPKernelApp' in IPYTHON_INSTANCE.config
        IS_NOTEBOOK = True
        # if running in a notebook, should almost always use nbAgg. May
        # eventually let user override this with environment variable
        # (e.g., to use ipympl, widget, or WXAgg in JupyterLab), but
        # switching backends in IPython is already a nightmare
        try:
            mpl.use('nbAgg')
            working_backend = 'nbAgg'
        except ImportError:
            BACKEND_WARNING = "Failed to switch to interactive notebook " \
                              "backend ('nbAgg'). Falling back to inline " \
                              "static plots."
            working_backend = 'inline'

        switch_backend = _switch_backend_notebook
        reset_backend = _defer_notebook_backend_reset

    except (NameError, AssertionError):
        # NameError: imported from script
        # AssertionError: imported from IPython shell
        IS_NOTEBOOK = False
        # excluding WebAgg - no way to test in advance if it will work
        backends = ('TkAgg', 'Qt5Agg', 'Qt4Agg', 'WXAgg', 'GTK3Agg')
        if sys.platform == 'darwin':
            # prefer cocoa backend on Mac - pretty much guaranteed to
            # work, appears to be faster, and Mac does NOT like Tkinter
            backends = ('MacOSX', *backends)

        # TODO: document setting environment variable
        # check for configurable environment variable
        env_backend = getenv("HYPERTOOLS_BACKEND")
        if env_backend is not None:
            # prefer user-specified backend, if set
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
            BACKEND_WARNING = "Failed to switch to any interactive backend " \
                              f"({', '.join(backends)}. Falling back to 'Agg'."
            working_backend = 'Agg'

        if env_backend is not None and working_backend.lower() != env_backend.lower():
            # The only time a plotting-related warning should be issued
            # on import rather than on call to hypertools.plot is if
            # $HYPERTOOLS_BACKEND specifies an incompatible backend,
            # since that will have been set manually.
            warnings.warn("failed to set matplotlib backend to backend "
                          f"specified in environment ('{env_backend}'). "
                          f"Falling back to '{working_backend}'")

        switch_backend = plt.switch_backend
        reset_backend = plt.switch_backend

    finally:
        # restore backend
        mpl.use(curr_backend)
        HYPERTOOLS_BACKEND = working_backend


def _switch_backend_notebook(backend):
    # have to import this here since ipykernel is only guaranteed to
    # be installed if running in notebook
    from ipykernel.pylab.backend_inline import flush_figures
    IPYTHON_INSTANCE.run_line_magic('matplotlib', backend)
    # see (1) below re: unregistering `flush_figures` callback
    while flush_figures in IPYTHON_INSTANCE.events.callbacks['post_execute']:
        IPYTHON_INSTANCE.events.unregister('post_execute', flush_figures)


def _defer_notebook_backend_reset(backend):
    # see (2) below re: deferring resetting the backend in notebooks
    def _callback():
        _callback.ipython_instance.run_line_magic('matplotlib', _callback.backend)
        _callback.ipython_instance.events.unregister('pre_run_cell', _callback)

    if backend == 'module://ipykernel.pylab.backend_inline':
        backend = 'inline'

    _callback.backend = backend
    _callback.ipython_instance = IPYTHON_INSTANCE
    IPYTHON_INSTANCE.events.register('pre_run_cell', _callback)


def requires_backend_change(curr_backend, plot_func, *func_args, **func_kwargs):
    func_signature = inspect.signature(plot_func)
    bound_args = func_signature.bind(*func_args, **func_kwargs)
    bound_args.apply_defaults()
    all_kwargs = bound_args.arguments
    if all_kwargs.get('animate') or all_kwargs.get('interactive'):
        plot_backend = all_kwargs.get('mpl_backend')
        if plot_backend == 'auto':
            plot_backend = HYPERTOOLS_BACKEND

        if plot_backend.lower() not in ('disable', curr_backend.lower()):
            return True

    return False


def manage_backend(plot_func):
    """
    Decorator for hypertools.plot that prevents unexpected changes to
    matplotlib rcParams (https://github.com/ContextLab/hypertools/issues/243)
    and handles temporarily changing the matplotlib backend for
    interactive and animated plots.
    Parameters
    ----------
    plot_func

    Returns
    -------

    """
    @functools.wraps(plot_func)
    def plot_wrapper(*args, **kwargs):
        # record current rcParams
        curr_rcParams = mpl.rcParams.copy()
        backend_changed = False
        try:
            curr_backend = mpl.get_backend()

            if requires_backend_change(curr_backend, plot_func, *args, **kwargs):
                if BACKEND_WARNING is not None:
                    warnings.warn(BACKEND_WARNING)

                switch_backend()
                backend_changed = True

            return plot_func(*args, **kwargs)

        finally:
            if backend_changed:
                reset_backend(curr_backend)

            with warnings.catch_warnings():
                # if the matplotlibrc was cached from <=v3.3.0, there's a TON
                # of harmless (as of v3.2.0) MatplotlibDeprecationWarnings
                # about rcParams fields related to axes.Axes3D objects
                warnings.simplefilter('ignore', mpl.MatplotlibDeprecationWarning)
                # restore rcParams prior to plot
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
