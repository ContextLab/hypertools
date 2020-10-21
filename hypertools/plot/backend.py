"""
Module that deals with managing the matplotlib backend for interactive
and/or animated plots created via `hypertools.plot` and
`hypertools.DataGeometry.plot`.  Main functionality is contained in
`set_interactive_backend` (front-end function) and `manage_backend`
(decorator/context manager for `hypertools.plot`).

Note that the interactive plotting backend is currently managed via a
module-scoped variable, and therefore this functionality isn't
thread-safe. However, since matplotlib itself isn't thread-safe either
(see https://matplotlib.org/faq/howto_faq.html#working-with-threads),
this isn't really a limiting problem and therefore probably okay.
"""

# FUTURE: matplotlib project leader says `nbagg` backend will be
#  retired "in the next year or two" in favor of the `ipympl` backend:
#  https://github.com/ipython/ipython/issues/12190#issuecomment-599154335.
#  For the Hypertools 2.0 revamp, we'll want to put the two options on
#  equal footing in order to support the various possible combinations
#  of new and older IPython/ipykernel/notebook versions going forward


import inspect
import sys
import traceback
import warnings
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

from .._shared.exceptions import HypertoolsBackendError


# ============================== GLOBALS ===============================
BACKEND_WARNING = None
HYPERTOOLS_BACKEND = None
IPYTHON_INSTANCE = None
IS_NOTEBOOK = None
reset_backend = None
switch_backend = None
BACKEND_KEYS = {
    'TkAgg': 'tk',
    'GTK3Agg': ['gtk3', 'gtk'],
    'WXAgg': 'wx',
    'Qt4Agg': 'qt4',
    'Qt5Agg': ['qt5', 'qt'],
    'MacOSX': 'osx',
    'nbAgg': ['notebook', 'nbagg'],
    'module://ipykernel.pylab.backend_inline': 'inline',
    'module://ipympl.backend_nbagg': ['ipympl', 'widget']
}
BACKEND_MAPPING = None
# ======================================================================


class ParrotDict(dict):
    """
    Dictionary subclass with a few changes in behavior:
      1. all keys and values are stored and indexed as
         `HypertoolsBackend` instances
      2. indexing a `ParrotDict` with a key that doesn't exist returns
         the key (it's "parroted" back to you). The key is converted to
         a `HypertoolsBackend` instance if it is not already. Similar
         to `collections.defaultdict`, but does *not* add missing keys
         on indexing.

    Useful for filtering/replacing some values while leaving others
    when the to-be-replaced values are known in advance.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __contains__(self, key):
        return HypertoolsBackend(key) in self.keys()

    def __getitem__(self, key):
        key = HypertoolsBackend(key)
        return super().__getitem__(key)

    def __missing__(self, key):
        return HypertoolsBackend(key)

    def __setitem__(self, key, value):
        key, value = HypertoolsBackend(key), HypertoolsBackend(value)
        return super().__setitem__(key, value)


class BackendMapping:
    """
    A two-way, non-unique dict-like mapping between keys used to set
    the matplotlib plotting backend in Python and IPython environments.
    Primarily used by `as_python` and `as_ipython` methods of
    `HypertoolsBackend`.  Funnels multiple equivalent keys within the
    same interpreter (Python vs. IPython) to a "default", then maps
    between that and the analog from the other interpreter type. At
    either step, a key with no corresponding value returns the key (see
    `OneWayMapping` for more info).
    """
    def __init__(self, _dict):
        # assumes format of _dict is {Python: IPython}
        self.py_to_ipy = ParrotDict()
        self.ipy_to_py = ParrotDict()
        self.equivalents = ParrotDict()

        for py_key, ipy_key in _dict.items():
            py_key_default = self._store_equivalents(py_key)
            ipy_key_default = self._store_equivalents(ipy_key)
            self.py_to_ipy[py_key_default] = ipy_key_default
            self.ipy_to_py[ipy_key_default] = py_key_default

    def _store_equivalents(self, keylist):
        if not isinstance(keylist, str) and isinstance(keylist, Iterable):
            default_key = keylist[0]
            for key_equiv in keylist[1:]:
                key_equiv = key_equiv
                self.equivalents[key_equiv] = default_key
        else:
            default_key = keylist
        return default_key


class HypertoolsBackend(str):
    """
    A subclass of the `str` built-in, intended for easy(ish...)
    conversion between the different valid matplotlib backend keys in
    Python vs IPython and equality/membership checks.

    Notes
    -----
    Normally, a lot of this could be simplified and a lot of grief saved
    by subclassing `collections.UserString` rather than `str` directly.
    The issue is that we're passing these to a ton of different
    low-level `matplotlib`/`ipython`/`ipykernel`/etc. functions in
    which they basically need to masquerade as strings. It's way too
    complex to try to trace through where they're passed as-is, plus
    any of these functions may be changed at any time, and subclasses
    of `UserString` fail type-checks for **actual** strings (i.e.,
    `isinstance(UserStringSubclass('a'), str)` returns False) whereas
    this approach doesn't. So this will (hopefully) be more stable
    long-term.
    """
    def __new__(cls, x):
        return super().__new__(cls, x)

    def __eq__(self, other):
        """
        case-insensitive comparison with both `str`s and other
        `HypertoolsBackend` instances
        """
        return str(self).casefold() == str(other).casefold()

    def __getattribute__(self, name):
        """
        Overrides `str.__getattribute__` in a way that causes all
        inherited `str` methods to return a `HypertoolsBackend`
        instance, rather than a `str`, which it would otherwise do.
        See class docstring for more information.

        Parameters
        ----------
        name : the attribute or method accessed

        Returns
        -------
        val : instance (or collection of instances) of
            `hypertools.plot.backend.HypertoolsBackend`. For inherited
            `str` attributes and methods, the return type is the same,
            but with all instances of `str` replaced with
            `hypertools.plot.backend.HypertoolsBackend`.
        """
        # only deal with string attributes/methods here
        if hasattr(str, name):
            def _subclass_method(self, *args, **kwargs):
                value = getattr(super(), name)(*args, **kwargs)
                if isinstance(value, str):
                    return HypertoolsBackend(value)
                elif isinstance(value, (list, tuple, set)):
                    return type(value)(HypertoolsBackend(v) for v in value)
                else:
                    return value

            # bind inner function to instance
            return _subclass_method.__get__(self)
        else:
            return super().__getattribute__(name)

    def __hash__(self):
        """
        needed to work for membership checks/lookups in dict/set/etc.
        """
        return str.__hash__(str(self).casefold())

    def as_ipython(self):
        """
        Return the IPython-compatible version of a matplotlib backend
        key, given either the Python-compatible or IPython-compatible
        version
        """
        default_key = BACKEND_MAPPING.equivalents[self]
        return HypertoolsBackend(BACKEND_MAPPING.py_to_ipy[default_key])

    def as_python(self):
        """
        Return the Python-compatible version of a matplotlib backend
        key, given either the Python-compatible or IPython-compatible
        version
        """
        default_key = BACKEND_MAPPING.equivalents[self]
        return HypertoolsBackend(BACKEND_MAPPING.ipy_to_py[default_key])

    def normalize(self):
        """
        Convert a given matplotlib backend key to its preferred
        equivalent for the correct interpreter
        """
        return self.as_ipython() if IS_NOTEBOOK else self.as_python()



def _init_backend():
    """
    Runs when hypertools is initially imported and sets the matplotlib
    backend used for animated/interactive plots.

    Notes
    -----
    Sets the following module-scoped variables:

    HYPERTOOLS_BACKEND : str
        The matplotlib backend used for interactive or animated
        plots.
    IPYTHON_INSTANCE : ipykernel.zmqshell.ZMQInteractiveShell or None
        The IPython InteractiveShell instance for the current
        IPython kernel, if any.  Otherwise, None.
    IS_NOTEBOOK : bool
        Whether or not hypertools is being run in a Jupyter notebook
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
    switch_backend : function
        The function called to switch to the temporary backend prior to
        plotting. `_switch_notebook_backend` if running in a Jupyter
        notebook, otherwise, `matplotlib.pyplot.switch_backend`.
    reset_backend : function
        The function called to switch back to the original backend after
        plotting. `_reset_backend_notebook` if running in a Jupyter
        notebook, otherwise `matplotlib.pyplot.switch_backend`.
    """
    global BACKEND_MAPPING, \
        BACKEND_WARNING, \
        HYPERTOOLS_BACKEND, \
        IPYTHON_INSTANCE, \
        IS_NOTEBOOK, \
        reset_backend, \
        switch_backend

    curr_backend = mpl.get_backend()

    try:
        # function exists in namespace if hypertools was imported from
        # IPython shell or Jupyter notebook
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
            BACKEND_WARNING = ("Failed to switch to interactive notebook "
                               "backend ('nbAgg'). Falling back to inline "
                               "static plots.")
            working_backend = 'inline'

        switch_backend = _switch_backend_notebook
        reset_backend = _reset_backend_notebook

    # NameError: imported from script
    # AssertionError: imported from IPython shell
    except (NameError, AssertionError):
        # Edge case: NameError is raised in a Jupyter notebook because
        # IPCompleter/Jedi greedy TAB-completion is enabled and used on
        # hypertools module names before hypertools is imported.
        block_greedy_completer_execution()
        IS_NOTEBOOK = False
        # (excluding WebAgg - no way to test in advance if it will work)
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

            except (ImportError, NameError):
                # raised if backend's dependencies aren't installed
                continue

        else:
            BACKEND_WARNING = ("Failed to switch to any interactive backend "
                               f"({', '.join(backends)}. Falling back to 'Agg'.")
            working_backend = 'Agg'

        if env_backend is not None and working_backend.lower() != env_backend.lower():
            # The only time a plotting-related warning should be issued
            # on import rather than on call to hypertools.plot is if
            # $HYPERTOOLS_BACKEND specifies an incompatible backend,
            # since that will have been set manually.
            warnings.warn("failed to set matplotlib backend to backend "
                          f"specified in environment ('{env_backend}'). "
                          f"Falling back to '{working_backend}'")

        switch_backend = reset_backend = _switch_backend_regular

    else:
        IS_NOTEBOOK = True
        # if running in a notebook, should almost always use nbAgg. May
        # eventually let user override this with environment variable
        # (e.g., to use ipympl, widget, or WXAgg in JupyterLab), but for
        # now this can be changed manually with
        # `hypertools.set_interactive_backend` or the `mpl_backend`
        # kwarg to `hypertools.plot`
        try:
            mpl.use('nbAgg')
            working_backend = 'nbAgg'
        except ImportError:
            BACKEND_WARNING = ("Failed to switch to interactive notebook "
                               "backend ('nbAgg'). Falling back to inline "
                               "static plots.")
            working_backend = 'inline'

        switch_backend = _switch_backend_notebook
        reset_backend = _reset_backend_notebook

    finally:
        # restore backend
        mpl.use(curr_backend)
        BACKEND_MAPPING = BackendMapping(BACKEND_KEYS)
        HYPERTOOLS_BACKEND = HypertoolsBackend(working_backend).normalize()


def _block_greedy_completer_execution():
    """
    Handles an annoying edge case in `init_backend()`:
      - IPython uses "greedy" TAB-completion, meaning code is actually
        executed in order to determine autocomplete suggestions
          + there is a config setting to disable this, but it's enabled
            by default
      - if TAB-completion is used in an import statement, the module is
        actually imported if it hasn't been previously [1], which means
        that for `hypertools`, `init_backend()` will be run
      - because the TAB-completion happens in a non-IPython subprocess,
        the backend will be initialized for non-notebook use.

    To correct this, the function:
      - looks through the stack trace for a call made from IPython's
        TAB-completion module (`IPython/core/completerlib.py`)
          + this is probably the safest way to do the search, since A)
            the module name is less likely to change than the function
            name or line number, and B) searching for *any* IPython
            module would break importing `hypertools` in an IPython shell
          + this is also probably the fastest way to search, since A) it
            short-circuits on the first call specific to this scenario,
            and B) the call stack will usually be at most a few imports
            deep in non-notebook environments
          + at minimum, the last 3 calls will always be from `hypertools`
            so they're skipped to save time
      - if it finds one, removes both `hypertools.plot` and
        `hypertools.plot.backend` (and also `numpy`) from `sys.modules`...
          + the `import` statement (`importlib.__import__()`) checks
            `sys.modules` for already-loaded modules before importing,
            so removing these causes them to be reloaded when the
            `import` command is actually run
          + both `hypertools.plot` and `hypertools.plot.backend` need to
            be removed to avoid using the cached call to `init_backend()`
          + `numpy` is also unloaded, otherwise its C extensions get
            confused when `hypertools` is re-imported and issue a whole
            slew of warnings
      - ...and raises a generic exception (handled in [1]), which skips
        running the rest of `init_backend()` while still allowing the
        TAB-completer to keep searching other `hypertools` modules
          + Also, since both of the removed modules will already have
            been seen by the completer at this point, they'll still be
            shown as autocomplete options despite not being in
            `sys.modules`.

    [1] https://github.com/ipython/ipython/blob/2b4bc75ac735a2541125b3baf299504e5513994a/IPython/core/completerlib.py#L158
    """
    stack_trace = traceback.extract_stack()[-4::-1]
    completer_module = 'IPython/core/completerlib.py'
    try:
        next(entry for entry in stack_trace if entry.filename.endswith(completer_module))
    except StopIteration:
        pass
    else:
        try:
            sys.modules.pop('hypertools.plot')
            sys.modules.pop('hypertools.plot.backend')
            sys.modules.pop('numpy')
        except KeyError:
            pass
        finally:
            raise Exception


def _switch_backend_regular(backend):
    """
    Switch the plotting backend via `matplotlib.pyplot.switch_backend()`.

    Used to set/reset the backend in non-notebook environments, and as a
    fallback method of doing so in notebook environments when the
    IPython magic command fails.

    Parameters
    ----------
    backend : str
        the matplotlib backend to switch to

    Raises
    ------
    HypertoolsBackendError
        if switching the backend fails

    """
    backend = backend.as_python()

    try:
        plt.switch_backend(backend)
    except Exception as e:
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            err_msg = (f"Failed to switch the plotting backend to "
                       f"{backend}. You may be missing required dependencies, "
                       "or this backend may not be available for your system")
        else:
            err_msg = ("An unexpected error occurred while trying to switch "
                       f"the plotting backend to {backend}")

        raise HypertoolsBackendError(err_msg) from e


def _switch_backend_notebook(backend):
    """
    Handles switching the matplotlib backend when running in a Jupyter
    notebook.

    Parameters
    ----------
    backend : str
        the interactive matplotlib backend to switch to

    Notes
    -----
    1. `flush_figures` is a post-cell execution callback that
       `plt.show()`s & `plt.close()`s all figures created in a cell so
       that later `plt.plot()` calls create new figures. There's a weird
       circular matplotlib/IPython interaction where:
         - `matplotlib.pyplot` (via `IPython.core.pylabtools`) registers
            `flush_figures` when it's imported into an IPython
            environment
         - The `%matplotlib inline` magic command also registers a
           `flush_figures` call each it's run, whether or not one has
           been registered already
         - IPython runs `%matplotlib inline` if it detects
           `matplotlib.pyplot` has been imported and no backend is set
           in the same cell.
       So depending on import order, whether imports happen across
       multiple cells, and whether/when/how many times the backend has
       been switched, there may be any number of `flush_figures`
       callbacks registered. Switching to the interactive notebook
       backend unregisters one `flush_figures` callback but leaves the
       other(s), and creating an interactive figure with `flush_figures`
       registered closes the figure immediately after the cell executes
       and causes the matplotlib event loop to throw an error. So we
       need to ensure all `flush_figures` instances are unregistered
       before plotting.
    2. In certain situations, the IPython magic command fails to switch
       the backend but the matplotlib command will succeed. And for some
       unfathomable reason, when the magic command fails, IPython
       **prints** a warning message rather than issuing it via
       `warnings.warn()` [1]. So the only way to catch this warning is
       to temporarily suppress and capture stdout.

    [1] https://github.com/ipython/ipython/blob/e394d65a6b499b5d91df7ca0306c1cb88c543f43/IPython/core/interactiveshell.py#L3495
    """
    # ipykernel is only guaranteed to be installed if running in notebook
    from ipykernel.pylab.backend_inline import flush_figures
    backend = backend.as_ipython()

    tmp_stdout = StringIO()
    exc = None

    with redirect_stdout(tmp_stdout):
        try:
            IPYTHON_INSTANCE.run_line_magic('matplotlib', backend)
        except KeyError as e:
            exc = e
            IPYTHON_INSTANCE.run_line_magic('matplotlib', '-l')

    output_msg = tmp_stdout.getvalue().strip()
    if exc is not None:
        # just in case something else was somehow sent to stdout while
        # redirected, or if we managed to catch a different KeyError
        backends_avail = output_msg.splitlines()[-1]
        raise ValueError(f"{backend} is not a valid IPython plotting "
                         f"backend.\n{backends_avail}") from exc

    elif output_msg.startswith('Warning: Cannot change to a different GUI toolkit'):
        try:
            _switch_backend_regular(backend)
        except HypertoolsBackendError as e:
            err_msg = (f'Failed to switch plotting backend to "{backend}" via '
                       f"IPython with the following message:\n\t{output_msg}\n\n"
                       f"Fell back to switching via matplotlib and failed with "
                       f"the above error")
            raise HypertoolsBackendError(err_msg) from e

    if backend != 'inline':
        while flush_figures in IPYTHON_INSTANCE.events.callbacks['post_execute']:
            IPYTHON_INSTANCE.events.unregister('post_execute', flush_figures)


def _reset_backend_notebook(backend):
    """
    Handles resetting the matplotlib backend after displaying an
    animated/interactive plot in a Jupyter notebook by registering a
    "one-shot" self-destructing pre_cell_run callback to the next cell

    Parameters
    ----------
    backend : str
        the matplotlib backend prior to running `hypertools.plot`

    Notes
    -----
    Changing the matplotlib backend in a Jupyter notebook immediately
    closes any open figures (meaning animation and interactivity stop),
    so we can't do it as part of the function call or register it as a
    post-execution hook for the current cell. We get around this by
    registering a callback function that runs *before* the *next* cell
    is run, resets the backend, and unregisters itself. This way, the
    animation runs and the plot is interactive until the user runs the
    next cell, but the backend is reset before any code is executed. And
    because the callback "self-destructs," we won't force later figures
    to close unnecessarily ("changing" the backend to the current backend
    still does this) and doing this multiple times won't pollute the
    callback list, which can slow down cell execution.

    We also have to define and register the callback inside a wrapper
    function, since we need to reference the backend we're switching to
    using the same syntax as `matplotlib.pyplot.switch_backend`. IPython
    callbacks have to have the same signature as the prototype for the
    event they're registered to, and the pre_run_cell prototype takes no
    arguments. And since the callback needs to reference the correct
    function object in order to unregister itself, we can't just wrap it
    in `functools.partial`.
    """
    def _deferred_reset_cb():
        _switch_backend_notebook(backend)
        IPYTHON_INSTANCE.events.unregister('pre_run_cell', _deferred_reset_cb)

    backend = backend.as_ipython()
    # need this check in case multiple interactive plots are created
    # within the same context block (`with set_interactive_backend...`)
    # or the same Jupyter notebook cell
    if _deferred_reset_cb not in IPYTHON_INSTANCE.events.callbacks['pre_run_cell']:
        IPYTHON_INSTANCE.events.register('pre_run_cell', _deferred_reset_cb)


def _get_runtime_args(func, *func_args, **func_kwargs):
    """
    Does some quick introspection to determine runtime values assigned
    to all parameters of a function for a given call, whether passed as
    args, kwargs, or defaults.

    Parameters
    ----------
    func : function
        The function to introspect
    func_args : tuple
        positional arguments passed to `func` at runtime
    func_kwargs : dict
        keyword arguments passed to `func` at runtime

    Returns
    -------
    runtime_vals : dict
        {parameter: value} mapping of runtime values
    """
    func_signature = inspect.signature(func)
    bound_args = func_signature.bind(*func_args, **func_kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


class set_interactive_backend:
    def __init__(self, backend):
        global HYPERTOOLS_BACKEND, BACKEND_WARNING

        self.old_backend = HYPERTOOLS_BACKEND.normalize() #str(HYPERTOOLS_BACKEND).casefold()
        self.old_backend_warning = BACKEND_WARNING
        self.new_backend = HypertoolsBackend(backend).normalize()
        self.new_is_different = self.new_backend != self.old_backend
        print(f'old is {self.old_backend}, new is {self.new_backend}, new_is_different is {self.new_is_different}')

        if self.new_is_different:
            HYPERTOOLS_BACKEND = self.new_backend
            print(f'and now HYPERTOOLS_BACKEND is {HYPERTOOLS_BACKEND}')
            BACKEND_WARNING = None

    def __enter__(self):
        if self.new_is_different:
            switch_backend(self.new_backend)

    def __exit__(self, *args):
        global HYPERTOOLS_BACKEND, BACKEND_WARNING
        if self.new_is_different:
            try:
                reset_backend(self.old_backend)
            except HypertoolsBackendError:
                raise
            else:
                HYPERTOOLS_BACKEND = self.old_backend
                BACKEND_WARNING = self.old_backend_warning


def manage_backend(plot_func):
    """
    Decorator for hypertools.plot that prevents unexpected changes to
    matplotlib rcParams (https://github.com/ContextLab/hypertools/issues/243)
    and handles temporarily changing the matplotlib backend for
    interactive and animated plots, as necessary.

    Parameters
    ----------
    plot_func : function
        Function around which to do setup and teardown. In this case,
        `hypertools.plot`.

    Returns
    -------
    plot_wrapper : function
        The decorated function.
    """
    @wraps(plot_func)
    def plot_wrapper(*args, **kwargs):
        # record current rcParams
        curr_rcParams = mpl.rcParams.copy()
        backend_switched = False
        try:
            curr_backend = HypertoolsBackend(mpl.get_backend()).normalize()
            plot_kwargs = _get_runtime_args(plot_func, *args, **kwargs)
            if plot_kwargs.get('animate') or plot_kwargs.get('interactive'):
                tmp_backend = plot_kwargs.get('mpl_backend')
                if tmp_backend == 'auto':
                    tmp_backend = HYPERTOOLS_BACKEND.normalize()

                if tmp_backend not in ('disable', curr_backend):
                    print(f'different -- tmp is {tmp_backend}, curr is {curr_backend}')
                    if BACKEND_WARNING is not None:
                        warnings.warn(BACKEND_WARNING)

                    switch_backend(tmp_backend)
                    backend_switched = True

            print('immediately pre-run\n', IPYTHON_INSTANCE.events.callbacks, '\n\n')
            return plot_func(*args, **kwargs)

        finally:
            if backend_switched:
                reset_backend(curr_backend)

            # restore rcParams prior to plot
            with warnings.catch_warnings():
                # if the matplotlibrc was cached from <=v3.3.0, a TON of
                # (harmless as of v3.2.0) MatplotlibDeprecationWarnings
                # about `axes.Axes3D`-related rcParams fields are issued
                warnings.simplefilter('ignore', mpl.MatplotlibDeprecationWarning)
                mpl.rcParams.update(**curr_rcParams)

    return plot_wrapper
