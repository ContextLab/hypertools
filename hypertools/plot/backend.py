"""
Module that deals with managing the matplotlib backend for interactive
and/or animated plots created via `hypertools.plot` and
`hypertools.DataGeometry.plot`.  Main functionality is contained in
`set_interactive_backend` (sole front-end function) and `manage_backend`
(decorator for `hypertools.plot`).

======================= MODULE-SCOPED VARIABLES ========================
Various information about the current state of the plotting backend is
managed by a set of semi-global (module-scoped) variables. While this
probably isn't an ideal setup long-term, it solves a bunch of problems
that would otherwise require either completely overhauling the plotting
API, recomputing the same values for every call, or doing a bunch of
hacky, computationally expensive object inspection. And while this
approach isn't thread-safe, neither is `matplotlib` itself [1], so this
isn't really a limiting problem and therefore probably okay.

BACKEND_MAPPING : `hypertools.plot.backend.BackendMapping`
    see `BackendMapping` docstring
BACKEND_KEYS : dict
    Maps between compatible `matplotlib` backend keys in a standard
    Python environment and their corresponding keys in an IPython
    environment (format: `{python_key: ipython_key(s)}`). In cases where
    multiple IPython backend keys denote the same Python-equivalent,
    the "preferred" key (the one `BACKEND_MAPPING` will funnel others
    into) is the first in the list.
BACKEND_WARNING : str or None
    The warning to be issued upon trying to create an interactive or
    animated plot, if any. This is set under two conditions:
      1. No compatible interactive backends are available
      2. Hypertools was imported into a notebook and the notebook-native
         interactive backend (nbAgg) is not available. This should never
         happen, but theoretically could if the
         `ipython`/`jupyter`/`jupyter-core`/`notebook` installation is
         faulty.
HYPERTOOLS_BACKEND : str
    The `matplotlib` backend used to create interactive and animated
    plots.
IN_SET_CONTEXT : bool
    A switch read by the `manage_backed` decorator to determine whether
    or not the wrapped call to `hypertools.plot` was made inside a
    `set_interactive_backend` context block.
IPYTHON_INSTANCE : `ipykernel.zmqshell.ZMQInteractiveShell` or None
    The IPython InteractiveShell instance for the current
    IPython kernel, if `hypertools` was imported into a Jupyter
    notebook. Otherwise, None. Used to register/unregister IPython
    callback functions and run magic commands.
IS_NOTEBOOK : bool
    Whether or not `hypertools` was imported into a Jupyter notebook.
reset_backend : function
    The function called to switch back to the original `matplotlib`
    plotting backend after creating an interactive/animated plot.
    `_reset_backend_notebook` if imported into a Jupyter notebook,
    otherwise `matplotlib.pyplot.switch_backend`.
switch_backend : function
    The function called to switch to the temporary backend prior to
    plotting. `_switch_notebook_backend` if running in a Jupyter
    notebook, otherwise `matplotlib.pyplot.switch_backend`.

========================================================================

FUTURE: `matplotlib` project leader says `nbagg` backend will be retired
"in the next year or two" in favor of the `ipympl` backend [2]. For the
Hypertools 2.0 revamp, the two options should be given equal priority in
order to support the various possible combinations of new and older
`IPython`/`ipykernel`/`notebook` versions going forward

[1] https://matplotlib.org/faq/howto_faq.html#working-with-threads
[2] https://github.com/ipython/ipython/issues/12190#issuecomment-599154335.
"""


import inspect
import sys
import traceback
import warnings
from contextlib import contextmanager, redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

from .._shared.exceptions import HypertoolsBackendError


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
BACKEND_WARNING = None
HYPERTOOLS_BACKEND = None
IN_SET_CONTEXT = False
IPYTHON_INSTANCE = None
IS_NOTEBOOK = None
reset_backend = None
switch_backend = None


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
    Primarily used by `as_python()` and `as_ipython()` methods of
    `HypertoolsBackend`.  Funnels multiple equivalent keys within the
    same interpreter (Python vs. IPython) to a "default", then maps
    between that and the analog from the other interpreter type. At
    either step, a key with no corresponding value returns the key (see
    `ParrotDict` docstring for more info).
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
    The issue is that these objects get passed to a ton of different
    low-level `matplotlib`/`ipython`/`ipykernel`/etc. functions that
    expect to be receiving strings, and subclasses of `UserString` fail
    type-checks for **actual** strings (i.e.,
    `isinstance(UserStringSubclass('a'), str)` returns False) while
    this approach doesn't. Since it's impossible to trace through every
    possible function these could be forwarded to in every possible
    scenario (and any of those functions could be changed at any time),
    this will hopefully be more stable long-term.
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
        # `get_ipython()` function exists in the namespace if
        # `hypertools` was imported from an IPython shell or Jupyter
        # notebook
        IPYTHON_INSTANCE = get_ipython()
        assert 'IPKernelApp' in IPYTHON_INSTANCE.config

    # NameError: raised if imported from a script
    # AssertionError: raised if imported from an IPython shell
    except (NameError, AssertionError):
        # see `_block_greedy_completer_execution()` docstring
        _block_greedy_completer_execution()

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
                               f"({', '.join(backends)}). Falling back to 'Agg'.")
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
        return
    else:
        for module in ('hypertools.plot', 'hypertools.plot.backend', 'numpy'):
            try:
                sys.modules.pop(module)
            except KeyError:
                pass

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
    tmp_stdout.close()
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
    animated/interactive plot in a Jupyter notebook. This needs to be
    done in a slightly roundabout way to handle various IPython/Jupyter
    notebook behaviors (see Notes below for details).

    Parameters
    ----------
    backend : str
        the matplotlib backend prior to running `hypertools.plot`

    Notes
    -----
    1. Changing the matplotlib backend in a Jupyter notebook immediately
       closes any open figures (killing any animation and/or
       interactivity). So the reset can't happen as part of the main
       `hypertools.plot` or even in the same cell, otherwise the plots
       would be closed as soon as they're rendered. To get around this,
       `_reset_backend_notebook` registers an IPython callback function
       (`_deferred_reset_cb()`) that runs *before* the code in the
       *next* cell is run and resets the backend. This way, the
       animation runs and the plot is interactive until the user runs
       the next cell, but the backend is still reset before any other
       code is executed. (Note: this does not require cells to be run in
       order or for the next cell to be pre-queued)
    2. The command to switch the `matplotlib` backend closes open
       figures when called, even if "switching" to the currently set
       backend. Normally, registered callbacks run for all future cells,
       which means the `_deferred_reset_cb()` callback would interfere
       with future animated plots and plots created across multiple
       cells. To prevent this, `_deferred_reset_cb()` unregisters itself
       after resetting the backend so it only exists for the first cell
       run after creating the plot. This also prevents polluting the
       list of registered callbacks with duplicates, which can slow down
       cell execution.
    3. IPython callbacks are required to have the same function
       signature as the prototype for the event that triggers them, and
       the `pre_run_cell` prototype takes no arguments. This means the
       callback to reset the backend has to be defined and registered
       inside a wrapper function, because:
         - The top-level function to reset the backend needs to have the
           same signature as `matplotlib.pyplot.switch_backend` (which
           takes the backend as an argument) so the two can be used
           interchangeably depending on the current interpreter.
         - The registered callback needs to reference the backend it's
           switching to, but can't take it as a parameter
         - The callback can't be wrapped with the backend it's switching
           to via `functools.partial`, because it also needs to
           reference its own function object in order to unregister
           itself and a partial function would be a separate object.
    4. The `_deferred_reset_cb()` callback is registered up to one time
       per cell (i.e., creating multiple interactive/animated plots in
       the same cell without the `set_interactive_backend` context
       manager doesn't lead to duplicate callbacks). To do this, the
       list of registered callbacks has to be checked by name rather
       than by object, since the inner `_deferred_reset_cb` function is
       re-defined as a different object each time.
    """
    def _deferred_reset_cb():
        _switch_backend_notebook(backend)
        IPYTHON_INSTANCE.events.unregister('pre_run_cell', _deferred_reset_cb)

    def _reset_cb_registered():
        for func in IPYTHON_INSTANCE.events.callbacks['pre_run_cell']:
            if func.__name__ == '_deferred_reset_cb':
                return True
        return False

    backend = backend.as_ipython()
    if not _reset_cb_registered():
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
    """
    Manually set the `matplotlib` backend used for generating
    interactive plots.

    Whereas `hypertools.plot`'s `mpl_backend` keyword argument can be
    used to specify the backend for a single plot,
    `hypertools.set_interactive_backend` is useful for doing so for
    multiple (or all) interactive plots at once, and can be in two
    different ways:

    1. directly, to change the backend for all subsequent interactive
       plots
          ```
          import hypertools as hyp
          geo = hyp.load('weights_avg')

          geo.plot(interactive=True)          # uses the default backend

          hyp.set_interactive_backend('TkAgg')
          geo.plot(interactive=True)          # uses the TkInter backend
          geo.plot(animate=True)              # uses the TkInter backend
          ```

    2. as a context manager with the `with` statement, to temporarily
       change the backend
          ```
          import hypertools as hyp
          geo = hyp.load('weights_avg')

          geo.plot(interactive=True)          # uses the default backend

          with hyp.set_interactive_backend('TkAgg'):
              geo.plot(interactive=True)      # uses the TkInter backend

          geo.plot(animate=True)              # uses the default backend
          ```

    Parameters
    ----------
    backend : str
        The `matplotlib` backend to use for interactive plots, either
        temporarily (when used as a context manager with `with`) or for
        the life of the interpreter (when called as a function)

    Notes
    -----
    1. `set_interactive_backend` is technically a class, but it
       shouldn't typically be used as one and is only designed this way
       to enable it to work as both a regular function and a context
       manager.
    2. Calling this directly does *not* immediately change the plotting
       backend; it changes the backend `hypertools` will use to create
       interactive plots going forward.
    3. However, when used as a context manager, the backend passed to
       `hypertools.set_interactive_backend` will be used for *all* plots
       created inside the context block, regardless of whether:
         - they are interactive/animated or static
         - the `mpl_backend` keyword argument is passed to
           `hypertools.plot`
         - they were created with `hypertools`, `matplotlib`, or a
           different `matplotlib`-based library (e.g., `seaborn`,
           `quail`, `umap-learn`)
       There are a few reasons for this behavior:
         - being able to skip inspecting the arguments passed to each
           `hypertools.plot` call means almost no overhead is added for
           calls after the first, and makes wrapping multiple calls much
           more efficient
         - the plotting backend is an attribute of `matplotlib` itself
           and `matplotlib` doesn't support running multiple backends
           simultaneously in the same namespace, so it's impossible to
           avoid it affecting other `matplotlib`-based plotting libraries
         - it's reasonable to assume this was the desired outcome when
           multiple plots are generated inside a context block, since A)
           the context block will always have been created manually by
           the user, and B) the API provides multiple other ways to set
           the backend without this effect
    3. The `manage_backend` decorator for `hypertools.plot` determines
       whether it's being called inside the
       `hypertools.set_interactive_backend` context manager by checking
       the value of a global variable (`IN_SET_CONTEXT`), which is
       switched to `True` when the the runtime context is entered and
       `False` when it's exited. This definitely isn't an ideal setup
       and could probably be refactored out in the v2.0 overhaul, but
       for now the alternatives are A) using something like
       `inspect.getframeinfo` or `traceback.extract_stack` to look for
       the context manager every time `hypertools.plot` is called, or B)
       re-running the same runtime argument checks every time, either of
       which would be much less efficient. So for now, the current setup
       is probably good enough.
    """
    def __init__(self, backend):
        global BACKEND_WARNING, HYPERTOOLS_BACKEND

        self.old_interactive_backend = HYPERTOOLS_BACKEND.normalize()
        self.old_backend_warning = BACKEND_WARNING
        self.new_interactive_backend = HypertoolsBackend(backend).normalize()
        self.new_is_different = self.new_interactive_backend != self.old_interactive_backend
        self.backend_switched = False

        if self.new_is_different:
            HYPERTOOLS_BACKEND = self.new_interactive_backend
            BACKEND_WARNING = None

    def __enter__(self):
        global IN_SET_CONTEXT

        IN_SET_CONTEXT = True
        self.curr_backend = HypertoolsBackend(mpl.get_backend()).normalize()
        if self.curr_backend != self.new_interactive_backend:
            # set this before calling switch_backend to make sure
            # `self.__exit__()` cleans up after any unexpected errors
            # while switching
            self.backend_switched = True
            switch_backend(self.new_interactive_backend)

    def __exit__(self, exc_type, exc_value, traceback):
        global BACKEND_WARNING, HYPERTOOLS_BACKEND, IN_SET_CONTEXT

        IN_SET_CONTEXT = False
        if self.new_is_different:
            HYPERTOOLS_BACKEND = self.old_interactive_backend
            BACKEND_WARNING = self.old_backend_warning

        if self.backend_switched:
            reset_backend(self.curr_backend)


@contextmanager
def _null_backend_context(dummy_backend):
    """
    A dummy context manager that does nothing, equivalent to
    `contextlib.nullcontext` (which isn't implemented in Python<3.7).
    Used in `manage_backend` when the decorated call to `hypertools.plot`
    happens inside the `hypertools.set_interactive_backend` context.

    Parameters
    ----------
    dummy_backend : object
        Arbitrary value that is never used, but is required to make this
        function syntactically match `hypertools.set_interactive_backend`
    """
    yield


def manage_backend(plot_func):
    """
    Decorator for hypertools.plot that prevents unexpected changes to
    matplotlib rcParams (https://github.com/ContextLab/hypertools/issues/243)
    and handles temporarily changing the matplotlib backend for
    interactive and animated plots, as necessary.

    Parameters
    ----------
    plot_func : function
        Function around which to set/reset the plotting backend and
        rcParams (currently, always `hypertools.plot`).

    Returns
    -------
    plot_wrapper : function
        The decorated function.

    Notes
    ------
    1. Capturing & restoring the rcParams needs to happen here rather
       than in `set_interactive_backend` so it's done independently for
       each plot
    2. Written in a slightly roundabout way in order to skip unnecessary
       & duplicate calls when the decorated call to `hypertools.plot`
       happens inside the `hypertools.set_interactive_backend` context.
    """
    @wraps(plot_func)
    def plot_wrapper(*args, **kwargs):
        # record current rcParams
        old_rcParams = mpl.rcParams.copy()
        # assume using the mock-`contextlib.nullcontext` context
        backend_context = _null_backend_context
        tmp_backend = None

        if not IN_SET_CONTEXT:
            plot_kwargs = _get_runtime_args(plot_func, *args, **kwargs)
            if plot_kwargs.get('animate') or plot_kwargs.get('interactive'):
                curr_backend = HypertoolsBackend(mpl.get_backend()).normalize()
                tmp_backend = plot_kwargs.get('mpl_backend')
                if tmp_backend == 'auto':
                    tmp_backend = HYPERTOOLS_BACKEND.normalize()

                if tmp_backend not in ('disable', curr_backend):
                    # if all conditions are met, use the real context
                    backend_context = set_interactive_backend

        try:
            with backend_context(tmp_backend):
                if BACKEND_WARNING is not None:
                    warnings.warn(BACKEND_WARNING)

                return plot_func(*args, **kwargs)

        finally:
            # restore rcParams prior to plot
            with warnings.catch_warnings():
                # if the matplotlibrc was cached from <=v3.3.0, a TON of
                # (harmless as of v3.2.0) MatplotlibDeprecationWarnings
                # about `axes.Axes3D`-related rcParams fields are issued
                warnings.simplefilter('ignore', mpl.MatplotlibDeprecationWarning)
                mpl.rcParams.update(**old_rcParams)

    return plot_wrapper
