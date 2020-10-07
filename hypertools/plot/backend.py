import sys
from contextlib import contextmanager
from functools import wraps
from os import getenv
from warnings import warn

import matplotlib as mpl
import matplotlib.pyplot as plt


HYPERTOOLS_BACKEND = None
IS_NOTEBOOK = False
BACKEND_WARNING = None


def set_backend():
    global HYPERTOOLS_BACKEND
    global IS_NOTEBOOK
    global BACKEND_WARNING

    curr_backend = mpl.get_backend()
    try:
        # passes if imported from Jupyter notebook
        assert 'IPKernelApp' in get_ipython().config
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
            backends = ('macosx', *backends)

        # check environment variable
        env_backend = getenv("HYPERTOOLS_BACKEND")
        if env_backend is not None:
            if env_backend in backends:
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

        if env_backend is not None and working_backend != env_backend:
            # The only time a warning is issued immediately on import is if
            # $HYPERTOOLS_BACKEND env var specifies an incompatible backend,
            # since that will have been set manually.
            warn("failed to set matplotlib backend to backend specified in "
                 f"environment ('{env_backend}'). Falling back to "
                 f"'{working_backend}'")

    finally:
        # restore backend
        mpl.use(curr_backend)
        HYPERTOOLS_BACKEND = working_backend