"""Import optional dependencies, installing them on demand.

hypertools declares its optional extras ONCE, in ``pyproject.toml``. This
module reads that declaration back from the installed package metadata, so
the pip specs it installs are exactly the ones pyproject pins: there is no
second list of versions to drift. The only thing declared here is the map
from an IMPORT name to the extra that provides it, because an import name is
not a distribution name (``skimage`` is scikit-image, ``chronos`` is
chronos-forecasting, ``sentence_transformers`` arrives with
``pydata-wrangler[hf]``).

Policy: a missing optional module is installed into the running interpreter
(``python -m pip install <the extra's requirements>``) and then imported.
Nothing about hypertools itself is reinstalled, so a development or
branch install is never replaced by a PyPI release.
``hypertools.set_autoinstall(False)`` (the `set_autoinstall` class below,
also a context manager) turns installation off; the import then fails with
the manual command. The environment variable ``HYPERTOOLS_AUTO_INSTALL=0``
sets the starting value for processes where no Python runs first (an image
built ahead of time). Every install prints a one-line notice. The setting
is per interpreter, so a child process that runs hypertools code (the
plotly animation-export worker) is started with ``subprocess_env()``,
which carries the effective value over as that variable.

``ensure_kaleido_chrome()`` provisions what plotly's static image export
needs at run time: a Chrome build for kaleido and, on Linux images that
lack them (a fresh Colab or Kaggle kernel, measured 2026-09-04), the four
shared libraries that Chrome needs to start.
"""

import importlib
import os
import shutil
import subprocess
import sys
import threading
from importlib import metadata

#: import name -> the hypertools extra that provides it (the ONLY mapping
#: kept outside pyproject.toml; the requirement strings themselves are read
#: from the package metadata, see `extra_requirements`).
EXTRA_FOR_MODULE = {
    'plotly': 'interactive',
    'kaleido': 'interactive',
    'gensim': 'gensim',
    'torch': 'torch',
    'kagglehub': 'kaggle',
    'skimage': 'density3d',
    'chronos': 'predict-hf',
    'skaters': 'predict',
    'pylsl': 'lsl',
    'openpyxl': 'io',
    'sentence_transformers': 'text',
    'transformers': 'text',
    'datasets': 'text',
}

#: Debian/Ubuntu packages a downloaded Chrome needs and a fresh Colab/Kaggle
#: image lacks (ldd on the live runtime, 2026-09-04: libatk-1.0.so.0,
#: libatk-bridge-2.0.so.0, libatspi.so.0, libXcomposite.so.1).
CHROME_APT_PACKAGES = ('libatk1.0-0', 'libatk-bridge2.0-0', 'libatspi2.0-0',
                       'libxcomposite1')

#: ceiling for the apt step (a fresh Colab kernel measured ~30 s; 2026-09-04)
APT_TIMEOUT_SECONDS = 600

_kaleido_ready = False

#: every `set_autoinstall` object that is in force, oldest first: a direct
#: call stays until superseded, a `with` block removes ITS entry on exit, and
#: the newest entry still in force decides. Process-global (shared by every
#: thread), guarded by `_AUTO_INSTALL_LOCK`; empty means the environment
#: decides (`auto_install_enabled`).
_AUTO_INSTALL_SCOPES = []
_AUTO_INSTALL_LOCK = threading.Lock()


def auto_install_enabled():
    """True when hypertools may install a missing optional extra: what the
    newest `set_autoinstall` still in force set or, if none is, the environment
    variable ``HYPERTOOLS_AUTO_INSTALL`` (on unless it is 0/false/no/off)."""
    with _AUTO_INSTALL_LOCK:
        if _AUTO_INSTALL_SCOPES:
            return _AUTO_INSTALL_SCOPES[-1].enabled
    return os.environ.get('HYPERTOOLS_AUTO_INSTALL', '1').strip().lower() \
        not in ('0', 'false', 'no', 'off')


def subprocess_env(env=None):
    """The environment for a child Python process that runs hypertools code,
    carrying this process's EFFECTIVE auto-install setting.

    A `set_autoinstall` call lives in this interpreter only; a child started
    with `subprocess` begins from the environment variable. So the variable is
    set here from `auto_install_enabled` -- ``'1'`` or ``'0'`` -- and the child
    starts where the parent stands, whichever way the two disagreed
    (``set_autoinstall(True)`` over ``HYPERTOOLS_AUTO_INSTALL=0`` gives the
    child ``'1'``; ``set_autoinstall(False)`` with the variable unset gives it
    ``'0'``). Every subprocess launch of hypertools code must pass this as
    ``env=`` (release audit 2026-09-07: the plotly animation-export worker
    ran pip with installation switched off in the parent).

    Parameters
    ----------
    env : mapping, optional
        The environment to start from; defaults to ``os.environ``. Not
        modified: a copy is returned.

    Returns
    -------
    dict
        A copy of `env` with ``HYPERTOOLS_AUTO_INSTALL`` set.
    """
    env = dict(os.environ if env is None else env)
    env['HYPERTOOLS_AUTO_INSTALL'] = '1' if auto_install_enabled() else '0'
    return env


class set_autoinstall:
    """
    Turn the on-demand installation of optional extras on or off.

    hypertools' optional features (the plotly backend, text embeddings,
    the ``Laplace`` and ``Chronos`` forecasters, the torch autoencoders,
    gensim models, Kaggle and Hugging Face loading, LSL streaming, 3-D
    density iso-surfaces, ``.xlsx`` files) are ``pip`` extras that install
    themselves on demand: the first call that needs a missing one installs
    that extra's requirements into the running interpreter, prints a
    one-line ``hypertools:`` notice, and carries on. Static image export
    with the plotly backend provisions kaleido's Chrome the same way.

    Like `hypertools.set_interactive_backend`, this can be used in two
    ways:

    1. directly, to change the setting for the rest of the session::

           import hypertools as hyp

           hyp.set_autoinstall(False)
           hyp.plot(data, backend='plotly')   # ImportError if plotly is
                                              # missing, naming the manual
                                              # pip install command
           hyp.set_autoinstall(True)          # back on

    2. as a context manager with the `with` statement, to change it for
       one block::

           with hyp.set_autoinstall(False):
               hyp.predict(data, model='Chronos', t=5)   # no install here

           hyp.predict(data, model='Chronos', t=5)       # installs on demand

    With installation off, a call that needs a missing extra raises
    ``ImportError`` naming the manual ``pip install "hypertools[<extra>]"``
    command, and nothing is installed. Turn it off in locked-down
    environments and anywhere pip should not run inside a Python process.
    For a process where no Python runs before hypertools is imported (a CI
    image built ahead of time), the environment variable
    ``HYPERTOOLS_AUTO_INSTALL=0`` sets the starting value; a
    `set_autoinstall` call overrides it.

    The setting is process-global: it is shared by every thread, and a
    subprocess that renders plotly animation frames inherits the effective
    value. The newest call still in force decides: a `with` block removes
    its own setting on exit and leaves any other block that is still open
    in force, so two threads each inside ``with set_autoinstall(False)``
    both keep installation off until the LAST of them exits, whichever
    order they finish in. A direct call stays in force until the next call.

    Parameters
    ----------
    enabled : bool, default True
        ``True`` to install missing extras on demand, ``False`` to raise
        ``ImportError`` instead. Applies temporarily when used as a context
        manager with `with`, or for the life of the interpreter when called
        as a function.

    Attributes
    ----------
    enabled : bool
        The value that was set.

    Raises
    ------
    TypeError
        If `enabled` is not ``True`` or ``False``.
    """

    def __init__(self, enabled=True):
        if not isinstance(enabled, bool):
            raise TypeError(
                f'set_autoinstall expects True or False, got {enabled!r}')
        self.enabled = enabled
        with _AUTO_INSTALL_LOCK:
            _AUTO_INSTALL_SCOPES.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # remove THIS setting wherever it sits: a block that is not the
        # newest (another thread's block opened after it) must not restore
        # a value from before that other block
        with _AUTO_INSTALL_LOCK:
            for i in range(len(_AUTO_INSTALL_SCOPES) - 1, -1, -1):
                if _AUTO_INSTALL_SCOPES[i] is self:
                    del _AUTO_INSTALL_SCOPES[i]
                    break

    def __repr__(self):
        return f'set_autoinstall({self.enabled})'


def extra_requirements(extra):
    """The requirement strings pyproject declares for ``extra``, read from the
    installed hypertools metadata (e.g. ``['plotly>=6.1.1', 'kaleido>=1.0']``
    for ``'interactive'``)."""
    found = []
    for req in metadata.requires('hypertools') or []:
        if ';' not in req:
            continue
        spec, marker = req.split(';', 1)
        if f'extra == "{extra}"' in marker or f"extra == '{extra}'" in marker:
            found.append(spec.strip())
    if not found:
        raise ValueError(f'hypertools declares no optional extra named {extra!r}')
    return found


def install_command(extra):
    """The manual command for ``extra``, for error messages."""
    return f'pip install "hypertools[{extra}]"'


def _notice(text):
    print(f'hypertools: {text}', flush=True)


def _pip_install(requirements):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *requirements],
                   check=True)
    importlib.invalidate_caches()


def lazy_import(module, purpose=None, extra=None, requirements=None):
    """Import ``module``, installing the extra that provides it first if it
    is missing.

    Parameters
    ----------
    module : str
        The import name (``'plotly'``, ``'skimage'``, ``'chronos'``, or a
        dotted path such as ``'plotly.io'``).
    purpose : str, optional
        What needs it, for the notice and the error (``'the plotly
        backend'``).
    extra : str, optional
        The hypertools extra to install; defaults to the map above.
    requirements : list of str, optional
        Explicit pip requirements instead of the extra's (used by the tests
        and for packages hypertools does not declare).

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        When the module is missing and cannot be installed (auto-install
        disabled, no network, no permission, or no extra provides it); the
        message carries the manual command.
    """
    try:
        return importlib.import_module(module)
    except ImportError as first:
        top = module.split('.')[0]
        extra = extra or EXTRA_FOR_MODULE.get(top)
        if requirements is None and extra is not None:
            requirements = extra_requirements(extra)
        need = f' (needed for {purpose})' if purpose else ''
        manual = (install_command(extra) if extra
                  else f'pip install {" ".join(requirements)}' if requirements
                  else None)
        if requirements is None:
            raise ImportError(
                f'{module} is not installed{need}, and hypertools declares no '
                'extra that provides it.') from first
        if not auto_install_enabled():
            raise ImportError(
                f'{module} is not installed{need}. Install it with `{manual}` '
                '(automatic installation is off; hypertools.set_autoinstall(True) '
                'turns it on).'
            ) from first
        _notice(f'installing {", ".join(requirements)}{need} ...')
        try:
            _pip_install(requirements)
            return importlib.import_module(module)
        except (subprocess.CalledProcessError, ImportError) as second:
            raise ImportError(
                f'{module} is not installed{need}, and installing it '
                f'automatically failed ({type(second).__name__}). Install it '
                f'with `{manual}` and try again.') from second


def _kaleido_can_render():
    pio = importlib.import_module('plotly.io')
    go = importlib.import_module('plotly.graph_objects')
    try:
        pio.to_image(go.Figure(), format='png')
        return True, None
    except RuntimeError as e:
        if 'chrome' not in str(e).lower():
            raise
        return False, e


def _apt_install(packages):
    """Install Debian packages when this process can (root, or password-less
    sudo); return True if the install ran."""
    if not sys.platform.startswith('linux') or shutil.which('apt-get') is None:
        return False
    cmd = ['apt-get', 'install', '-qq', '-y', *packages]
    if hasattr(os, 'geteuid') and os.geteuid() != 0:
        if shutil.which('sudo') is None:
            return False
        cmd = ['sudo', '-n', *cmd]
    _notice(f'installing the system libraries Chrome needs ({" ".join(packages)}) ...')
    env = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
    try:
        # no inherited stdin (a dpkg prompt must never wait on a kernel's
        # stdin) and a hard ceiling, so a wedged apt cannot hang the caller
        subprocess.run(cmd, check=True, env=env, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=APT_TIMEOUT_SECONDS)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def ensure_kaleido_chrome():
    """Make plotly's static image export (kaleido) able to render, installing
    what is missing on demand. Idempotent; the check runs once per process.

    Raises
    ------
    HypertoolsIOError
        When no working Chrome could be provided; the message says what to do.
    """
    global _kaleido_ready
    if _kaleido_ready:
        return
    lazy_import('kaleido', purpose='plotly static image export')
    pio = importlib.import_module('plotly.io')
    ok, err = _kaleido_can_render()
    if not ok and auto_install_enabled():
        _apt_install(CHROME_APT_PACKAGES)
        _notice('kaleido found no Chrome it can run; downloading one for it (about 150 MB) ...')
        try:
            pio.get_chrome()
        except Exception as e:      # reported below with the render error
            err = e
        ok, err2 = _kaleido_can_render()
        err = err2 or err
    if not ok:
        from .exceptions import HypertoolsIOError
        raise HypertoolsIOError(
            "plotly's static image export (kaleido) needs a Chrome/Chromium "
            "binary and found none it can run. Fetch one for kaleido with "
            "`import plotly.io as pio; pio.get_chrome()` (about 150 MB); on "
            "Debian/Ubuntu images (Colab, Kaggle) also install the libraries "
            f"Chrome needs: `apt-get install -y {' '.join(CHROME_APT_PACKAGES)}`. "
            "Or install Chrome, or save the figure with the matplotlib "
            f"backend (backend='matplotlib'). kaleido said: {err}")
    _kaleido_ready = True
