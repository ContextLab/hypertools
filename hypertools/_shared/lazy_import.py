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
branch install is never replaced by a PyPI release. Set
``HYPERTOOLS_AUTO_INSTALL=0`` to disable installation; the import then fails
with the manual command. Every install prints a one-line notice.

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
}

#: Debian/Ubuntu packages a downloaded Chrome needs and a fresh Colab/Kaggle
#: image lacks (ldd on the live runtime, 2026-09-04: libatk-1.0.so.0,
#: libatk-bridge-2.0.so.0, libatspi.so.0, libXcomposite.so.1).
CHROME_APT_PACKAGES = ('libatk1.0-0', 'libatk-bridge2.0-0', 'libatspi2.0-0',
                       'libxcomposite1')

_kaleido_ready = False


def auto_install_enabled():
    """True unless ``HYPERTOOLS_AUTO_INSTALL`` is set to 0/false/no/off."""
    return os.environ.get('HYPERTOOLS_AUTO_INSTALL', '1').strip().lower() \
        not in ('0', 'false', 'no', 'off')


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
                '(automatic installation is disabled by HYPERTOOLS_AUTO_INSTALL=0).'
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
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, OSError):
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
