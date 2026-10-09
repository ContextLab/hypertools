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

An extra that IS installed is also checked against the requirement
pyproject declares for it (`outdated_requirements`): every distribution of
the extra, not only the module asked for, because a feature needs them
together (kaleido 1.x cannot export with plotly < 6.1.1). A distribution
below its declared floor is upgraded through the same pip call, with a
notice naming the installed version and the requirement -- but only while
none of its modules is imported yet. An imported package cannot be swapped
under a running interpreter, so in that case, and whenever installation is
off, the call raises ``ImportError`` naming the installed version, the
requirement and the command. The check needs the ``packaging`` library,
which is not a declared hypertools dependency of its own: matplotlib (a
core dependency) requires it, so it is present in every resolved install;
where it is not, the check is skipped rather than guessed at. A passed
check is remembered per (module, extra), so it runs once per process.

``ensure_kaleido_chrome()`` provisions what plotly's static image export
needs at run time: a Chrome build for kaleido and, on Linux images that
lack them (a fresh Colab or Kaggle kernel, measured 2026-09-04), the four
shared libraries that Chrome needs to start.
"""

import bisect
import functools
import importlib
import os
import shutil
import subprocess
import sys
import threading
import weakref
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

#: (top-level module, extra, explicit requirements or None) keys whose
#: installed versions were checked against the declared requirements and
#: whose module imported: `lazy_import` returns these without reading any
#: metadata again. Only a PASSED check is remembered.
_VERSIONS_VERIFIED = set()

#: the `set_autoinstall` handles that are alive, oldest first, as `_Scope`
#: records (a weak reference each), plus the BASELINE: the value the newest
#: direct call left in force (None: the environment decides). The newest
#: live record decides; a handle that dies without having entered a block
#: was a direct call, so its weakref callback folds its value into the
#: baseline and drops its record at once (nothing is retained: Codex round
#: 7); a handle that is alive but not yet entered is never touched (a
#: construct-then-enter race across threads: Codex round 8); a block removes
#: only its own record on exit. Process-global, guarded by
#: `_AUTO_INSTALL_LOCK`.
_AUTO_INSTALL_SCOPES = []
_AUTO_INSTALL_LOCK = threading.RLock()
_AUTO_INSTALL_BASELINE = [None, -1]     # [enabled or None, seq of that call]
_AUTO_INSTALL_SEQ = [0]


class _Scope:
    """One live `set_autoinstall` handle: its value, its construction order,
    whether it is inside its `with` block, and a weak reference to it."""
    __slots__ = ('enabled', 'seq', 'entered', 'finished', 'ref')

    def __init__(self, enabled, seq):
        self.enabled = enabled
        self.seq = seq
        self.entered = False
        self.finished = False      # its block has exited: it was never direct
        self.ref = None


def _scope_handle_died(scope, _ref=None):
    """Weakref callback (`_ref` is the dead weak reference, unused): the
    handle of `scope` is gone. Inside a block that
    cannot happen (the block holds it), so this was a direct call: the
    newest direct call's value is the baseline.

    A handle still alive when the interpreter exits dies during module
    teardown, after this module's globals have been cleared to None; there
    is no setting left to maintain then, so the callback does nothing
    rather than raise (which Python reports as "Exception ignored in ...";
    review 2026-09-11). The callback holds this function itself (see
    `set_autoinstall.__init__`), so it never looks the name up then."""
    if _AUTO_INSTALL_LOCK is None or _AUTO_INSTALL_SCOPES is None \
            or _AUTO_INSTALL_BASELINE is None:
        return                      # interpreter shutdown
    with _AUTO_INSTALL_LOCK:
        if scope.finished or scope.entered:
            return                  # a block's handle, not a direct call
        if scope.seq > _AUTO_INSTALL_BASELINE[1]:
            _AUTO_INSTALL_BASELINE[:] = [scope.enabled, scope.seq]
        for i in range(len(_AUTO_INSTALL_SCOPES) - 1, -1, -1):
            if _AUTO_INSTALL_SCOPES[i] is scope:
                del _AUTO_INSTALL_SCOPES[i]
                break


def auto_install_enabled():
    """True when hypertools may install a missing optional extra: what the
    newest `set_autoinstall` still in force set or, if none is, the environment
    variable ``HYPERTOOLS_AUTO_INSTALL`` (on unless it is 0/false/no/off)."""
    with _AUTO_INSTALL_LOCK:
        # the newest CALL decides: a live handle's record, or the baseline a
        # newer direct call folded in (an older handle kept in a variable
        # does not outrank a later direct call)
        top = _AUTO_INSTALL_SCOPES[-1] if _AUTO_INSTALL_SCOPES else None
        base_enabled, base_seq = _AUTO_INSTALL_BASELINE
        if top is not None and top.seq > base_seq:
            return top.enabled
        if base_enabled is not None:
            return base_enabled
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

    An extra that is installed but OLDER than the requirement hypertools
    declares (a notebook image with plotly 5 where ``plotly>=6.1.1`` is
    required) is upgraded the same way, with a notice naming the installed
    version and the requirement -- provided the old version has not been
    imported yet. Python cannot replace a package that is already imported,
    so in that case nothing is installed and the call raises ``ImportError``
    asking for the upgrade command and a restart.

    With installation off, a call that needs a missing extra raises
    ``ImportError`` naming the manual ``pip install "hypertools[<extra>]"``
    command, and nothing is installed. An extra that is installed but too
    old raises ``ImportError`` too, naming the installed version, the
    requirement and the same command. Turn it off in locked-down
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
    Calls are ordered by when ``set_autoinstall(...)`` was called, not by
    when a handle enters its block: ``with h:`` on a handle ``h`` created
    before a later call does not outrank that later call.

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
            _AUTO_INSTALL_SEQ[0] += 1
            self._scope = _Scope(enabled, _AUTO_INSTALL_SEQ[0])
            # the callback binds the function object now: at interpreter
            # shutdown a surviving handle dies after the module's globals
            # are cleared, when the name would resolve to None
            self._scope.ref = weakref.ref(
                self, functools.partial(_scope_handle_died, self._scope))
            _AUTO_INSTALL_SCOPES.append(self._scope)

    def __enter__(self):
        with _AUTO_INSTALL_LOCK:
            if self._scope.finished:          # re-entered after an exit
                self._scope.finished = False
                # back in CALL order (the list is oldest first): entering is
                # not a new call, so a re-entered handle ranks where its
                # construction put it, exactly as on its first entry -- not
                # on top of newer handles (review 2026-09-11)
                bisect.insort(_AUTO_INSTALL_SCOPES, self._scope,
                              key=lambda s: s.seq)
            self._scope.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # remove THIS setting wherever it sits: a block that is not the
        # newest (another thread's block opened after it) must not restore
        # a value from before that other block
        with _AUTO_INSTALL_LOCK:
            self._scope.entered = False
            self._scope.finished = True
            for i in range(len(_AUTO_INSTALL_SCOPES) - 1, -1, -1):
                if _AUTO_INSTALL_SCOPES[i] is self._scope:
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
    """The manual command for ``extra``, for error messages. It also
    UPGRADES: pip re-resolves the extra's requirements for an installed
    hypertools and replaces a distribution that is below its floor, without
    reinstalling hypertools itself (no ``-U``, which would)."""
    return f'pip install "hypertools[{extra}]"'


def _manual_command(extra, requirements):
    """The command a person runs instead: the extra's, or the explicit
    requirements quoted for a shell (an unquoted ``>=`` is a redirection)."""
    if extra:
        return install_command(extra)
    if requirements:
        return 'pip install ' + ' '.join(f'"{r}"' for r in requirements)
    return None


def _notice(text):
    print(f'hypertools: {text}', flush=True)


def _pip_install(requirements):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *requirements],
                   check=True)
    importlib.invalidate_caches()


def _meets_floor(installed, requirement):
    """False only when the version string ``installed`` is certainly BELOW
    the floor ``requirement`` declares.

    Only the floor-setting clauses count (``>=``, ``>``, ``~=``, and ``==``
    without a wildcard); an upper bound or an exclusion cannot make an
    installed version too old. Versions are compared by their release
    numbers, so a development, pre-release, post-release or local build of
    the floor version (``6.1.1.dev0``, ``6.1.1rc1``, ``6.1.1+local``) meets
    ``>=6.1.1``. Anything that carries no usable version -- a string that
    is not a version, an all-zero "unknown" placeholder such as
    ``0+unknown`` or ``0.0.0`` from an untagged source build, a requirement
    that does not parse, or no ``packaging`` library to parse with -- is
    treated as meeting the floor: never a false alarm.
    """
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
        req = Requirement(requirement)
        have = Version(Version(installed).base_version)
    except Exception:           # ImportError, InvalidRequirement, InvalidVersion
        return True
    if not any(have.release):
        return True
    for clause in req.specifier:
        if clause.version.endswith('.*'):
            continue
        try:
            floor = Version(Version(clause.version).base_version)
        except Exception:
            continue
        if clause.operator in ('>=', '~=', '==') and have < floor:
            return False
        if clause.operator == '>' and have <= floor:
            return False
    return True


def outdated_requirements(requirements):
    """The requirements whose INSTALLED distribution is below the declared
    floor.

    Each requirement string is looked up by the distribution name it
    declares (``scikit-image``, not the import name ``skimage``) in the
    installed package metadata. A distribution that is not installed is not
    "too old" (a missing module is `lazy_import`'s other branch), and
    neither is one whose marker does not apply here or whose version cannot
    be compared (see `_meets_floor`).

    Parameters
    ----------
    requirements : iterable of str
        Requirement strings, e.g. ``['plotly>=6.1.1', 'kaleido>=1.0']``.

    Returns
    -------
    list of (str, str, str)
        ``(requirement, distribution name, installed version)`` for each
        requirement the installed version does not meet, in order.
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:
        return []
    found = []
    for requirement in requirements or ():
        try:
            req = Requirement(requirement)
            if req.marker is not None and not req.marker.evaluate():
                continue
            installed = metadata.version(req.name)
        except Exception:       # unparseable, or the distribution is absent
            continue
        if not _meets_floor(installed, requirement):
            found.append((requirement, req.name, installed))
    return found


def _imported_modules_of(dist_name):
    """The top-level modules of the installed distribution ``dist_name`` that
    are imported in this process (``['_plotly_utils', 'plotly']``)."""
    try:
        dist = metadata.distribution(dist_name)
    except Exception:
        return []
    names = set((dist.read_text('top_level.txt') or '').split())
    if not names:
        for f in dist.files or ():
            first = f.parts[0] if f.parts else ''
            if len(f.parts) == 1 and first.endswith('.py'):
                first = first[:-3]
            if first.isidentifier():
                names.add(first)
    return sorted(n for n in names if n in sys.modules)


def _too_old_text(outdated):
    return '; '.join(f'{name} {have} is installed, but hypertools needs {req}'
                     for req, name, have in outdated)


def _upgrade_outdated(outdated, requirements, extra, need):
    """Bring the distributions in ``outdated`` (from `outdated_requirements`)
    up to their requirements, or raise ImportError saying exactly what is
    installed, what is required and what to run."""
    what = _too_old_text(outdated) + need
    manual = _manual_command(extra, requirements)
    imported = sorted({name for _req, name, _have in outdated
                       if _imported_modules_of(name)})
    restart = (' and restart Python (in a notebook, restart the kernel or '
               'runtime)')
    if not auto_install_enabled():
        raise ImportError(
            f'{what}. Upgrade with `{manual}`'
            + (f'{restart}, since {", ".join(imported)} is already imported '
               'in this process' if imported else '')
            + ' (automatic installation is off; '
            'hypertools.set_autoinstall(True) turns it on).')
    if imported:
        # an imported package cannot be replaced under a running
        # interpreter: its loaded modules stay the old ones and anything it
        # imports later comes from the new files. Nothing is installed.
        raise ImportError(
            f'{what}. {", ".join(imported)} is already imported in this '
            'Python process, so hypertools did not upgrade it in place. Run '
            f'`{manual}`{restart}.')
    _notice('upgrading '
            + ', '.join(f'{name} {have} to {req}' for req, name, have in outdated)
            + f'{need} ...')
    try:
        _pip_install(requirements)
    except subprocess.CalledProcessError as e:
        raise ImportError(
            f'{what}, and upgrading automatically failed '
            f'({type(e).__name__}). Run `{manual}` and try again.') from e
    still = outdated_requirements(requirements)
    if still:
        raise ImportError(
            f'{_too_old_text(still)}{need}, and upgrading automatically '
            f'left it in place. Run `{manual}` and try again.')


def lazy_import(module, purpose=None, extra=None, requirements=None):
    """Import ``module``, first installing the extra that provides it if it
    is missing, or upgrading it if what is installed is older than the
    requirement hypertools declares.

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
        disabled, no network, no permission, or no extra provides it), or
        when an installed distribution of the extra is below its declared
        requirement and cannot be upgraded (auto-install disabled, the old
        version already imported in this process, or pip failed). The
        message carries the manual command and, for a version problem, the
        installed version and the requirement.

    Notes
    -----
    Every requirement of the extra is checked, not only the distribution
    behind ``module``: ``lazy_import('kaleido')`` also verifies plotly,
    because static export needs the two together. The check reads the
    installed package metadata once; a passed check is remembered for the
    life of the process, so later calls only import. See
    `outdated_requirements` for what counts as too old.
    """
    top = module.split('.')[0]
    extra = extra or EXTRA_FOR_MODULE.get(top)
    key = (top, extra, None if requirements is None else tuple(requirements))
    if key in _VERSIONS_VERIFIED:
        try:
            return importlib.import_module(module)
        except ImportError:
            pass                    # e.g. a submodule: decided below
    need = f' (needed for {purpose})' if purpose else ''
    declared = requirements
    if declared is None and extra is not None:
        try:
            declared = extra_requirements(extra)
        except (ValueError, metadata.PackageNotFoundError):
            # an unknown extra, or hypertools run from a source tree with no
            # installed metadata: nothing to check an installed module
            # against (a MISSING module still reports this, below)
            declared = None
    outdated = outdated_requirements(declared) if declared else []
    if outdated:
        # before the import: an upgrade is only possible while the old
        # version is not imported yet
        _upgrade_outdated(outdated, declared, extra, need)
    try:
        imported = importlib.import_module(module)
    except ImportError as first:
        if requirements is None and extra is not None:
            requirements = extra_requirements(extra)
        manual = _manual_command(extra, requirements)
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
            imported = importlib.import_module(module)
        except (subprocess.CalledProcessError, ImportError) as second:
            raise ImportError(
                f'{module} is not installed{need}, and installing it '
                f'automatically failed ({type(second).__name__}). Install it '
                f'with `{manual}` and try again.') from second
    _VERSIONS_VERIFIED.add(key)
    return imported


def _installed_version(dist_name):
    try:
        return metadata.version(dist_name)
    except Exception:
        return '(unknown version)'


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
    except ValueError as e:
        # plotly's "Image export using the "kaleido" engine requires the
        # kaleido package" while kaleido IS importable: the two versions do
        # not work together (kaleido 1.x with plotly < 6.1.1; measured with
        # plotly 5.24.1 + kaleido 1.3.0, review 2026-10-09). Say that, not
        # "install kaleido". `lazy_import` catches the declared floors
        # first; this is for an environment it could not check.
        if 'kaleido' not in str(e).lower():
            raise
        from .exceptions import HypertoolsIOError
        raise HypertoolsIOError(
            f"plotly {_installed_version('plotly')} and kaleido "
            f"{_installed_version('kaleido')} are both installed but cannot "
            "export a static image together (hypertools needs "
            f"{', '.join(extra_requirements('interactive'))}). Bring them in "
            f"line with `{install_command('interactive')}` and restart "
            f"Python. plotly said: {str(e).strip()}") from e


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
