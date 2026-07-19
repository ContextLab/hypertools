# -*- coding: utf-8 -*-
"""Regression gate for GH #276 (docstring sweep): every public (non-
underscore-prefixed) function/method/class defined anywhere in the
``hypertools`` package -- EXCEPT ``hypertools/_externals/`` (vendored,
third-party code we do not own/maintain) -- must have a non-empty
docstring.

This is a real AST scan of the source tree, not a snapshot of a known-
good count: it will fail the moment anyone adds a new undocumented
public ``def``/``class`` anywhere in the package (including nested
functions/methods, which the round17 Task 16 sweep also documented, to
match how the originating GH #276 audit counted things).
"""
import ast
import os

import hypertools

PACKAGE_ROOT = os.path.dirname(os.path.abspath(hypertools.__file__))
EXCLUDED_DIR_NAMES = {'_externals'}


def _iter_python_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded/hidden directories in-place so os.walk never
        # descends into them
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIR_NAMES and not d.startswith('.')
            and d != '__pycache__'
        ]
        for fname in filenames:
            if fname.endswith('.py'):
                yield os.path.join(dirpath, fname)


def _find_undocumented(root):
    """Return a list of ``"relative/path.py:lineno: kind qualified.name"``
    strings for every public def/class in `root` (excluding
    `EXCLUDED_DIR_NAMES`) whose docstring is missing or empty."""
    missing = []

    for fpath in _iter_python_files(root):
        rel_path = os.path.relpath(fpath, root)
        with open(fpath, 'r', encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src, filename=fpath)

        def visit(node, class_stack):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = child.name
                    if not name.startswith('_'):
                        doc = ast.get_docstring(child)
                        if doc is None or doc.strip() == '':
                            qual = '.'.join(class_stack + [name])
                            kind = 'method' if class_stack else 'function'
                            missing.append(
                                f'{rel_path}:{child.lineno}: {kind} {qual}')
                    # recurse regardless, to reach nested defs/classes
                    visit(child, class_stack)
                elif isinstance(child, ast.ClassDef):
                    name = child.name
                    if not name.startswith('_'):
                        doc = ast.get_docstring(child)
                        if doc is None or doc.strip() == '':
                            qual = '.'.join(class_stack + [name])
                            missing.append(
                                f'{rel_path}:{child.lineno}: class {qual}')
                    visit(child, class_stack + [name])
                elif isinstance(child, (ast.If, ast.Try, ast.With)):
                    # module-level control flow (e.g. try/except optional
                    # imports) can itself contain top-level defs/classes
                    visit(child, class_stack)

        visit(tree, [])

    return sorted(missing)


def test_no_undocumented_public_definitions():
    """AST-scan the whole hypertools package (excluding `_externals/`) and
    assert there are ZERO public function/method/class definitions with a
    missing or empty docstring (GH #276)."""
    missing = _find_undocumented(PACKAGE_ROOT)
    assert missing == [], (
        f'{len(missing)} public def(s)/class(es) outside _externals/ are '
        f'missing a docstring (GH #276 regression gate):\n'
        + '\n'.join(missing)
    )


def test_manip_dispatcher_has_docstring():
    """`hyp.manip` specifically had `__doc__ is None` per the original GH
    #276 audit; confirm it now has a real (non-trivial) docstring."""
    assert hypertools.manip.__doc__ is not None
    assert hypertools.manip.__doc__.strip() != ''
    assert len(hypertools.manip.__doc__.strip()) > 20
