# -*- coding: utf-8 -*-
"""A declared dependency floor must be installable on every Python the
package claims (review 2026-09-11).

``density3d = ["scikit-image>=0.23.2"]`` (and the same pin in ``[dev]`` and
``docs/doc_requirements.txt``) named a release with no CPython 3.13 wheel,
while the classifiers list 3.13: a lowest-version resolve on 3.13 (``uv pip
install --resolution lowest``) had to build scikit-image from source. The
first release with cp313 wheels is 0.25.0 (measured on PyPI's JSON API,
2026-09-11: 0.23.x and 0.24.0 ship cp310-cp312 only).

This is a REAL query of PyPI's JSON API for the exact floor version, wrapped
in ``skip_on_transient_network`` so an outage skips while a wrong floor
fails.
"""
import re
from pathlib import Path

import pytest
import requests

from tests._netskip import skip_on_transient_network

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
DOC_REQUIREMENTS = (REPO / 'docs' / 'doc_requirements.txt').read_text(
    encoding='utf-8')


def _classifier_pythons():
    found = re.findall(r'Programming Language :: Python :: (3\.\d+)"',
                       PYPROJECT)
    assert found, 'no Python version classifiers found in pyproject.toml'
    return found


def _declared_floors(package):
    """Every ``<package>>=X`` floor pyproject and the docs requirements
    declare (quoted in pyproject, bare in the requirements file)."""
    name = re.escape(package)
    floors = re.findall(rf'"{name}>=([0-9][0-9.]*)', PYPROJECT)
    floors += re.findall(rf'^{name}>=([0-9][0-9.]*)', DOC_REQUIREMENTS, re.M)
    assert floors, f'{package} is not declared with a >= floor'
    return sorted(set(floors))


def _wheel_filenames(package, version):
    resp = requests.get(f'https://pypi.org/pypi/{package}/{version}/json',
                        timeout=30)
    resp.raise_for_status()
    return [f['filename'] for f in resp.json()['urls']
            if f['packagetype'] == 'bdist_wheel']


def test_scikit_image_floor_has_a_wheel_for_every_claimed_python():
    pythons = _classifier_pythons()
    assert '3.13' in pythons
    floors = _declared_floors('scikit-image')
    # pyproject's [density3d] and [dev] and the docs requirements agree
    assert len(floors) == 1, floors
    with skip_on_transient_network('querying PyPI for scikit-image wheels'):
        wheels = _wheel_filenames('scikit-image', floors[0])
    for py in pythons:
        tag = f'-cp{py.replace(".", "")}-'
        assert any(tag in w for w in wheels), (
            f'scikit-image {floors[0]} (the declared floor) has no {py} '
            f'wheel; wheels: {sorted(wheels)}')


# Core runtime floors (review 2026-10): each was raised to the first release
# with cp313 wheels (numpy 2.0.x, pandas 2.2.2, scipy 1.13/1.14.0, matplotlib
# 3.9.0/3.9.1, scikit-learn <=1.5.1, statsmodels 0.14.2 and pillow 10.3.0
# ship cp310-cp312 only; measured on PyPI's JSON API, 2026-10-08).
CORE_COMPILED = ['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn',
                 'statsmodels', 'pillow']


@pytest.mark.parametrize('package', CORE_COMPILED)
def test_core_floor_has_a_wheel_for_every_claimed_python(package):
    pythons = _classifier_pythons()
    assert '3.13' in pythons
    floors = _declared_floors(package)
    # pyproject and the docs requirements (where listed) agree
    assert len(floors) == 1, floors
    with skip_on_transient_network(f'querying PyPI for {package} wheels'):
        wheels = _wheel_filenames(package, floors[0])
    for py in pythons:
        tag = f'-cp{py.replace(".", "")}-'
        assert any(tag in w for w in wheels), (
            f'{package} {floors[0]} (the declared floor) has no {py} '
            f'wheel; wheels: {sorted(wheels)}')


def test_readme_requirements_match_pyproject_core_floors():
    readme = (REPO / 'readme.md').read_text(encoding='utf-8')
    for package in CORE_COMPILED:
        (floor,) = _declared_floors(package)
        assert f'{package}>={floor}' in readme, (
            f'readme.md Requirements does not list {package}>={floor}')
