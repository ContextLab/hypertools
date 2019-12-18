# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

os.environ["MPLCONFIGDIR"] = "."


def parse_dependencies(requirements_path, vcs_id, egg_id):
    requirements = []
    dependency_links = []
    with open(requirements_path, 'r') as f:
        reqs = f.read().splitlines()

    for req in reqs:
        if req.startswith(vcs_id) and egg_id in req:
            package_name = req[req.find(egg_id) + len(egg_id):]
            requirements.append(package_name)
            dependency_links.append(req)
        else:
            requirements.append(req)

    return requirements, dependency_links


NAME = 'hypertools'
VERSION = '0.6.1'
AUTHOR = 'Contextual Dynamics Lab'
AUTHOR_EMAIL = 'contextualdynamics@gmail.com'
URL = 'https://github.com/ContextLab/hypertools'
DOWNLOAD_URL = URL
LICENSE = 'MIT'
REQUIRES_PYTHON = '>=3'
PACKAGES = find_packages(exclude=('images', 'examples', 'tests'))
REQUIREMENTS, DEPENDENCY_LINKS = parse_dependencies('requirements.txt', 'git+', '#egg=')


DESCRIPTION = 'A python package for visualizing and manipulating high-dimensional data'
LONG_DESCRIPTION = """\
HyperTools is a library for visualizing and manipulating high-dimensional data in Python. It is built on top of matplotlib (for plotting), seaborn (for plot styling), and scikit-learn (for data manipulation).

For sample Jupyter notebooks using the package: https://github.com/ContextLab/hypertools-paper-notebooks

For more examples: https://github.com/ContextLab/hypertools/tree/master/examples

Some key features of HyperTools are:

- Functions for plotting high-dimensional datasets in 2/3D.
- Static and animated plots
- Simple API for customizing plot styles
- A set of powerful data manipulation tools including hyperalignment, k-means clustering, normalizing and more.
- Support for lists of Numpy arrays, Pandas dataframes, String, Geos or mixed lists.
"""
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Multimedia :: Graphics',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS']


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url = DOWNLOAD_URL,
    license=LICENSE,
    python_requires=REQUIRES_PYTHON,
    packages=PACKAGES,
    install_requires=REQUIREMENTS,
    dependency_links=DEPENDENCY_LINKS
    classifiers=CLASSIFIERS,
)
