# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

os.environ["MPLCONFIGDIR"] = "."


class PostInstall(install):
    github_pkg = 'https://api.github.com/repos/lmcinnes/umap/tarball/5f9488a9540d1e0ac149e2dd42ebf03c39706110#egg=umap_learn'

    def run(self):
        install.run(self)
        output = subprocess.run([sys.executable, '-m', 'pip', 'install', self.github_pkg],
                                stdout=subprocess.PIPE)
        print(output.stdout.decode('utf-8'))


NAME = 'hypertools'
VERSION = '0.6.2'
AUTHOR = 'Contextual Dynamics Lab'
AUTHOR_EMAIL = 'contextualdynamics@gmail.com'
URL = 'https://github.com/ContextLab/hypertools'
DOWNLOAD_URL = URL
LICENSE = 'MIT'
REQUIRES_PYTHON = '>=3.5'
PACKAGES = find_packages(exclude=('images', 'examples', 'tests'))
with open('requirements.txt', 'r') as f:
    REQUIREMENTS = f.read().splitlines()

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
    'Operating System :: MacOS'
]
CMDCLASS = {
    'install': PostInstall
}


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
    classifiers=CLASSIFIERS,
    cmdclass=CMDCLASS,
)
