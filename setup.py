# -*- coding: utf-8 -*-
import os
os.environ["MPLCONFIGDIR"] = "."

from setuptools import setup, find_packages

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
- Support for lists of Numpy arrays or Pandas dataframes
"""

LICENSE = 'MIT'

setup(
    name='hypertools',
    version='0.1.5',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='Contextual Dynamics Lab',
    author_email='contextualdynamics@gmail.com',
    url='https://github.com/ContextLab/hypertools',
    download_url = 'https://github.com/ContextLab/hypertools',
    license=LICENSE,
    packages=find_packages(exclude=('images', 'examples', 'tests')),
    install_requires=[
   'PPCA>=0.0.2',
   'scikit-learn>=0.18.1',
   'pandas>=0.18.0',
   'seaborn>=0.7.1',
   'matplotlib>=1.5.1',
   'scipy>=0.17.1',
   'numpy>=1.10.4',
   'future'
   ],
    classifiers=[
             'Intended Audience :: Science/Research',
             'Programming Language :: Python :: 2.7',
             'Programming Language :: Python :: 3.4',
             'Topic :: Scientific/Engineering :: Visualization',
             'Topic :: Multimedia :: Graphics',
             'Operating System :: POSIX',
             'Operating System :: Unix',
             'Operating System :: MacOS'],
)
