# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

os.environ["MPLCONFIGDIR"] = "."

DESCRIPTION = 'A python package for visualizing and manipulating high-dimensional data'
LONG_DESCRIPTION = """
HyperTools is a library for visualizing and manipulating high-dimensional data in Python. It is built on top of
matplotlib (for plotting), seaborn (for plot styling), and scikit-learn (for data manipulation).

For sample Jupyter notebooks using the package: https://github.com/ContextLab/hypertools-paper-notebooks

For more examples: https://github.com/ContextLab/hypertools/tree/master/examples

Some key features of HyperTools are:

- Functions for plotting high-dimensional datasets in 2/3D.
- Static and animated plots
- Simple API for customizing plot styles
- A set of powerful data manipulation tools including hyperalignment, k-means clustering, normalizing and more.
- Support for lists of Numpy arrays, Pandas dataframes, String, Geos or mixed lists.
"""

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

with open('requirements_dev.txt') as requirements_dev_file:
    dev_requirements = requirements_dev_file.read()

test_requirements = ['pytest>=3']

setup(
    author="Contextual Dynamics Lab",
    author_email='contextualdynamics@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Multimedia :: Graphics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'],
    description=DESCRIPTION,
    install_requires=requirements,
    extras_require={'dev': dev_requirements},
    license="MIT license",
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords=['hypertools', 'python', 'visualization', 'graphics', 'plotly',
    'matplotlib', 'wrangling', 'animation', 'plotting', 'natural language processing'],
    name='hypertools',
    packages=find_packages(include=['hypertools', 'hypertools.*']),
    test_suite='tests',
    tests_require=dev_requirements,
    url='https://github.com/ContextLab/hypertools',
    version='0.8.0',
    zip_safe=False,
)