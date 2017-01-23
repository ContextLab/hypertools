# -*- coding: utf-8 -*-
import os
os.environ["MPLCONFIGDIR"] = "."

from setuptools import setup, find_packages

with open('readme.md') as f:
    readme = f.read()

# NEED TO ADD IN LICENSE
with open('LICENSE') as f:
    license = f.read()

setup(
    name='hypertools',
    version='0.1.0',
    description='A python package for visualizing high dimensional data',
    long_description=readme,
    author='Contextual Dynamics Lab',
    author_email='contextualdynamics@gmail.com',
    url='https://github.com/ContextLab/hypertools',
    license=license,
    packages=find_packages(exclude=('tests', 'examples', 'images')),
    install_requires=[
   'PPCA>=0.0.2',
   'scikit-learn>=0.18.1',
   'pandas>=0.18.0',
   'seaborn>=0.7.1',
   'matplotlib>=1.5.1',
   'scipy>=0.17.1',
   'numpy>=1.10.4'
   ]
)
