# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('readme.md') as f:
    readme = f.read()

# NEED TO ADD IN LICENSE
with open('LICENSE') as f:
    license = f.read()

setup(
    name='hypertools',
    version='1.0.0',
    description='A python package for visualizing high dimensional data',
    long_description=readme,
    author='Contextual Dynamics Lab',
    author_email='contextualdynamics@gmail.com',
    url='https://github.com/ContextLab/hypertools',
    license=license,
    packages=find_packages(exclude=('tests', 'examples', 'images'))
)
