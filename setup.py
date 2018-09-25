#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='xgeo',
    version='0.1.0',
    description="This library combines functionalities from xarray, rasterio/gdal and shapely/ogr for analysis and processing of geoscientific data",
    long_description=readme + '\n\n',
    author="Dirk Eilander",
    author_email='dirk.eilander@vu.nl',
    url='https://github.com//xgeo',
    packages=[
        'xgeo',
    ],
    package_dir={'xgeo':
                 'xgeo'},
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='xgeo',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    install_requires=[],  # FIXME: add your package's dependencies to this list
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'dev':  ['prospector[with_pyroma]', 'yapf', 'isort'],
    }
)
