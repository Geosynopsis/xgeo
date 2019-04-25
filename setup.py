from os import path

from setuptools import (setup, find_packages)

root = path.abspath(path.dirname(__file__))
with open(path.join(root, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='xgeo',
    version='0.0.1',
    url='',
    author='Abhishek Manandhar, Stefan Kirmaier',
    description='Utility library for geo-rasters that uses xarray and rasterio.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'xarray>=0.11.3',
        'pandas>=0.24.1'
        'rasterio>=1.0.18',
        'geopandas>=0.4.0',
        'netCDF4>=1.4.2',
        'shapely>=1.6.4'
    ],
    test_requires=['pytest>=4.3.0'],
    test_suite='',
    packages=find_packages(exclude=['tests']),
    zip_safe=False,
    include_package_data=True,
    platform='any',
    classifiers=[
        'Operating System: OS Independent',
        'Programming Language: Python',
        'Topic:: Software Development :: Libraries :: Python Modules'
    ]
)
