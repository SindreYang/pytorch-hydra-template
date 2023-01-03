#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='src',
    version='1.0.0',
    description='项目描述',
    author='sindre',
    author_email='yx@mviai.com',
    url='项目地址',
    install_requires=["pytorch","hydra-core"],
    packages=find_packages(),
)

