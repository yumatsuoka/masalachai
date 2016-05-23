#!/usr/bin/env python

from setuptools import setup

chainer_version = 'chainer>=1.8.2'
install_requires = [ chainer_version ]

setup(
        name = 'masalachai',
        version = '0.1.0',
        packages = ['masalachai'],
        description = '',
        author = 'Daiki Shimada',
        author_email = 'daiki.shimada.9g@stu.hosei.ac.jp',
        url = 'https://github.com/DaikiShimada/masalachai.git',
        install_requires = install_requires,
        license = 'http://www.apache.org/licenses/LICENSE-2.0'
    )
