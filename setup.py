#!/usr/bin/env python

from setuptools import setup

pandas_version = 'pandas'
six_version = 'six>=1.9.0'
chainer_version = 'chainer>=1.8.2'
install_requires = [ pandas_version, six_version, chainer_version ]

setup(
        name = 'masalachai',
        version = '0.3.10',
        packages = ['masalachai',
                    'masalachai.datafeeders',
                    'masalachai.preprocesses',
                    'masalachai.trainers',
                    'masalachai.models',
                    'masalachai.optimizer_schedulers'],
        description = 'Masala Chai: easy chainer tool',
        author = 'Daiki Shimada',
        author_email = 'daiki.shimada.9g@stu.hosei.ac.jp',
        url = 'https://github.com/DaikiShimada/masalachai.git',
        install_requires = install_requires,
        license = 'http://www.apache.org/licenses/LICENSE-2.0'
    )
