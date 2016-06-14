# -*- coding: utf-8 -*-

import six
import numpy
import pandas
import itertools

from masalachai import DataFeeder

class AutoencodeFeeder(DataFeeder):
    def __init__(self, data_dict=None):
        super(AutoencodeFeeder, self).__init__(data_dict)

    def batch(self, batchsize, shuffle=True):
        perm = numpy.random.permutation(self.n) if shuffle else numpy.arange(self.n)
        gen = itertools.cycle(perm)
        while True:
            indexes = [gen_f.next() for b in six.moves.range(0, batchsize)]
            yield {'data': (self.data_dict['data'][indexes],), 'target': (None,)}

