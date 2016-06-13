# -*- coding: utf-8 -*-

import six
import numpy
import pandas
import itertools

from masalachai import DataFeeder

class SiameseFeeder(DataFeeder):
    def __init__(self, data_dict=None):
        super(SiameseFeeder, self).__init__(data_dict)

    def batch(self, batchsize, shuffle=True):
        perm_f = numpy.random.permutation(self.n) if shuffle else numpy.arange(self.n)
        perm_r = perm_f[::-1]
        gen_f = itertools.cycle(perm_f)
        gen_r = itertools.cycle(perm_r)
        while True:
            indexes_f = [gen_f.next() for b in six.moves.range(0, batchsize)]
            indexes_r = [gen_r.next() for b in six.moves.range(0, batchsize)]
            yield {'data': (self.data_dict['data'][indexes_f],self.data_dict['data'][indexes_r]), 'target': (numpy.array(self.data_dict['target'][indexes_f]==self.data_dict['target'][indexes_r], dtype=numpy.int32),)}
