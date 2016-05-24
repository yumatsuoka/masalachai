# -*- coding: utf-8 -*-

import six
import numpy
import pandas
import itertools

class DataFeeder(object):
    def __init__(self, data_dict=None):
        self.data_dict = data_dict
        self.n = len(data_dict['data']) if self.data_dict is not None else None

    def load_from_csv(self, fname):
        df = pandas.read_csv(fname)
        self.data_dict = df.to_dict()
        self.n = len(df)

    def batch(self, batchsize, shuffle=True):
        perm = numpy.random.permutation(self.n) if shuffle else numpy.arange(self.n)
        gen = itertools.cycle(perm)
        while True:
            indexes = [gen.next() for b in six.moves.range(0, batchsize)]
            yield {'data': self.data_dict['data'][indexes], 'target': self.data_dict['target'][indexes]}
