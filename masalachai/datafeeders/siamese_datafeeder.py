# -*- coding: utf-8 -*-

import six
import multiprocessing
import numpy
import itertools

import masalachai
from masalachai import DataFeeder

def siamese_preprocess(inp):
    data = {}
    data0, data1 = inp
    data['data0'] = data0['data']
    data['data1'] = data1['data']
    data['target'] = numpy.asarray(1).astype(numpy.int32) if data0['target']==data1['target'] else numpy.asarray(0).astype(numpy.int32)
    return data

class SiameseFeeder(DataFeeder):
    def __init__(self, data_dict, batchsize=1, shuffle=True, loaderjob=8):
        super(SiameseFeeder, self).__init__(data_dict, batchsize, shuffle, loaderjob)
        self.hook_preprocess(siamese_preprocess)

    def run(self):
        pool = multiprocessing.Pool(self.loaderjob)
        while not self.stop.is_set():
            perm0 = numpy.random.permutation(self.n) if self.shuffle else numpy.arange(self.n)
            perm1 = numpy.random.permutation(self.n) if self.shuffle else numpy.arange(self.n)
            gen0 = itertools.cycle(perm0)
            gen1 = itertools.cycle(perm1)
            cnt = 0
            while cnt < self.n and not self.stop.is_set():
                indexes0 = [next(gen0) for b in six.moves.range(0, self.batchsize)]
                indexes1 = [next(gen1) for b in six.moves.range(0, self.batchsize)]
                data_dict_list0 = self.get_data_dict_list(indexes0)
                data_dict_list1 = self.get_data_dict_list(indexes1)
                batch_pool = []
                for i in range(len(indexes0)):
                    batch_pool.append(
                            pool.apply_async(
                                masalachai.datafeeder.preprocess, 
                                (self.preprocess_hooks, (data_dict_list0[i], data_dict_list1[i]))
                            )
                    )

                self.queue.put(self.get_data_dict_from_list([p.get() for p in batch_pool]))
                cnt += self.batchsize
        pool.close()
        pool.join()

