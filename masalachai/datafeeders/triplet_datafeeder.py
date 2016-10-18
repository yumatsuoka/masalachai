# -*- coding: utf-8 -*-

import six
import multiprocessing
import numpy
import itertools

import masalachai
from masalachai.datafeeder import DataFeeder

def triplet_preprocess(inp):
    data = {}
    data0, data1, data2 = inp
    data['data0'] = data0['data']
    data['data1'] = data1['data']
    data['data2'] = data2['data']
    return data

class TripletFeeder(DataFeeder):
    def __init__(self, data_dict, batchsize=1, shuffle=True, loaderjob=8):
        super(TripletFeeder, self).__init__(data_dict, batchsize, shuffle, loaderjob)
        self.hook_preprocess(triplet_preprocess)
        self.xa = None
        self.xp = None
        self.xn = None

    def run(self):
        pool = multiprocessing.Pool(self.loaderjob)
        while not self.stop.is_set():
            n_class = len(list(set(self.data_dict['target'])))
            nl = len(self.data_dict['data'])
            nl_class = [len(numpy.where(self.data_dict['target'] == c)[0]) 
                    for c in range(n_class)]
            self.xa = numpy.asarray([self.data_dict['data'][idx] 
                for c in six.moves.range(n_class) for i in range(nl - nl_class[c]) 
                for idx in numpy.random.permutation(
                    numpy.where(self.data_dict['target'] == c)[0])])
            self.xp = numpy.asarray([self.data_dict['data'][idx]
                for c in six.moves.range(n_class) for i in range(nl - nl_class[c])
                for idx in numpy.random.permutation(
                    numpy.where(self.data_dict['target'] == c)[0])])
            self.xn = numpy.asarray([self.data_dict['data'][idx]
                for c in six.moves.range(n_class) for i in range(nl_class[c])
                for idx in numpy.random.permutation(
                    numpy.where(self.data_dict['target'] != c)[0])])
            self.n = len(self.xa)
            perm = numpy.random.permutation(self.n) if self.shuffle \
                    else numpy.arange(self.n)
            gen = itertools.cycle(perm)
            cnt = 0
            while cnt < self.n and not self.stop.is_set():
                indexes = [next(gen) for b in six.moves.range(0, self.batchsize)]
                data_dict_list0 = self.get_data_dict_list(indexes, {'data':self.xa})
                data_dict_list1 = self.get_data_dict_list(indexes, {'data':self.xp})
                data_dict_list2 = self.get_data_dict_list(indexes, {'data':self.xn})
                batch_pool = []
                for i in range(len(indexes)):
                    batch_pool.append(
                            pool.apply_async(
                                masalachai.datafeeder.preprocess, (
                                    self.preprocess_hooks, (
                                        data_dict_list0[i],
                                        data_dict_list1[i],
                                        data_dict_list2[i]))))
                self.queue.put(self.get_data_dict_from_list(
                    [p.get() for p in batch_pool]))
                cnt += self.batchsize
        pool.close()
        pool.join()
    def get_data_dict_list(self, indexes, temp_dic):
        return [{k : v[i] if getattr(v,'__iter__',False) and len(v)==self.n \
                else v for k,v in temp_dic.items()} for i in indexes]
