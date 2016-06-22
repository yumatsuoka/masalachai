# -*- coding: utf-8 -*-

import six
import threading
import multiprocessing
import numpy
import pandas
import itertools

def from_csv(filename):
    df = pandas.read_csv(fname)
    return DataFeeder(df.to_dict())

class DataFeeder(threading.Thread):
    # data_dict has following keys: 'data' and 'target'.
    def __init__(self, data_dict, shuffle=True, loaderjob=8):
        super(DataFeeder, self).__init__()

        self.data_dict = data_dict
        self.n = len(data_dict['data'])

        self.shuffle = shuffle
        self.loaderjob = loaderjob

        preprocess_hooks = []
        self.queue = None
        self.stop = None


    def __getitem__(self, index):
        return self.data_dict[index]


    def hook_preprocess(self, func):
        self.preprocess_hooks.append(func)


    def preprocess(self, data):
        for p in self.preprocess_hooks:
            data = p(data)
        return data


    def setQueue(queue):
        self.queue = queue


    def setEvent(event):
        self.stop = stop


    def run(self):
        pool = multiprocessing.Pool(self.loaderjob)
        batch_pool = []
        while not self.stop.is_set():
            perm = numpy.random.permutation(self.n) if self.shuffle else numpy.arange(self.n)
            gen = itertools.cycle(perm)
            cnt = 0
            while cnt < self.n and not self.stop.is_set():
                indexes = [gen.next() for b in six.moves.range(0, batchsize)]
                for i in indexes:
                    batch_pool.append(pool.apply_async(preprocess, (self.data_dict['data'][i])))

                batch_array = np.asarray([p.get() for p in batch_pool])
                self.queue.put({'data': (batch_array,), 'target': (self.data_dict['target'][indexes],)})
                cnt += batchsize
        pool.close()
        pool.join()


    def batch(self, batchsize):
        while True:
            perm = numpy.random.permutation(self.n) if self.shuffle else numpy.arange(self.n)
            gen = itertools.cycle(perm)
            cnt = 0
            while cnt < self.n:
                indexes = [gen.next() for b in six.moves.range(0, batchsize)]
                cnt += batchsize
                yield {'data': (self.data_dict['data'][indexes],), 'target': (self.data_dict['target'][indexes],)}
