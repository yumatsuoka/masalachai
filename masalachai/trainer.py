import time
import numpy
import six
import chainer
import queue
import threading
from chainer import cuda

from masalachai.logger import Logger
from masalachai.datafeeder import DataFeeder


def res_dic_add(dic0, dic1):
    dic1_items = dic1.items()
    for k, v in dic1_items:
        if k in dic0:
            if k is not None and dic0[k] is not None:
                #dic[k] += v
                dic0[k] += v
        else:
            dic0[k] = v
    return dic0

def res_dic_mul(dic, value):
    for k in dic.keys():
        if dic[k] is not None:
            dic[k] *= value
    return dic

class Trainer(object):
    queue_size = 3

    def __init__(self, optimizer, logger, train_data, test_data=None, gpu=-1):
        self.optimizer = optimizer
        self.gpu = gpu

        # for train data feeder
        self.train_data = train_data
        self.train_data_queue = queue.Queue(self.queue_size)

        # for test data feeder
        if test_data is not None:
            self.test_data = test_data
            self.test_data_queue = queue.Queue(self.queue_size)

        # for logger
        self.logger = logger
        self.log_queue = queue.Queue()
        self.logger.setQueue(self.log_queue)

        self._optimizer_param_schedulers = []


    def add_optimizer_scheduler(self, s):
        # Optimizer のパラメータスケジューラの登録
        self._optimizer_param_schedulers.append(s)


    def optimizer_param_process(self, t):
        # Optimizer のパラメータスケジューラの駆動
        for s in self._optimizer_param_schedulers:
            self.optimizer.__dict__[s.param_name] = s.next(t)


    def wait_train_data_queue(self, t=0.05):
        while self.train_data_queue.empty():
            time.sleep(t)

    def wait_test_data_queue(self, t=0.05):
        while self.test_data_queue.empty():
            time.sleep(t)


    def predict(self, batchsize, train=False):
        # This method will be called when TEST phase
        raise NotImplementedError


    def update(self, batchsize):
        # This method will be called when TRAIN phase
        raise NotImplementedError


    def train(self, nitr, batchsize, 
              log_interval=1, 
              test_interval=1, test_nitr=1, test_batchsize=1):

        # setting batchsize of datafeeders
        self.train_data.batchsize = batchsize
        if self.test_data is not None:
            self.test_data.batchsize = test_batchsize
        
        # setting stop event for data feeder
        stop_feeding = threading.Event()
        self.train_data.generateThread(self.train_data_queue, stop_feeding)

        # start threads
        self.train_data.thread.start()
        self.logger.start()

        self.wait_train_data_queue()
        train_res = self.update(batchsize)
        for i in six.moves.range(1, int(nitr)):
            # update
            self.wait_train_data_queue()
            #train_res = res_dic_add(train_res, self.update(batchsize))
            train_res = res_dic_add(train_res, self.update(batchsize))
            self.optimizer_param_process(i)

            # logging
            if i % log_interval == 0:
                #train_res = res_dic_mul(train_res, float(batchsize)/log_interval)
                train_res = res_dic_mul(train_res, 1./log_interval)
                train_res['iteration'] = i
                self.log_queue.put(self.logger.train_log_mode)
                self.log_queue.put(train_res)
                train_res = {}
            # test
            if self.test_data is not None and i % test_interval == 0:
                test_res = self.test(test_nitr, test_batchsize)
                self.log_queue.put(self.logger.test_log_mode)
                self.log_queue.put(test_res)

        # end of training
        stop_feeding.set()

        # clearn data queue
        while not self.train_data_queue.empty():
            self.train_data_queue.get()

        self.log_queue.put('END')
        self.train_data.thread.join()
        self.logger.join()


    def test(self, nitr, batchsize):

        # setting stop event for data feeder
        stop_feeding = threading.Event()
        self.test_data.generateThread(self.test_data_queue, stop_feeding)

        # start threads
        self.test_data.thread.start()

        self.wait_train_data_queue()
        test_res = self.predict(batchsize)
        for i in six.moves.range(1, int(nitr)):
            self.wait_test_data_queue()
            test_res_ = res_dic_add(test_res, self.predict(batchsize))
        test_res = res_dic_mul(test_res, 1./nitr)

        # end of testing
        stop_feeding.set()

        # clearn data queue
        while not self.test_data_queue.empty():
            self.test_data_queue.get()

        self.test_data.thread.join()
        return test_res

